/***
DEVSIM
Copyright 2013 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#include "ExternalPreconditioner.hh"
#include "CompressedMatrix.hh"
#include "dsAssert.hh"
#include "ObjectHolder.hh"
#include "Interpreter.hh"
#include "GlobalData.hh"
#include "NodeKeeper.hh"
#include "OutputStream.hh"
#include <utility>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <sstream>

namespace dsMath {
namespace
{
enum class CuDSSResultMode {
  HOST,
  DEVICE_EXPERIMENTAL,
};

CuDSSResultMode GetCuDSSResultMode()
{
  GlobalData &gdata = GlobalData::GetInstance();
  if (auto dbent = gdata.GetDBEntryOnGlobal("cudss_result_mode"); dbent.first)
  {
    const auto &val = dbent.second.GetString();
    if (val == "device_experimental")
    {
      return CuDSSResultMode::DEVICE_EXPERIMENTAL;
    }
  }
  return CuDSSResultMode::HOST;
}

const char *ToResultModeText(CuDSSResultMode mode)
{
  return (mode == CuDSSResultMode::DEVICE_EXPERIMENTAL) ? "device_experimental" : "host";
}
bool IsCuDSSProfileEnabled()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_PROFILE"))
  {
    const std::string val(v);
    return (val == "1") || (val == "true") || (val == "TRUE") || (val == "on") || (val == "ON");
  }
  return false;
}

void preswap(std::vector<double> &xin, std::vector<double> &xout)
{
  xin.swap(xout);
}
void convertToType(std::vector<double> &xin, std::vector<double> &xout)
{
  xout.swap(xin);
}
void preswap(ComplexDoubleVec_t<double> &xin, ComplexDoubleVec_t<double> &xout)
{
  xin.swap(xout);
}
void convertToType(ComplexDoubleVec_t<double> &xin, ComplexDoubleVec_t<double> &xout)
{
  xout.swap(xin);
}
}
}

#ifdef DEVSIM_EXTENDED_PRECISION
#include "Float128.hh"
namespace dsMath {
namespace
{
void preswap(std::vector<double> &xin, std::vector<float128> &xout)
{
}
void convertToType(std::vector<double> &xin, std::vector<float128> &xout)
{
  xout.resize(xin.size());
  std::copy(xin.begin(), xin.end(), xout.begin());
}
void preswap(ComplexDoubleVec_t<double> &xin, ComplexDoubleVec_t<float128> &xout)
{
}
void convertToType(ComplexDoubleVec_t<double> &xin, ComplexDoubleVec_t<float128> &xout)
{
  xout.resize(xin.size());
  std::transform(xin.begin(), xin.end(), xout.begin(), [](auto x){return static_cast<ComplexDouble_t<float128>>(x);});
}
}
}
#endif


namespace dsMath {
template <typename DoubleType>
ExternalPreconditioner<DoubleType>::ExternalPreconditioner(size_t sz, PEnum::TransposeType_t transpose)
  : Preconditioner<DoubleType>(sz, transpose),
    last_device_result_token_(""),
    last_device_result_location_("host"),
    result_generation_(0),
    profile_enabled_(IsCuDSSProfileEnabled())
{
  device_result_buffer_.valid = false;
  device_result_buffer_.backend = "none";
}

template <typename DoubleType>
dsMath::CompressionType ExternalPreconditioner<DoubleType>::GetRealMatrixCompressionType() const
{
  return compression_type_;
}

template <typename DoubleType>
dsMath::CompressionType ExternalPreconditioner<DoubleType>::GetComplexMatrixCompressionType() const
{
  return compression_type_;
}

template <typename DoubleType>
bool ExternalPreconditioner<DoubleType>::HasDeviceResult() const
{
  return !last_device_result_token_.empty();
}

template <typename DoubleType>
std::string ExternalPreconditioner<DoubleType>::GetDeviceResultToken() const
{
  return last_device_result_token_;
}

template <typename DoubleType>
std::string ExternalPreconditioner<DoubleType>::GetDeviceResultLocation() const
{
  return last_device_result_location_;
}

template <typename DoubleType>
ResultView<DoubleType> ExternalPreconditioner<DoubleType>::GetResultView() const
{
  ResultView<DoubleType> out;
  out.host_payload = &last_result_object_;
  out.device_result = device_result_buffer_.valid ? &device_result_buffer_ : nullptr;
  out.has_device_values = !last_device_result_token_.empty();
  out.device_token = last_device_result_token_;
  out.location = last_device_result_location_;
  return out;
}

template <typename DoubleType>
ExternalPreconditioner<DoubleType>::~ExternalPreconditioner()
{
  if (profile_enabled_)
  {
    std::ostringstream os;
    os << "cuDSS external profile:"
       << " factor_calls=" << factor_calls_
       << " solve_calls=" << solve_calls_
       << " factor_total_seconds=" << factor_total_seconds_
       << " factor_build_seconds=" << factor_build_seconds_
       << " factor_interpreter_seconds=" << factor_interpreter_seconds_
       << " factor_result_seconds=" << factor_result_seconds_
       << " factor_validate_seconds=" << factor_validate_seconds_
       << " factor_parse_seconds=" << factor_parse_seconds_
       << " solve_total_seconds=" << solve_total_seconds_
       << " solve_build_seconds=" << solve_build_seconds_
       << " solve_interpreter_seconds=" << solve_interpreter_seconds_
       << " solve_result_seconds=" << solve_result_seconds_
       << " solve_validate_seconds=" << solve_validate_seconds_
       << " solve_device_meta_seconds=" << solve_device_meta_seconds_
       << " solve_parse_seconds=" << solve_parse_seconds_
       << " solve_materialize_seconds=" << solve_materialize_seconds_
       << " solve_device_view_seconds=" << solve_device_view_seconds_ << "\n";
    for (size_t i = 0; i < factor_call_profiles_.size(); ++i)
    {
      const auto &call = factor_call_profiles_[i];
      os << "cuDSS external factor call:"
         << " call=" << (i + 1)
         << " total_seconds=" << call.total_seconds
         << " build_seconds=" << call.build_seconds
         << " interpreter_seconds=" << call.interpreter_seconds
         << " result_seconds=" << call.result_seconds
         << " validate_seconds=" << call.validate_seconds
         << " other_seconds=" << call.other_seconds << "\n";
    }
    for (size_t i = 0; i < solve_call_profiles_.size(); ++i)
    {
      const auto &call = solve_call_profiles_[i];
      os << "cuDSS external solve call:"
         << " call=" << (i + 1)
         << " total_seconds=" << call.total_seconds
         << " build_seconds=" << call.build_seconds
         << " interpreter_seconds=" << call.interpreter_seconds
         << " result_seconds=" << call.result_seconds
         << " validate_seconds=" << call.validate_seconds
         << " device_meta_seconds=" << call.device_meta_seconds
         << " materialize_seconds=" << call.materialize_seconds
         << " device_view_seconds=" << call.device_view_seconds
         << " need_host_vector=" << call.need_host_vector
         << " device_view_enabled=" << call.device_view_enabled
         << " other_seconds=" << call.other_seconds << "\n";
    }
    OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
  }
}

template <typename DoubleType>
bool ExternalPreconditioner<DoubleType>::init(ObjectHolder oh, std::string &error_string)
{
  device_result_buffer_.valid = false;
  device_result_buffer_.backend = "none";
  device_result_buffer_.copy_to_host_double = nullptr;
  device_result_buffer_.copy_rows_to_host_double = nullptr;
  last_device_result_token_.clear();
  last_device_result_location_ = "host";
  last_result_object_.clear();

  if (!oh.IsCallable())
  {
    error_string += "python solver object \"" + oh.GetString() + "\" is not callable\n";
    return false;
  }
  const std::string return_keys[] = {
    "matrix_format",
    "solver_object",
    "status",
    "message",
  };
  ObjectHolderMap_t init_args = {
    {"action", ObjectHolder("init")},
    {"transpose", ObjectHolder(this->GetTransposeSolve())},
    {"n", ObjectHolder(static_cast<int>(this->size()))},
  };

  command_handle_ = oh;
  Interpreter interpreter;
  bool ret = interpreter.RunCommand(command_handle_, init_args);
  if (!ret)
  {
    error_string = interpreter.GetErrorString();
  }
  else
  {
    ObjectHolderMap_t result_dictionary;
    auto result = interpreter.GetResult();
    ret = result.GetHashMap(result_dictionary);
    if (!ret)
    {
      error_string += "python solver object did not return a dictionary\n";
    }
    else
    {
      for (const auto &arg: return_keys)
      {
        if (auto it = result_dictionary.find(arg); it == result_dictionary.end())
        {
          error_string += "python solver object did not return a dictionary containing \"" + arg + "\"\n";
          ret = false;
        }
      }
      if (ret)
      {
        auto matrix_format = result_dictionary["matrix_format"].GetString();
        if (matrix_format == "csc")
        {
          compression_type_ = dsMath::CompressionType::CCM;
        }
        else if (matrix_format == "csr")
        {
          compression_type_ = dsMath::CompressionType::CRM;
        }
        else
        {
          error_string += R"(python solver object did not return a dictionary containing "csc" or "csr" for "matrix_format")" "\n";
          ret = false;
        }
        auto status = result_dictionary["status"].GetBoolean().second;
        auto message = result_dictionary["message"].GetString();
        error_string += message;
        dsAssert(status, error_string);
      }
      command_data_ = result_dictionary["solver_object"];
    }
  }
  return ret;
}

template <typename DoubleType>
bool ExternalPreconditioner<DoubleType>::DerivedLUFactor(Matrix<DoubleType> *m)
{
  const auto begin = std::chrono::steady_clock::now();
  FactorCallProfile call_profile;
  ++factor_calls_;
  dsAssert(command_handle_.IsCallable(), "python solver command is not callable\n");
  dsAssert(!command_data_.empty(), "python solver invalid data\n");

  CompressedMatrix<DoubleType> *cm = dynamic_cast<CompressedMatrix<DoubleType> *>(m);
  dsAssert(cm, "UNEXPECTED");
  dsAssert(cm->GetCompressionType() == compression_type_, "UNEXPECTED");

  const std::string return_keys[] = {
    "status",
    "message",
  };

  ObjectHolderMap_t factor_args = {
    {"action", ObjectHolder("factor")},
    {"solver_object", command_data_},
    {"Ap", CreateIntPODArray(cm->GetAp())},
    {"Ai", CreateIntPODArray(cm->GetAi())},
    {"is_same_symbolic", ObjectHolder(cm->GetSymbolicStatus() == SymbolicStatus_t::SAME_SYMBOLIC)},
  };

  if (cm->GetMatrixType() == MatrixType::COMPLEX)
  {
    const auto &ax = cm->GetAx();
    const auto &az = cm->GetAz();
    dsAssert(ax.size() == az.size(), "UNEXPECTED");
    std::vector<DoubleType> ac(2*ax.size());
    for (size_t i = 0, j = 0; i < ax.size(); ++i)
    {
      ac[j++] = ax[i];
      ac[j++] = az[i];
    }
    factor_args["is_complex"] = ObjectHolder(true);
    factor_args["Ax"] = CreateDoublePODArray(ac);
  }
  else
  {
    factor_args["is_complex"] = ObjectHolder(false);
    factor_args["Ax"] = CreateDoublePODArray(cm->GetAx());
  }
  call_profile.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  factor_build_seconds_ += call_profile.build_seconds;

  Interpreter interpreter;
  const auto interp_begin = std::chrono::steady_clock::now();
  bool ret = interpreter.RunCommand(command_handle_, factor_args);
  call_profile.interpreter_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - interp_begin).count();
  factor_interpreter_seconds_ += call_profile.interpreter_seconds;
  if (!ret)
  {
    std::string error = "while factorizing matrix using python solver\n";
    error += interpreter.GetErrorString();
    OutputStream::WriteOut(OutputStream::OutputType::FATAL, error.c_str());
  }
  else
  {
    std::string error_string;
    ObjectHolderMap_t result_dictionary;
    const auto result_begin = std::chrono::steady_clock::now();
    auto result = interpreter.GetResult();
    ret = result.GetHashMap(result_dictionary);
    call_profile.result_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - result_begin).count();
    factor_result_seconds_ += call_profile.result_seconds;
    if (!ret)
    {
      error_string += "python solver object did not return a dictionary\n";
    }
    else
    {
      const auto validate_begin = std::chrono::steady_clock::now();
      for (const auto &arg: return_keys)
      {
        if (auto it = result_dictionary.find(arg); it == result_dictionary.end())
        {
          error_string += "python solver object did not return a dictionary containing \"" + arg + "\"\n";
          ret = false;
        }
      }
      auto status = result_dictionary["status"].GetBoolean().second;
      auto message = result_dictionary["message"].GetString();
      error_string += message;
      dsAssert(status, error_string);
      call_profile.validate_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - validate_begin).count();
      factor_validate_seconds_ += call_profile.validate_seconds;
    }
  }
  factor_parse_seconds_ += call_profile.result_seconds + call_profile.validate_seconds;
  call_profile.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  call_profile.other_seconds = std::max(
    0.0,
    call_profile.total_seconds
      - call_profile.build_seconds
      - call_profile.interpreter_seconds
      - call_profile.result_seconds
      - call_profile.validate_seconds
  );
  factor_call_profiles_.push_back(call_profile);
  factor_total_seconds_ += call_profile.total_seconds;
  return ret;
}

template <typename DoubleType>
void ExternalPreconditioner<DoubleType>::DerivedLUSolve(DoubleVec_t<DoubleType> &x, const DoubleVec_t<DoubleType> &b) const
{
  const auto begin = std::chrono::steady_clock::now();
  SolveCallProfile call_profile;
  ++solve_calls_;
  dsAssert(command_handle_.IsCallable(), "python solver command is not callable\n");
  dsAssert(!command_data_.empty(), "python solver invalid data\n");
  device_result_buffer_.valid = false;
  device_result_buffer_.copy_rows_to_host_double = nullptr;
  device_result_buffer_.copy_to_host_double = nullptr;

  const std::string return_keys[] = {
    "status",
    "message",
    "x",
  };
  const auto result_mode = GetCuDSSResultMode();
  const bool mode_forces_host = (result_mode == CuDSSResultMode::HOST);
  const bool node_forces_host = NodeKeeper::instance().HaveNodes();
  const bool need_host_vector = mode_forces_host || node_forces_host;
  call_profile.need_host_vector = need_host_vector ? 1 : 0;
  ObjectHolderMap_t factor_args = {
    {"action", ObjectHolder("solve")},
    {"solver_object", command_data_},
    {"is_complex", ObjectHolder(false)},
    {"result_mode", ObjectHolder(ToResultModeText(result_mode))},
    {"require_host_x", ObjectHolder(need_host_vector)},
    {"b", CreateDoublePODArray(b)},
  };
  call_profile.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  solve_build_seconds_ += call_profile.build_seconds;

  Interpreter interpreter;
  const auto interp_begin = std::chrono::steady_clock::now();
  bool ret = interpreter.RunCommand(command_handle_, factor_args);
  call_profile.interpreter_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - interp_begin).count();
  solve_interpreter_seconds_ += call_profile.interpreter_seconds;
  if (!ret)
  {
    std::string error = "while solving matrix using python solver\n";
    error += interpreter.GetErrorString();
    OutputStream::WriteOut(OutputStream::OutputType::FATAL, error.c_str());
  }
  else
  {
    std::string error_string;
    ObjectHolderMap_t result_dictionary;
    const auto result_begin = std::chrono::steady_clock::now();
    auto result = interpreter.GetResult();
    ret = result.GetHashMap(result_dictionary);
    call_profile.result_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - result_begin).count();
    solve_result_seconds_ += call_profile.result_seconds;
    if (!ret)
    {
      error_string += "python solver object did not return a dictionary\n";
    }
    else
    {
      const auto validate_begin = std::chrono::steady_clock::now();
      for (const auto &arg: return_keys)
      {
        if (auto it = result_dictionary.find(arg); it == result_dictionary.end())
        {
          error_string += "python solver object did not return a dictionary containing \"" + arg + "\"\n";
          ret = false;
        }
      }
      auto status = result_dictionary["status"].GetBoolean().second;
      auto message = result_dictionary["message"].GetString();
      error_string += message;
      dsAssert(status, error_string);
      call_profile.validate_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - validate_begin).count();
      solve_validate_seconds_ += call_profile.validate_seconds;

      const auto device_meta_begin = std::chrono::steady_clock::now();
      if (auto it = result_dictionary.find("x_device_token"); it != result_dictionary.end())
      {
        last_device_result_token_ = it->second.GetString();
      }
      else
      {
        last_device_result_token_.clear();
      }
      if (auto it = result_dictionary.find("x_location"); it != result_dictionary.end())
      {
        last_device_result_location_ = it->second.GetString();
      }
      else
      {
        last_device_result_location_ = "host";
      }

      last_result_object_ = result_dictionary["x"];
      call_profile.device_meta_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - device_meta_begin).count();
      solve_device_meta_seconds_ += call_profile.device_meta_seconds;
      device_result_buffer_.valid = false;
      if (need_host_vector)
      {
        const auto materialize_begin = std::chrono::steady_clock::now();
        std::vector<double> xv;
        preswap(xv, x);
        xv.resize(b.size());
        bool ok = result_dictionary["x"].GetDoubleValues(xv.data(), xv.size());
        if (!ok)
        {
          xv.clear();
          ok = result_dictionary["x"].GetDoubleList(xv);
        }
        convertToType(xv, x);
        dsAssert(ok && (x.size() == b.size()), "Mismatch in returned x");
        call_profile.materialize_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - materialize_begin).count();
        solve_materialize_seconds_ += call_profile.materialize_seconds;
      }
      else
      {
        x.clear();
        device_result_buffer_.valid = !last_device_result_token_.empty() &&
          (last_device_result_location_.rfind("device_experimental", 0) == 0);
        if (device_result_buffer_.valid)
        {
          const auto device_view_begin = std::chrono::steady_clock::now();
          ++result_generation_;
          device_result_buffer_.backend = "cudss_callback";
          device_result_buffer_.length = b.size();
          device_result_buffer_.generation = result_generation_;
          device_result_buffer_.copy_rows_to_host_double = [this](const std::vector<size_t> &rows, std::vector<double> &out) -> bool {
            std::vector<int> irows;
            irows.reserve(rows.size());
            for (const auto row : rows)
            {
              if (row > static_cast<size_t>(std::numeric_limits<int>::max()))
              {
                return false;
              }
              irows.push_back(static_cast<int>(row));
            }
            ObjectHolderMap_t gather_args = {
              {"action", ObjectHolder("gather_rows")},
              {"solver_object", command_data_},
              {"rows", CreateIntPODArray(irows)},
            };
            Interpreter gather_interpreter;
            if (!gather_interpreter.RunCommand(command_handle_, gather_args))
            {
              return false;
            }
            ObjectHolderMap_t gather_result;
            if (!gather_interpreter.GetResult().GetHashMap(gather_result))
            {
              return false;
            }
            auto sit = gather_result.find("status");
            if (sit != gather_result.end() && !sit->second.GetBoolean().second)
            {
              return false;
            }
            auto vit = gather_result.find("values");
            if (vit == gather_result.end())
            {
              return false;
            }
            out.resize(rows.size());
            bool ok = vit->second.GetDoubleValues(out.data(), out.size());
            if (!ok)
            {
              out.clear();
              ok = vit->second.GetDoubleList(out);
            }
            return ok && (out.size() == rows.size());
          };
          call_profile.device_view_enabled = 1;
          call_profile.device_view_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - device_view_begin).count();
          solve_device_view_seconds_ += call_profile.device_view_seconds;
        }
      }
    }
  }
  solve_parse_seconds_ += call_profile.result_seconds + call_profile.validate_seconds + call_profile.device_meta_seconds;
  call_profile.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  call_profile.other_seconds = std::max(
    0.0,
    call_profile.total_seconds
      - call_profile.build_seconds
      - call_profile.interpreter_seconds
      - call_profile.result_seconds
      - call_profile.validate_seconds
      - call_profile.device_meta_seconds
      - call_profile.materialize_seconds
      - call_profile.device_view_seconds
  );
  solve_call_profiles_.push_back(call_profile);
  solve_total_seconds_ += call_profile.total_seconds;
}

template <typename DoubleType>
void ExternalPreconditioner<DoubleType>::DerivedLUSolve(ComplexDoubleVec_t<DoubleType> &x, const ComplexDoubleVec_t<DoubleType> &b) const
{
  const auto begin = std::chrono::steady_clock::now();
  SolveCallProfile call_profile;
  ++solve_calls_;
  dsAssert(command_handle_.IsCallable(), "python solver command is not callable\n");
  dsAssert(!command_data_.empty(), "python solver invalid data\n");
  device_result_buffer_.valid = false;
  device_result_buffer_.copy_rows_to_host_double = nullptr;
  device_result_buffer_.copy_to_host_double = nullptr;
  last_device_result_token_.clear();
  last_device_result_location_ = "host";

  const std::string return_keys[] = {
    "status",
    "message",
    "x",
  };

  ObjectHolderMap_t factor_args = {
    {"action", ObjectHolder("solve")},
    {"solver_object", command_data_},
    {"is_complex", ObjectHolder(true)},
    {"b", CreateComplexDoublePODArray(b)},
  };
  call_profile.need_host_vector = 1;
  call_profile.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  solve_build_seconds_ += call_profile.build_seconds;

  Interpreter interpreter;
  const auto interp_begin = std::chrono::steady_clock::now();
  bool ret = interpreter.RunCommand(command_handle_, factor_args);
  call_profile.interpreter_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - interp_begin).count();
  solve_interpreter_seconds_ += call_profile.interpreter_seconds;
  if (!ret)
  {
    std::string error = "while solving matrix using python solver\n";
    error += interpreter.GetErrorString();
    OutputStream::WriteOut(OutputStream::OutputType::FATAL, error.c_str());
  }
  else
  {
    std::string error_string;
    ObjectHolderMap_t result_dictionary;
    const auto result_begin = std::chrono::steady_clock::now();
    auto result = interpreter.GetResult();
    ret = result.GetHashMap(result_dictionary);
    call_profile.result_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - result_begin).count();
    solve_result_seconds_ += call_profile.result_seconds;
    if (!ret)
    {
      error_string += "python solver object did not return a dictionary\n";
    }
    else
    {
      const auto validate_begin = std::chrono::steady_clock::now();
      for (const auto &arg: return_keys)
      {
        if (auto it = result_dictionary.find(arg); it == result_dictionary.end())
        {
          error_string += "python solver object did not return a dictionary containing \"" + arg + "\"\n";
          ret = false;
        }
      }
      auto status = result_dictionary["status"].GetBoolean().second;
      auto message = result_dictionary["message"].GetString();
      error_string += message;
      dsAssert(status, error_string);
      call_profile.validate_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - validate_begin).count();
      solve_validate_seconds_ += call_profile.validate_seconds;

      bool ret = false;
      const auto materialize_begin = std::chrono::steady_clock::now();
      ComplexDoubleVec_t<double> xv;
      preswap(xv, x);
      ret = result_dictionary["x"].GetComplexDoubleList(xv);
      convertToType(xv, x);
      dsAssert(ret && (x.size() == b.size()), "Mismatch in returned x");
      call_profile.materialize_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - materialize_begin).count();
      solve_materialize_seconds_ += call_profile.materialize_seconds;
    }
  }
  solve_parse_seconds_ += call_profile.result_seconds + call_profile.validate_seconds;
  call_profile.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  call_profile.other_seconds = std::max(
    0.0,
    call_profile.total_seconds
      - call_profile.build_seconds
      - call_profile.interpreter_seconds
      - call_profile.result_seconds
      - call_profile.validate_seconds
      - call_profile.materialize_seconds
  );
  solve_call_profiles_.push_back(call_profile);
  solve_total_seconds_ += call_profile.total_seconds;
}
}


template class dsMath::ExternalPreconditioner<double>;
#ifdef DEVSIM_EXTENDED_PRECISION
#include "Float128.hh"
template class dsMath::ExternalPreconditioner<float128>;
#endif
