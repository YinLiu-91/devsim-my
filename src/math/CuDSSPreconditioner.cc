/***
 * DEVSIM
 * SPDX-License-Identifier: Apache-2.0
 ***/

#include "CuDSSPreconditioner.hh"
#include "CompressedMatrix.hh"
#include "GlobalData.hh"
#include "Interpreter.hh"
#include "NodeKeeper.hh"
#include "OutputStream.hh"
#include "dsAssert.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <glob.h>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsMath {
namespace
{
enum class CuDSSResultMode {
  HOST,
  DEVICE_EXPERIMENTAL,
};
enum class CuDSSBackendPolicy {
  AUTO,
  NATIVE,
  CALLBACK,
};

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
bool IsCuDSSProfileEnabled()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_PROFILE"))
  {
    const std::string val(v);
    return (val == "1") || (val == "true") || (val == "TRUE") || (val == "on") || (val == "ON");
  }
  return false;
}
bool ParseTrueValue(const std::string &val)
{
  return (val == "1") || (val == "true") || (val == "TRUE") || (val == "on") || (val == "ON");
}
bool IsPinnedStagingEnabled()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_PINNED_STAGING"))
  {
    return ParseTrueValue(std::string(v));
  }
  return true;
}
bool IsMTModeEnabled()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_MT_MODE"))
  {
    return ParseTrueValue(std::string(v));
  }
  return false;
}
std::vector<std::string> ExpandGlobPattern(const std::string &pattern)
{
  std::vector<std::string> out;
  glob_t glob_result {};
  if (glob(pattern.c_str(), 0, nullptr, &glob_result) == 0)
  {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
    {
      out.emplace_back(glob_result.gl_pathv[i]);
    }
  }
  globfree(&glob_result);
  return out;
}
std::vector<std::string> GetPythonRuntimeLibraryCandidates(const std::string &lib_basename)
{
  std::vector<std::string> out;
  const std::string patterns[] = {
    "/usr/local/lib/python*/dist-packages/nvidia/cu*/lib/" + lib_basename,
    "/usr/local/lib/python*/site-packages/nvidia/cu*/lib/" + lib_basename,
    "/usr/local/lib/python*/dist-packages/nvidia/cudss/lib/" + lib_basename,
    "/usr/local/lib/python*/site-packages/nvidia/cudss/lib/" + lib_basename,
    "/usr/lib/python*/dist-packages/nvidia/cu*/lib/" + lib_basename,
    "/usr/lib/python*/site-packages/nvidia/cu*/lib/" + lib_basename,
    "/usr/lib/python*/dist-packages/nvidia/cudss/lib/" + lib_basename,
    "/usr/lib/python*/site-packages/nvidia/cudss/lib/" + lib_basename,
  };
  for (const auto &pattern : patterns)
  {
    auto matches = ExpandGlobPattern(pattern);
    out.insert(out.end(), matches.begin(), matches.end());
  }
  return out;
}
std::string FindPythonRuntimeLibrary(const std::string &lib_basename)
{
  for (const auto &candidate : GetPythonRuntimeLibraryCandidates(lib_basename))
  {
    if (access(candidate.c_str(), R_OK) == 0)
    {
      return candidate;
    }
  }
  return "";
}
bool IsNativeStreamEnabled()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_USE_STREAM"))
  {
    return ParseTrueValue(std::string(v));
  }
  return false;
}
std::string GetMTThreadingLayerPath()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_THREADING_LAYER"))
  {
    return std::string(v);
  }
  if (const auto candidate = FindPythonRuntimeLibrary("libcudss_mtlayer_gomp.so.0"); !candidate.empty())
  {
    return candidate;
  }
  return "libcudss_mtlayer_gomp.so.0";
}
bool IsSymbolicHashReuseEnabled()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_SYMBOLIC_HASH_REUSE"))
  {
    return ParseTrueValue(std::string(v));
  }
  return false;
}
std::uint64_t HashPattern(const std::vector<int> &ap, const std::vector<int> &ai)
{
  std::uint64_t h = 1469598103934665603ULL;
  constexpr std::uint64_t prime = 1099511628211ULL;
  for (const auto v : ap)
  {
    h ^= static_cast<std::uint64_t>(v);
    h *= prime;
  }
  h ^= 0x9e3779b97f4a7c15ULL;
  h *= prime;
  for (const auto v : ai)
  {
    h ^= static_cast<std::uint64_t>(v);
    h *= prime;
  }
  return h;
}
std::vector<std::pair<int, int>> ParseConfigSetPairs()
{
  std::vector<std::pair<int, int>> out;
  const auto *raw = std::getenv("DEVSIM_CUDSS_CONFIG_SET");
  if (!raw)
  {
    return out;
  }
  std::string cfg(raw);
  std::size_t pos = 0;
  while (pos < cfg.size())
  {
    const auto comma = cfg.find(',', pos);
    const auto token = cfg.substr(pos, (comma == std::string::npos) ? std::string::npos : (comma - pos));
    if (!token.empty())
    {
      const auto eq = token.find('=');
      if (eq != std::string::npos)
      {
        try
        {
          const int param = std::stoi(token.substr(0, eq));
          const int value = std::stoi(token.substr(eq + 1));
          out.emplace_back(param, value);
        }
        catch (...)
        {
        }
      }
    }
    if (comma == std::string::npos)
    {
      break;
    }
    pos = comma + 1;
  }
  return out;
}
CuDSSBackendPolicy GetCuDSSBackendPolicy()
{
  if (const auto *v = std::getenv("DEVSIM_CUDSS_BACKEND_POLICY"))
  {
    const std::string val(v);
    if (val == "native")
    {
      return CuDSSBackendPolicy::NATIVE;
    }
    if (val == "callback")
    {
      return CuDSSBackendPolicy::CALLBACK;
    }
    if (val != "auto")
    {
      std::ostringstream os;
      os << "Unrecognized DEVSIM_CUDSS_BACKEND_POLICY=\"" << val
         << "\". Falling back to auto.\n";
      OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
    }
    return CuDSSBackendPolicy::AUTO;
  }

  // Backward-compatible toggle: keep honoring DEVSIM_CUDSS_NATIVE_CPP.
  if (const auto *v = std::getenv("DEVSIM_CUDSS_NATIVE_CPP"))
  {
    return ParseTrueValue(std::string(v)) ? CuDSSBackendPolicy::NATIVE : CuDSSBackendPolicy::CALLBACK;
  }
  return CuDSSBackendPolicy::AUTO;
}
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
    if (val != "host")
    {
      std::ostringstream os;
      os << "Unrecognized \"cudss_result_mode\" value \"" << val
         << "\". Falling back to \"host\".\n";
      OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
    }
  }
  return CuDSSResultMode::HOST;
}
const char *ToResultModeText(CuDSSResultMode mode)
{
  return (mode == CuDSSResultMode::DEVICE_EXPERIMENTAL) ? "device_experimental" : "host";
}

constexpr int CUDA_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int CUDA_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int CUDSS_STATUS_SUCCESS = 0;
constexpr int CUDSS_PHASE_ANALYSIS = 0x1 | 0x2;
constexpr int CUDSS_PHASE_FACTORIZATION = 0x4;
constexpr int CUDSS_PHASE_REFACTORIZATION = 0x8;
constexpr int CUDSS_PHASE_SOLVE = 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200;
constexpr int CUDSS_BASE_ZERO = 0;
constexpr int CUDSS_MTYPE_GENERAL = 0;
constexpr int CUDSS_MVIEW_FULL = 0;
constexpr int CUDSS_LAYOUT_COL_MAJOR = 0;
constexpr int CUDSS_CONFIG_REORDERING_ALG = 0;
constexpr int CUDSS_CONFIG_HYBRID_MODE = 12;
constexpr int CUDSS_CONFIG_HOST_NTHREADS = 15;
constexpr int CUDSS_CONFIG_HYBRID_EXECUTE_MODE = 16;
constexpr int CUDA_R_64I = 24;
constexpr int CUDA_R_64F = 1;

using CudssCreate_t = int (*)(void **);
using CudssDestroy_t = int (*)(void *);
using CudssSetThreadingLayer_t = int (*)(void *, const char *);
using CudssSetStream_t = int (*)(void *, void *);
using CudssConfigCreate_t = int (*)(void **);
using CudssConfigDestroy_t = int (*)(void *);
using CudssConfigSet_t = int (*)(void *, int, const void *, std::size_t);
using CudssDataCreate_t = int (*)(void *, void **);
using CudssDataDestroy_t = int (*)(void *, void *);
using CudssMatrixCreateCsr_t = int (*)(void **, std::int64_t, std::int64_t, std::int64_t, void *, void *, void *, void *, int, int, int, int, int);
using CudssMatrixCreateDn_t = int (*)(void **, std::int64_t, std::int64_t, std::int64_t, void *, int, int);
using CudssMatrixDestroy_t = int (*)(void *);
using CudssExecute_t = int (*)(void *, int, void *, void *, void *, void *, void *);

using CudaGetDeviceCount_t = int (*)(int *);
using CudaMalloc_t = int (*)(void **, std::size_t);
using CudaFree_t = int (*)(void *);
using CudaMemcpy_t = int (*)(void *, const void *, std::size_t, int);
using CudaDeviceSynchronize_t = int (*)();
using CudaHostAlloc_t = int (*)(void **, std::size_t, unsigned int);
using CudaFreeHost_t = int (*)(void *);
using CudaStreamCreate_t = int (*)(void **);
using CudaStreamDestroy_t = int (*)(void *);
using CudaStreamSynchronize_t = int (*)(void *);

struct NativeCuDSSState {
  void *cudss_lib = nullptr;
  void *cudart_lib = nullptr;

  CudssCreate_t cudssCreate = nullptr;
  CudssDestroy_t cudssDestroy = nullptr;
  CudssSetThreadingLayer_t cudssSetThreadingLayer = nullptr;
  CudssSetStream_t cudssSetStream = nullptr;
  CudssConfigCreate_t cudssConfigCreate = nullptr;
  CudssConfigDestroy_t cudssConfigDestroy = nullptr;
  CudssConfigSet_t cudssConfigSet = nullptr;
  CudssDataCreate_t cudssDataCreate = nullptr;
  CudssDataDestroy_t cudssDataDestroy = nullptr;
  CudssMatrixCreateCsr_t cudssMatrixCreateCsr = nullptr;
  CudssMatrixCreateDn_t cudssMatrixCreateDn = nullptr;
  CudssMatrixDestroy_t cudssMatrixDestroy = nullptr;
  CudssExecute_t cudssExecute = nullptr;

  CudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
  CudaMalloc_t cudaMalloc = nullptr;
  CudaFree_t cudaFree = nullptr;
  CudaMemcpy_t cudaMemcpy = nullptr;
  CudaDeviceSynchronize_t cudaDeviceSynchronize = nullptr;
  CudaHostAlloc_t cudaHostAlloc = nullptr;
  CudaFreeHost_t cudaFreeHost = nullptr;
  CudaStreamCreate_t cudaStreamCreate = nullptr;
  CudaStreamDestroy_t cudaStreamDestroy = nullptr;
  CudaStreamSynchronize_t cudaStreamSynchronize = nullptr;

  void *handle = nullptr;
  void *config = nullptr;
  void *data = nullptr;
  void *stream = nullptr;
  void *mat_a = nullptr;
  void *mat_rhs = nullptr;
  void *mat_sol = nullptr;
  void *d_ap = nullptr;
  void *d_ai = nullptr;
  void *d_ax = nullptr;
  void *d_rhs = nullptr;
  void *d_x = nullptr;
  void *h_rhs_pinned = nullptr;
  void *h_x_pinned = nullptr;
  bool pinned_rhs_enabled = false;
  bool mt_mode_enabled = false;
  bool stream_enabled = false;
  std::size_t config_set_applied = 0;
  std::size_t n = 0;
  std::size_t nnz = 0;
  bool symbolic_ready = false;
  std::vector<std::int64_t> ap64;
  std::vector<std::int64_t> ai64;
  std::vector<double> xcache;
  std::size_t h2d_bytes = 0;
  std::size_t d2h_bytes = 0;
  std::size_t analysis_calls = 0;
  std::size_t refactor_calls = 0;
  std::size_t solve_calls = 0;
  std::size_t hash_reuse_hits = 0;
  std::size_t hash_reuse_checks = 0;
  std::uint64_t last_pattern_hash = 0;
  std::uint64_t generation = 0;
};

void ApplyNamedConfigSet(const char *env_name, int param, NativeCuDSSState &st)
{
  if (!st.cudssConfigSet || !st.config)
  {
    return;
  }

  const auto *raw = std::getenv(env_name);
  if (!raw)
  {
    return;
  }

  try
  {
    const int value = std::stoi(raw);
    if (st.cudssConfigSet(st.config, param, &value, sizeof(value)) == CUDSS_STATUS_SUCCESS)
    {
      ++st.config_set_applied;
    }
    else
    {
      std::ostringstream os;
      os << "native cuDSS backend: cudssConfigSet failed for " << env_name
         << "=" << value << " (param=" << param << ")\n";
      OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
    }
  }
  catch (...)
  {
    std::ostringstream os;
    os << "native cuDSS backend: invalid integer for " << env_name << "\n";
    OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
  }
}

std::unordered_map<const void *, std::unique_ptr<NativeCuDSSState>> &GetNativeStateMap()
{
  static std::unordered_map<const void *, std::unique_ptr<NativeCuDSSState>> states;
  return states;
}

void *LoadSharedObject(const char *env_key, const char *default_name)
{
  if (const auto *path = std::getenv(env_key))
  {
    if (auto *lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL))
    {
      return lib;
    }
  }
  if (auto *lib = dlopen(default_name, RTLD_LAZY | RTLD_LOCAL))
  {
    return lib;
  }
  if (const auto candidate = FindPythonRuntimeLibrary(default_name); !candidate.empty())
  {
    if (auto *lib = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_LOCAL))
    {
      return lib;
    }
  }
  return nullptr;
}

template <typename FnType>
bool LoadSymbol(void *lib, const char *name, FnType &fn, std::string &error)
{
  fn = reinterpret_cast<FnType>(dlsym(lib, name));
  if (!fn)
  {
    error += std::string("Failed to load symbol: ") + name + "\n";
    return false;
  }
  return true;
}

void NativeDestroyState(NativeCuDSSState &st)
{
  if (st.cudssMatrixDestroy)
  {
    if (st.mat_a) st.cudssMatrixDestroy(st.mat_a);
    if (st.mat_rhs) st.cudssMatrixDestroy(st.mat_rhs);
    if (st.mat_sol) st.cudssMatrixDestroy(st.mat_sol);
  }
  st.mat_a = st.mat_rhs = st.mat_sol = nullptr;
  if (st.cudaFree)
  {
    if (st.d_ap) st.cudaFree(st.d_ap);
    if (st.d_ai) st.cudaFree(st.d_ai);
    if (st.d_ax) st.cudaFree(st.d_ax);
    if (st.d_rhs) st.cudaFree(st.d_rhs);
    if (st.d_x) st.cudaFree(st.d_x);
  }
  st.d_ap = st.d_ai = st.d_ax = st.d_rhs = st.d_x = nullptr;
  if (st.cudaFreeHost && st.h_rhs_pinned)
  {
    st.cudaFreeHost(st.h_rhs_pinned);
  }
  st.h_rhs_pinned = nullptr;
  if (st.cudaFreeHost && st.h_x_pinned)
  {
    st.cudaFreeHost(st.h_x_pinned);
  }
  st.h_x_pinned = nullptr;
  if (st.cudssDataDestroy && st.handle && st.data)
  {
    st.cudssDataDestroy(st.handle, st.data);
  }
  st.data = nullptr;
  if (st.cudaStreamDestroy && st.stream)
  {
    st.cudaStreamDestroy(st.stream);
  }
  st.stream = nullptr;
  if (st.cudssConfigDestroy && st.config)
  {
    st.cudssConfigDestroy(st.config);
  }
  st.config = nullptr;
  if (st.cudssDestroy && st.handle)
  {
    st.cudssDestroy(st.handle);
  }
  st.handle = nullptr;
  // cuDSS MT mode may retain internal threading-layer state past solver teardown;
  // unloading the shared objects at process shutdown has triggered crashes in practice.
  if (!st.mt_mode_enabled)
  {
    if (st.cudss_lib)
    {
      dlclose(st.cudss_lib);
      st.cudss_lib = nullptr;
    }
    if (st.cudart_lib)
    {
      dlclose(st.cudart_lib);
      st.cudart_lib = nullptr;
    }
  }
}

NativeCuDSSState *GetNativeState(const void *owner)
{
  auto &states = GetNativeStateMap();
  if (auto it = states.find(owner); it != states.end())
  {
    return it->second.get();
  }
  return nullptr;
}

void RemoveNativeState(const void *owner)
{
  auto &states = GetNativeStateMap();
  if (auto it = states.find(owner); it != states.end())
  {
    NativeDestroyState(*it->second);
    states.erase(it);
  }
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
CuDSSPreconditioner<DoubleType>::CuDSSPreconditioner(size_t sz, PEnum::TransposeType_t transpose)
  : Preconditioner<DoubleType>(sz, transpose),
    compression_type_(CompressionType::CRM),
    symbolic_initialized_(false),
    last_device_result_token_(""),
    last_device_result_location_("host"),
    profile_enabled_(IsCuDSSProfileEnabled()),
    factor_calls_(0),
    solve_calls_(0),
    analysis_calls_(0),
    refactor_calls_(0),
    host_materialize_mode_calls_(0),
    host_materialize_node_calls_(0),
    analysis_miss_first_factor_calls_(0),
    analysis_miss_symbolic_status_calls_(0),
    analysis_miss_hash_mismatch_calls_(0),
    analysis_miss_dim_change_calls_(0),
    analysis_miss_backend_mode_calls_(0),
    last_symbolic_nnz_(0),
    last_symbolic_nnz_valid_(false),
    analysis_seconds_(0.0),
    refactor_seconds_(0.0),
    factor_seconds_(0.0),
    solve_seconds_(0.0),
    backend_fallback_to_callback_(false)
{
  device_result_buffer_.backend = "none";
}

template <typename DoubleType>
bool CuDSSPreconditioner<DoubleType>::init(ObjectHolder oh, std::string &error_string)
{
  device_result_buffer_.valid = false;
  device_result_buffer_.backend = "none";
  device_result_buffer_.length = 0;
  device_result_buffer_.generation = 0;
  device_result_buffer_.copy_to_host_double = nullptr;
  device_result_buffer_.copy_rows_to_host_double = nullptr;

  if constexpr (std::is_same_v<DoubleType, double>)
  {
    const auto backend_policy = GetCuDSSBackendPolicy();
    const bool prefer_native = (backend_policy != CuDSSBackendPolicy::CALLBACK);
    backend_fallback_to_callback_ = false;
    if (prefer_native)
    {
      auto native_state = std::make_unique<NativeCuDSSState>();
      native_state->n = this->size();
      native_state->cudss_lib = LoadSharedObject("DEVSIM_CUDSS_LIB", "libcudss.so.0");
      native_state->cudart_lib = LoadSharedObject("DEVSIM_CUDART_LIB", "libcudart.so");
      if (!native_state->cudss_lib || !native_state->cudart_lib)
      {
        if (native_state->cudss_lib) dlclose(native_state->cudss_lib);
        if (native_state->cudart_lib) dlclose(native_state->cudart_lib);
        if (backend_policy == CuDSSBackendPolicy::NATIVE)
        {
          error_string += "native cuDSS backend requested but failed to load libcudss/libcudart\n";
          return false;
        }
        backend_fallback_to_callback_ = true;
      }
      else
      {
        bool ok = true;
        ok &= LoadSymbol(native_state->cudss_lib, "cudssCreate", native_state->cudssCreate, error_string);
        ok &= LoadSymbol(native_state->cudss_lib, "cudssDestroy", native_state->cudssDestroy, error_string);
        native_state->cudssSetThreadingLayer = reinterpret_cast<CudssSetThreadingLayer_t>(dlsym(native_state->cudss_lib, "cudssSetThreadingLayer"));
        native_state->cudssSetStream = reinterpret_cast<CudssSetStream_t>(dlsym(native_state->cudss_lib, "cudssSetStream"));
        ok &= LoadSymbol(native_state->cudss_lib, "cudssConfigCreate", native_state->cudssConfigCreate, error_string);
        ok &= LoadSymbol(native_state->cudss_lib, "cudssConfigDestroy", native_state->cudssConfigDestroy, error_string);
        native_state->cudssConfigSet = reinterpret_cast<CudssConfigSet_t>(dlsym(native_state->cudss_lib, "cudssConfigSet"));
        ok &= LoadSymbol(native_state->cudss_lib, "cudssDataCreate", native_state->cudssDataCreate, error_string);
        ok &= LoadSymbol(native_state->cudss_lib, "cudssDataDestroy", native_state->cudssDataDestroy, error_string);
        ok &= LoadSymbol(native_state->cudss_lib, "cudssMatrixCreateCsr", native_state->cudssMatrixCreateCsr, error_string);
        ok &= LoadSymbol(native_state->cudss_lib, "cudssMatrixCreateDn", native_state->cudssMatrixCreateDn, error_string);
        ok &= LoadSymbol(native_state->cudss_lib, "cudssMatrixDestroy", native_state->cudssMatrixDestroy, error_string);
        ok &= LoadSymbol(native_state->cudss_lib, "cudssExecute", native_state->cudssExecute, error_string);
        ok &= LoadSymbol(native_state->cudart_lib, "cudaGetDeviceCount", native_state->cudaGetDeviceCount, error_string);
        ok &= LoadSymbol(native_state->cudart_lib, "cudaMalloc", native_state->cudaMalloc, error_string);
        ok &= LoadSymbol(native_state->cudart_lib, "cudaFree", native_state->cudaFree, error_string);
        ok &= LoadSymbol(native_state->cudart_lib, "cudaMemcpy", native_state->cudaMemcpy, error_string);
        ok &= LoadSymbol(native_state->cudart_lib, "cudaDeviceSynchronize", native_state->cudaDeviceSynchronize, error_string);
        native_state->cudaStreamCreate = reinterpret_cast<CudaStreamCreate_t>(dlsym(native_state->cudart_lib, "cudaStreamCreate"));
        native_state->cudaStreamDestroy = reinterpret_cast<CudaStreamDestroy_t>(dlsym(native_state->cudart_lib, "cudaStreamDestroy"));
        native_state->cudaStreamSynchronize = reinterpret_cast<CudaStreamSynchronize_t>(dlsym(native_state->cudart_lib, "cudaStreamSynchronize"));
        native_state->pinned_rhs_enabled = false;
        if (IsPinnedStagingEnabled())
        {
          native_state->cudaHostAlloc = reinterpret_cast<CudaHostAlloc_t>(dlsym(native_state->cudart_lib, "cudaHostAlloc"));
          native_state->cudaFreeHost = reinterpret_cast<CudaFreeHost_t>(dlsym(native_state->cudart_lib, "cudaFreeHost"));
          native_state->pinned_rhs_enabled = (native_state->cudaHostAlloc != nullptr) && (native_state->cudaFreeHost != nullptr);
          if (!native_state->pinned_rhs_enabled)
          {
            std::ostringstream os;
            os << "native cuDSS backend: cudaHostAlloc/cudaFreeHost unavailable, disable pinned RHS staging.\n";
            OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
          }
        }
        if (ok)
        {
          int gpu_count = 0;
          ok = (native_state->cudaGetDeviceCount(&gpu_count) == 0) && (gpu_count > 0);
          if (!ok)
          {
            error_string += "native cuDSS backend could not detect CUDA GPU\n";
          }
        }
        if (ok) ok = (native_state->cudssCreate(&native_state->handle) == CUDSS_STATUS_SUCCESS);
        if (ok && IsNativeStreamEnabled())
        {
          if (native_state->cudssSetStream && native_state->cudaStreamCreate && native_state->cudaStreamDestroy && native_state->cudaStreamSynchronize)
          {
            if ((native_state->cudaStreamCreate(&native_state->stream) == 0) &&
                (native_state->cudssSetStream(native_state->handle, native_state->stream) == CUDSS_STATUS_SUCCESS))
            {
              native_state->stream_enabled = true;
            }
            else
            {
              OutputStream::WriteOut(OutputStream::OutputType::INFO,
                "native cuDSS backend: failed to enable custom stream mode.\n");
              if (native_state->stream && native_state->cudaStreamDestroy)
              {
                native_state->cudaStreamDestroy(native_state->stream);
                native_state->stream = nullptr;
              }
            }
          }
          else
          {
            OutputStream::WriteOut(OutputStream::OutputType::INFO,
              "native cuDSS backend: cudssSetStream/cudaStream* unavailable; stream mode disabled.\n");
          }
        }
        if (ok && IsMTModeEnabled())
        {
          if (native_state->cudssSetThreadingLayer)
          {
            const auto threading_layer = GetMTThreadingLayerPath();
            const auto status = native_state->cudssSetThreadingLayer(native_state->handle, threading_layer.c_str());
            if (status == CUDSS_STATUS_SUCCESS)
            {
              native_state->mt_mode_enabled = true;
            }
            else
            {
              std::ostringstream os;
              os << "native cuDSS backend: failed to enable MT mode with threading layer \""
                 << threading_layer << "\" (status " << status << ")\n";
              OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
            }
          }
          else
          {
            OutputStream::WriteOut(OutputStream::OutputType::INFO,
              "native cuDSS backend: cudssSetThreadingLayer unavailable; MT mode disabled.\n");
          }
        }
        if (ok) ok = (native_state->cudssConfigCreate(&native_state->config) == CUDSS_STATUS_SUCCESS);
        if (ok && native_state->cudssConfigSet)
        {
          ApplyNamedConfigSet("DEVSIM_CUDSS_REORDERING_ALG", CUDSS_CONFIG_REORDERING_ALG, *native_state);
          ApplyNamedConfigSet("DEVSIM_CUDSS_HYBRID_MODE", CUDSS_CONFIG_HYBRID_MODE, *native_state);
          ApplyNamedConfigSet("DEVSIM_CUDSS_HOST_NTHREADS", CUDSS_CONFIG_HOST_NTHREADS, *native_state);
          ApplyNamedConfigSet("DEVSIM_CUDSS_HYBRID_EXECUTE_MODE", CUDSS_CONFIG_HYBRID_EXECUTE_MODE, *native_state);
          for (const auto &[param, value] : ParseConfigSetPairs())
          {
            if (native_state->cudssConfigSet(native_state->config, param, &value, sizeof(value)) == CUDSS_STATUS_SUCCESS)
            {
              ++native_state->config_set_applied;
            }
            else
            {
              std::ostringstream os;
              os << "native cuDSS backend: cudssConfigSet failed for param=" << param
                 << " value=" << value << "\n";
              OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
            }
          }
        }
        if (ok) ok = (native_state->cudssDataCreate(native_state->handle, &native_state->data) == CUDSS_STATUS_SUCCESS);
        if (ok)
        {
          GetNativeStateMap()[this] = std::move(native_state);
          symbolic_initialized_ = false;
          command_handle_.clear();
          command_data_.clear();
          device_result_buffer_.backend = "cudss_native_cpp";
          device_result_buffer_.length = this->size();
          device_result_buffer_.copy_to_host_double = [this](std::vector<double> &out) -> bool {
            if (auto *st = GetNativeState(this))
            {
              if (!st->d_x)
              {
                return false;
              }
              out.resize(st->n);
              if (st->cudaMemcpy(out.data(), st->d_x, st->n * sizeof(double), CUDA_MEMCPY_DEVICE_TO_HOST) != 0)
              {
                return false;
              }
              st->d2h_bytes += st->n * sizeof(double);
              return true;
            }
            return false;
          };
          device_result_buffer_.copy_rows_to_host_double = [this](const std::vector<size_t> &rows, std::vector<double> &out) -> bool {
            if (auto *st = GetNativeState(this))
            {
              if (!st->d_x || !st->cudaMemcpy)
              {
                return false;
              }
              out.resize(rows.size());
              for (size_t i = 0; i < rows.size(); ++i)
              {
                const size_t row = rows[i];
                if (row >= st->n)
                {
                  return false;
                }
                auto *src = static_cast<const char *>(st->d_x) + row * sizeof(double);
                if (st->cudaMemcpy(&out[i], src, sizeof(double), CUDA_MEMCPY_DEVICE_TO_HOST) != 0)
                {
                  return false;
                }
                st->d2h_bytes += sizeof(double);
              }
              return true;
            }
            return false;
          };
          return true;
        }
        if (backend_policy == CuDSSBackendPolicy::NATIVE)
        {
          error_string += "native cuDSS backend initialization failed\n";
          return false;
        }
        backend_fallback_to_callback_ = true;
      }
    }
  }

  if (!oh.IsCallable())
  {
    error_string += "solver_callback not callable and native cuDSS backend unavailable\n";
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
    return ret;
  }

  ObjectHolderMap_t result_dictionary;
  auto result = interpreter.GetResult();
  ret = result.GetHashMap(result_dictionary);
  if (!ret)
  {
    error_string += "python solver object did not return a dictionary\n";
    return ret;
  }

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
    const auto matrix_format = result_dictionary["matrix_format"].GetString();
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
      error_string += R"(python solver object did not return "csc" or "csr" for "matrix_format")" "\n";
      ret = false;
    }
    const auto status = result_dictionary["status"].GetBoolean().second;
    const auto message = result_dictionary["message"].GetString();
    error_string += message;
    dsAssert(status, error_string);
  }

  command_data_ = result_dictionary["solver_object"];
  symbolic_initialized_ = false;
  return ret;
}

template <typename DoubleType>
dsMath::CompressionType CuDSSPreconditioner<DoubleType>::GetRealMatrixCompressionType() const
{
  return compression_type_;
}

template <typename DoubleType>
dsMath::CompressionType CuDSSPreconditioner<DoubleType>::GetComplexMatrixCompressionType() const
{
  return compression_type_;
}

template <typename DoubleType>
bool CuDSSPreconditioner<DoubleType>::HasDeviceResult() const
{
  return !last_device_result_token_.empty();
}

template <typename DoubleType>
std::string CuDSSPreconditioner<DoubleType>::GetDeviceResultToken() const
{
  return last_device_result_token_;
}

template <typename DoubleType>
std::string CuDSSPreconditioner<DoubleType>::GetDeviceResultLocation() const
{
  return last_device_result_location_;
}

template <typename DoubleType>
ResultView<DoubleType> CuDSSPreconditioner<DoubleType>::GetResultView() const
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
CuDSSPreconditioner<DoubleType>::~CuDSSPreconditioner()
{
  if constexpr (std::is_same_v<DoubleType, double>)
  {
    if (auto *st = GetNativeState(this))
    {
      if (profile_enabled_)
      {
        std::ostringstream so;
        so << "cuDSS transfer stats: h2d_bytes=" << st->h2d_bytes
           << " d2h_bytes=" << st->d2h_bytes
           << " analysis_calls=" << st->analysis_calls
           << " refactor_calls=" << st->refactor_calls
           << " solve_calls=" << st->solve_calls
           << " hash_reuse_hits=" << st->hash_reuse_hits
           << " hash_reuse_checks=" << st->hash_reuse_checks
           << " config_set_applied=" << st->config_set_applied
           << " stream_mode=" << (st->stream_enabled ? 1 : 0)
           << " mt_mode=" << (st->mt_mode_enabled ? 1 : 0) << "\n";
        OutputStream::WriteOut(OutputStream::OutputType::INFO, so.str());
      }
      // MT mode keeps cuDSS/gomp worker state alive beyond normal solver teardown;
      // defer full cleanup to process exit to avoid shutdown crashes.
      if (!st->mt_mode_enabled)
      {
        RemoveNativeState(this);
      }
    }
  }

  if (profile_enabled_)
  {
    if (command_handle_.IsCallable() && !command_data_.empty())
    {
      ObjectHolderMap_t stats_args = {
        {"action", ObjectHolder("stats")},
        {"solver_object", command_data_},
      };
      Interpreter interpreter;
      const bool ok = interpreter.RunCommand(command_handle_, stats_args);
      if (ok)
      {
        ObjectHolderMap_t stats_dictionary;
        const auto result = interpreter.GetResult();
        if (result.GetHashMap(stats_dictionary))
        {
          auto get_int = [&stats_dictionary](const char *key) {
            if (auto it = stats_dictionary.find(key); it != stats_dictionary.end())
            {
              return it->second.GetInteger().second;
            }
            return 0;
          };
          auto get_double = [&stats_dictionary](const char *key) {
            if (auto it = stats_dictionary.find(key); it != stats_dictionary.end())
            {
              return it->second.GetDouble().second;
            }
            return 0.0;
          };
          std::ostringstream so;
          so << "cuDSS transfer stats: h2d_bytes=" << get_int("h2d_bytes")
             << " d2h_bytes=" << get_int("d2h_bytes")
             << " analysis_calls=" << get_int("analysis_calls")
             << " refactor_calls=" << get_int("refactor_calls")
             << " solve_calls=" << get_int("solve_calls")
             << " config_set_applied=" << get_int("config_set_applied")
             << " init_calls=" << get_int("init_calls")
             << " init_reuse_hits=" << get_int("init_reuse_hits")
             << " analysis_seconds=" << get_double("analysis_seconds")
             << " factorization_seconds=" << get_double("factorization_seconds")
             << " refactor_seconds=" << get_double("refactor_seconds")
             << " factor_total_seconds=" << get_double("factor_total_seconds")
             << " solve_total_seconds=" << get_double("solve_total_seconds")
             << " solve_h2d_seconds=" << get_double("solve_h2d_seconds")
             << " solve_execute_seconds=" << get_double("solve_execute_seconds")
             << " solve_d2h_seconds=" << get_double("solve_d2h_seconds") << "\n";
          OutputStream::WriteOut(OutputStream::OutputType::INFO, so.str());
        }
      }
    }

    std::ostringstream os;
    os << "cuDSS profile: factor_calls=" << factor_calls_
       << " solve_calls=" << solve_calls_
       << " analysis_calls=" << analysis_calls_
       << " refactor_calls=" << refactor_calls_
       << " host_materialize_mode_calls=" << host_materialize_mode_calls_
       << " host_materialize_node_calls=" << host_materialize_node_calls_
       << " analysis_miss_first_factor_calls=" << analysis_miss_first_factor_calls_
       << " analysis_miss_symbolic_status_calls=" << analysis_miss_symbolic_status_calls_
       << " analysis_miss_hash_mismatch_calls=" << analysis_miss_hash_mismatch_calls_
       << " analysis_miss_dim_change_calls=" << analysis_miss_dim_change_calls_
       << " analysis_miss_backend_mode_calls=" << analysis_miss_backend_mode_calls_
       << " analysis_seconds=" << analysis_seconds_
       << " refactor_seconds=" << refactor_seconds_
       << " factor_seconds=" << factor_seconds_
       << " solve_seconds=" << solve_seconds_ << "\n";
    OutputStream::WriteOut(OutputStream::OutputType::INFO, os.str());
  }
}

template <typename DoubleType>
bool CuDSSPreconditioner<DoubleType>::DerivedLUFactor(Matrix<DoubleType> *m)
{
  const auto begin = std::chrono::steady_clock::now();
  ++factor_calls_;

  auto *cm = dynamic_cast<CompressedMatrix<DoubleType> *>(m);
  dsAssert(cm, "UNEXPECTED");
  dsAssert(cm->GetCompressionType() == compression_type_, "UNEXPECTED");

  if constexpr (std::is_same_v<DoubleType, double>)
  {
    if (auto *st = GetNativeState(this))
    {
      const bool symbolic_reports_same = symbolic_initialized_ &&
        (cm->GetSymbolicStatus() == SymbolicStatus_t::SAME_SYMBOLIC);
      bool hash_reuse_same = false;
      bool hash_reuse_mismatch = false;
      if (!symbolic_reports_same && symbolic_initialized_ && IsSymbolicHashReuseEnabled())
      {
        ++st->hash_reuse_checks;
        const auto hash_now = HashPattern(cm->GetAp(), cm->GetAi());
        if (hash_now == st->last_pattern_hash)
        {
          hash_reuse_same = true;
          ++st->hash_reuse_hits;
        }
        else
        {
          hash_reuse_mismatch = true;
        }
      }
      const bool same_symbolic = symbolic_reports_same || hash_reuse_same;

      auto cuda_check = [&st](int status, const char *name) {
        if (status != 0)
        {
          std::ostringstream os;
          os << "native cuDSS backend " << name << " failed with cuda error code " << status << "\n";
          OutputStream::WriteOut(OutputStream::OutputType::FATAL, os.str());
        }
      };
      auto cudss_check = [&st](int status, const char *name) {
        if (status != CUDSS_STATUS_SUCCESS)
        {
          std::ostringstream os;
          os << "native cuDSS backend " << name << " failed with status " << status << "\n";
          OutputStream::WriteOut(OutputStream::OutputType::FATAL, os.str());
        }
      };
      auto sync_stream = [&st, &cuda_check]() {
        if (st->stream_enabled && st->cudaStreamSynchronize)
        {
          cuda_check(st->cudaStreamSynchronize(st->stream), "cudaStreamSynchronize(cuDSS)");
        }
      };

      const bool first_factor = (st->d_ax == nullptr);
      const bool dim_changed = (!first_factor) &&
        ((st->n != this->size()) || (st->nnz != cm->GetAi().size()));
      const bool needs_pattern = first_factor || !same_symbolic || dim_changed;
      if (needs_pattern)
      {
        st->ap64.assign(cm->GetAp().begin(), cm->GetAp().end());
        st->ai64.assign(cm->GetAi().begin(), cm->GetAi().end());
        st->nnz = st->ai64.size();
        st->last_pattern_hash = HashPattern(cm->GetAp(), cm->GetAi());
      }

      const auto &ax = cm->GetAx();
      if (ax.size() != st->nnz)
      {
        std::ostringstream os;
        os << "native cuDSS backend Ax size mismatch: got " << ax.size() << " expected " << st->nnz << "\n";
        OutputStream::WriteOut(OutputStream::OutputType::FATAL, os.str());
      }

      if (needs_pattern)
      {
        if (st->mat_a) { st->cudssMatrixDestroy(st->mat_a); st->mat_a = nullptr; }
        if (st->mat_rhs) { st->cudssMatrixDestroy(st->mat_rhs); st->mat_rhs = nullptr; }
        if (st->mat_sol) { st->cudssMatrixDestroy(st->mat_sol); st->mat_sol = nullptr; }
        if (st->d_ap) { st->cudaFree(st->d_ap); st->d_ap = nullptr; }
        if (st->d_ai) { st->cudaFree(st->d_ai); st->d_ai = nullptr; }
        if (st->d_ax) { st->cudaFree(st->d_ax); st->d_ax = nullptr; }
        if (st->d_rhs) { st->cudaFree(st->d_rhs); st->d_rhs = nullptr; }
        if (st->d_x) { st->cudaFree(st->d_x); st->d_x = nullptr; }

        cuda_check(st->cudaMalloc(&st->d_ap, st->ap64.size() * sizeof(std::int64_t)), "cudaMalloc(d_ap)");
        cuda_check(st->cudaMalloc(&st->d_ai, st->ai64.size() * sizeof(std::int64_t)), "cudaMalloc(d_ai)");
        cuda_check(st->cudaMalloc(&st->d_ax, ax.size() * sizeof(double)), "cudaMalloc(d_ax)");
        cuda_check(st->cudaMalloc(&st->d_rhs, this->size() * sizeof(double)), "cudaMalloc(d_rhs)");
        cuda_check(st->cudaMalloc(&st->d_x, this->size() * sizeof(double)), "cudaMalloc(d_x)");

        cuda_check(st->cudaMemcpy(st->d_ap, st->ap64.data(), st->ap64.size() * sizeof(std::int64_t), CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(H2D,Ap)");
        cuda_check(st->cudaMemcpy(st->d_ai, st->ai64.data(), st->ai64.size() * sizeof(std::int64_t), CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(H2D,Ai)");
        cuda_check(st->cudaMemcpy(st->d_ax, ax.data(), ax.size() * sizeof(double), CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(H2D,Ax)");
        st->h2d_bytes += st->ap64.size() * sizeof(std::int64_t);
        st->h2d_bytes += st->ai64.size() * sizeof(std::int64_t);
        st->h2d_bytes += ax.size() * sizeof(double);

        cudss_check(st->cudssMatrixCreateCsr(
          &st->mat_a,
          static_cast<std::int64_t>(this->size()),
          static_cast<std::int64_t>(this->size()),
          static_cast<std::int64_t>(st->nnz),
          st->d_ap,
          nullptr,
          st->d_ai,
          st->d_ax,
          CUDA_R_64I,
          CUDA_R_64F,
          CUDSS_MTYPE_GENERAL,
          CUDSS_MVIEW_FULL,
          CUDSS_BASE_ZERO), "cudssMatrixCreateCsr");
        cudss_check(st->cudssMatrixCreateDn(
          &st->mat_rhs,
          static_cast<std::int64_t>(this->size()),
          1,
          static_cast<std::int64_t>(this->size()),
          st->d_rhs,
          CUDA_R_64F,
          CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(B)");
        cudss_check(st->cudssMatrixCreateDn(
          &st->mat_sol,
          static_cast<std::int64_t>(this->size()),
          1,
          static_cast<std::int64_t>(this->size()),
          st->d_x,
          CUDA_R_64F,
          CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(X)");
      }
      else
      {
        cuda_check(st->cudaMemcpy(st->d_ax, ax.data(), ax.size() * sizeof(double), CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(H2D,Ax)");
        st->h2d_bytes += ax.size() * sizeof(double);
      }

      const bool require_analysis = (!same_symbolic) || dim_changed;
      if (require_analysis)
      {
        if (first_factor)
        {
          ++analysis_miss_first_factor_calls_;
        }
        else
        {
          if (!symbolic_reports_same)
          {
            ++analysis_miss_symbolic_status_calls_;
          }
          if (hash_reuse_mismatch)
          {
            ++analysis_miss_hash_mismatch_calls_;
          }
          if (dim_changed)
          {
            ++analysis_miss_dim_change_calls_;
          }
        }
        ++analysis_calls_;
        ++st->analysis_calls;
        const auto phase_begin = std::chrono::steady_clock::now();
        cudss_check(st->cudssExecute(st->handle, CUDSS_PHASE_ANALYSIS, st->config, st->data, st->mat_a, st->mat_sol, st->mat_rhs), "cudssExecute(ANALYSIS)");
        cudss_check(st->cudssExecute(st->handle, CUDSS_PHASE_FACTORIZATION, st->config, st->data, st->mat_a, st->mat_sol, st->mat_rhs), "cudssExecute(FACTORIZATION)");
        sync_stream();
        analysis_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_begin).count();
      }
      else
      {
        ++refactor_calls_;
        ++st->refactor_calls;
        const auto phase_begin = std::chrono::steady_clock::now();
        cudss_check(st->cudssExecute(st->handle, CUDSS_PHASE_REFACTORIZATION, st->config, st->data, st->mat_a, st->mat_sol, st->mat_rhs), "cudssExecute(REFACTORIZATION)");
        sync_stream();
        refactor_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_begin).count();
      }
      symbolic_initialized_ = true;
      last_symbolic_nnz_ = cm->GetAi().size();
      last_symbolic_nnz_valid_ = true;
      factor_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      return true;
    }
  }

  dsAssert(command_handle_.IsCallable(), "python solver command is not callable\n");
  dsAssert(!command_data_.empty(), "python solver invalid data\n");

  const std::string return_keys[] = {
    "status",
    "message",
  };

  const bool same_symbolic = symbolic_initialized_ &&
    (cm->GetSymbolicStatus() == SymbolicStatus_t::SAME_SYMBOLIC);

  ObjectHolderMap_t factor_args = {
    {"action", ObjectHolder("factor")},
    {"solver_object", command_data_},
    {"is_same_symbolic", ObjectHolder(same_symbolic)},
  };

  if (!same_symbolic)
  {
    if (!symbolic_initialized_)
    {
      ++analysis_miss_first_factor_calls_;
    }
    else
    {
      ++analysis_miss_symbolic_status_calls_;
      if (last_symbolic_nnz_valid_ && (last_symbolic_nnz_ != cm->GetAi().size()))
      {
        ++analysis_miss_dim_change_calls_;
      }
    }
    if (backend_fallback_to_callback_)
    {
      ++analysis_miss_backend_mode_calls_;
    }
    factor_args["Ap"] = CreateIntPODArray(cm->GetAp());
    factor_args["Ai"] = CreateIntPODArray(cm->GetAi());
    ++analysis_calls_;
  }
  else
  {
    ++refactor_calls_;
  }

  if (cm->GetMatrixType() == MatrixType::COMPLEX)
  {
    const auto &ax = cm->GetAx();
    const auto &az = cm->GetAz();
    dsAssert(ax.size() == az.size(), "UNEXPECTED");
    std::vector<DoubleType> ac(2 * ax.size());
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

  Interpreter interpreter;
  bool ret = interpreter.RunCommand(command_handle_, factor_args);
  if (!ret)
  {
    std::string error = "while factorizing matrix using cuDSS solver callback\n";
    error += interpreter.GetErrorString();
    OutputStream::WriteOut(OutputStream::OutputType::FATAL, error.c_str());
  }
  else
  {
    std::string error_string;
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
      const auto status = result_dictionary["status"].GetBoolean().second;
      const auto message = result_dictionary["message"].GetString();
      error_string += message;
      dsAssert(status, error_string);
    }
  }

  const double factor_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  if (ret)
  {
    symbolic_initialized_ = true;
    last_symbolic_nnz_ = cm->GetAi().size();
    last_symbolic_nnz_valid_ = true;
    if (same_symbolic)
    {
      refactor_seconds_ += factor_elapsed;
    }
    else
    {
      analysis_seconds_ += factor_elapsed;
    }
  }
  factor_seconds_ += factor_elapsed;
  return ret;
}

template <typename DoubleType>
void CuDSSPreconditioner<DoubleType>::DerivedLUSolve(DoubleVec_t<DoubleType> &x, const DoubleVec_t<DoubleType> &b) const
{
  const auto begin = std::chrono::steady_clock::now();
  ++solve_calls_;
  const auto result_mode = GetCuDSSResultMode();
  device_result_buffer_.valid = false;

  if constexpr (std::is_same_v<DoubleType, double>)
  {
    if (auto *st = GetNativeState(this))
    {
      auto cuda_check = [&st](int status, const char *name) {
        if (status != 0)
        {
          std::ostringstream os;
          os << "native cuDSS backend " << name << " failed with cuda error code " << status << "\n";
          OutputStream::WriteOut(OutputStream::OutputType::FATAL, os.str());
        }
      };
      auto cudss_check = [&st](int status, const char *name) {
        if (status != CUDSS_STATUS_SUCCESS)
        {
          std::ostringstream os;
          os << "native cuDSS backend " << name << " failed with status " << status << "\n";
          OutputStream::WriteOut(OutputStream::OutputType::FATAL, os.str());
        }
      };
      auto sync_stream = [&st, &cuda_check]() {
        if (st->stream_enabled && st->cudaStreamSynchronize)
        {
          cuda_check(st->cudaStreamSynchronize(st->stream), "cudaStreamSynchronize(cuDSS)");
        }
      };

      const auto rhs_size_bytes = b.size() * sizeof(double);
      if (st->pinned_rhs_enabled)
      {
        if (!st->h_rhs_pinned)
        {
          cuda_check(st->cudaHostAlloc(&st->h_rhs_pinned, rhs_size_bytes, 0U), "cudaHostAlloc(rhs)");
        }
        std::memcpy(st->h_rhs_pinned, b.data(), rhs_size_bytes);
        cuda_check(st->cudaMemcpy(st->d_rhs, st->h_rhs_pinned, rhs_size_bytes, CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(H2D,RHS,pinned)");
      }
      else
      {
        cuda_check(st->cudaMemcpy(st->d_rhs, b.data(), rhs_size_bytes, CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(H2D,RHS)");
      }
      st->h2d_bytes += b.size() * sizeof(double);
      ++st->solve_calls;
      cudss_check(st->cudssExecute(st->handle, CUDSS_PHASE_SOLVE, st->config, st->data, st->mat_a, st->mat_sol, st->mat_rhs), "cudssExecute(SOLVE)");
      sync_stream();
      ++st->generation;
      device_result_buffer_.valid = true;
      device_result_buffer_.backend = "cudss_native_cpp";
      device_result_buffer_.length = b.size();
      device_result_buffer_.generation = st->generation;

      const bool mode_forces_host = (result_mode == CuDSSResultMode::HOST);
      const bool node_forces_host = NodeKeeper::instance().HaveNodes();
      const bool need_host_vector = mode_forces_host || node_forces_host;
        if (need_host_vector)
        {
        if (mode_forces_host)
        {
          ++host_materialize_mode_calls_;
        }
        if (node_forces_host)
        {
          ++host_materialize_node_calls_;
        }
          st->xcache.resize(b.size());
        if (st->pinned_rhs_enabled)
        {
          if (!st->h_x_pinned)
          {
            cuda_check(st->cudaHostAlloc(&st->h_x_pinned, b.size() * sizeof(double), 0U), "cudaHostAlloc(x)");
          }
          cuda_check(st->cudaMemcpy(st->h_x_pinned, st->d_x, b.size() * sizeof(double), CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpy(D2H,X,pinned)");
          std::memcpy(st->xcache.data(), st->h_x_pinned, b.size() * sizeof(double));
        }
        else
        {
          cuda_check(st->cudaMemcpy(st->xcache.data(), st->d_x, b.size() * sizeof(double), CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpy(D2H,X)");
        }
        st->d2h_bytes += b.size() * sizeof(double);
        x.assign(st->xcache.begin(), st->xcache.end());
        last_result_object_ = CreateDoublePODArray(st->xcache);
      }
      else
      {
        x.clear();
        last_result_object_.clear();
      }
      last_device_result_token_ = "native-" + std::to_string(st->generation);
      last_device_result_location_ = (result_mode == CuDSSResultMode::DEVICE_EXPERIMENTAL) ? "device_experimental_native" : "host";
      solve_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      return;
    }
  }

  dsAssert(command_handle_.IsCallable(), "python solver command is not callable\n");
  dsAssert(!command_data_.empty(), "python solver invalid data\n");

  const std::string return_keys[] = {
    "status",
    "message",
    "x",
  };
  const bool mode_forces_host = (result_mode == CuDSSResultMode::HOST);
  const bool node_forces_host = NodeKeeper::instance().HaveNodes();
  const bool need_host_vector = mode_forces_host || node_forces_host;
  ObjectHolderMap_t solve_args = {
    {"action", ObjectHolder("solve")},
    {"solver_object", command_data_},
    {"is_complex", ObjectHolder(false)},
    {"result_mode", ObjectHolder(ToResultModeText(result_mode))},
    {"require_host_x", ObjectHolder(need_host_vector)},
    {"b", CreateDoublePODArray(b)},
  };

  Interpreter interpreter;
  bool ret = interpreter.RunCommand(command_handle_, solve_args);
  if (!ret)
  {
    std::string error = "while solving matrix using cuDSS solver callback\n";
    error += interpreter.GetErrorString();
    OutputStream::WriteOut(OutputStream::OutputType::FATAL, error.c_str());
  }
  else
  {
    std::string error_string;
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
      const auto status = result_dictionary["status"].GetBoolean().second;
      const auto message = result_dictionary["message"].GetString();
      error_string += message;
      dsAssert(status, error_string);

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
      device_result_buffer_.valid = false;
      if (need_host_vector)
      {
        if (mode_forces_host)
        {
          ++host_materialize_mode_calls_;
        }
        if (node_forces_host)
        {
          ++host_materialize_node_calls_;
        }
        std::vector<double> xv;
        preswap(xv, x);
        xv.resize(b.size());
        bool xret = result_dictionary["x"].GetDoubleValues(xv.data(), xv.size());
        if (!xret)
        {
          xv.clear();
          xret = result_dictionary["x"].GetDoubleList(xv);
        }
        convertToType(xv, x);
        dsAssert(xret && (x.size() == b.size()), "Mismatch in returned x");
      }
      else
      {
        x.clear();
        device_result_buffer_.valid = !last_device_result_token_.empty() &&
          (last_device_result_location_.rfind("device_experimental", 0) == 0);
        if (device_result_buffer_.valid)
        {
          device_result_buffer_.backend = "cudss_callback";
          device_result_buffer_.length = b.size();
          device_result_buffer_.generation = solve_calls_;
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
            if (auto it = gather_result.find("status"); it != gather_result.end())
            {
              if (!it->second.GetBoolean().second)
              {
                return false;
              }
            }
            auto it = gather_result.find("values");
            if (it == gather_result.end())
            {
              return false;
            }
            out.resize(rows.size());
            bool ok = it->second.GetDoubleValues(out.data(), out.size());
            if (!ok)
            {
              out.clear();
              ok = it->second.GetDoubleList(out);
            }
            return ok && (out.size() == rows.size());
          };
        }
      }
    }
  }

  solve_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

template <typename DoubleType>
void CuDSSPreconditioner<DoubleType>::DerivedLUSolve(ComplexDoubleVec_t<DoubleType> &x, const ComplexDoubleVec_t<DoubleType> &b) const
{
  const auto begin = std::chrono::steady_clock::now();
  ++solve_calls_;
  const auto result_mode = GetCuDSSResultMode();
  device_result_buffer_.valid = false;

  if constexpr (std::is_same_v<DoubleType, double>)
  {
    if (GetNativeState(this))
    {
      std::ostringstream os;
      os << "native cuDSS backend currently supports real-valued solve path only\n";
      OutputStream::WriteOut(OutputStream::OutputType::FATAL, os.str());
    }
  }

  dsAssert(command_handle_.IsCallable(), "python solver command is not callable\n");
  dsAssert(!command_data_.empty(), "python solver invalid data\n");

  const std::string return_keys[] = {
    "status",
    "message",
    "x",
  };
  ObjectHolderMap_t solve_args = {
    {"action", ObjectHolder("solve")},
    {"solver_object", command_data_},
    {"is_complex", ObjectHolder(true)},
    {"result_mode", ObjectHolder(ToResultModeText(result_mode))},
    {"b", CreateComplexDoublePODArray(b)},
  };

  Interpreter interpreter;
  bool ret = interpreter.RunCommand(command_handle_, solve_args);
  if (!ret)
  {
    std::string error = "while solving matrix using cuDSS solver callback\n";
    error += interpreter.GetErrorString();
    OutputStream::WriteOut(OutputStream::OutputType::FATAL, error.c_str());
  }
  else
  {
    std::string error_string;
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
      const auto status = result_dictionary["status"].GetBoolean().second;
      const auto message = result_dictionary["message"].GetString();
      error_string += message;
      dsAssert(status, error_string);

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
      bool xret = false;
      ComplexDoubleVec_t<double> xv;
      preswap(xv, x);
      xret = result_dictionary["x"].GetComplexDoubleList(xv);
      convertToType(xv, x);
      dsAssert(xret && (x.size() == b.size()), "Mismatch in returned x");
    }
  }

  solve_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}
}

template class dsMath::CuDSSPreconditioner<double>;
#ifdef DEVSIM_EXTENDED_PRECISION
template class dsMath::CuDSSPreconditioner<float128>;
#endif
