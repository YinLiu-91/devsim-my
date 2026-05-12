/***
DEVSIM
Copyright 2023 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef DS_EXTERNAL_PRECONDITIONER_HH
#define DS_EXTERNAL_PRECONDITIONER_HH
#include "Preconditioner.hh"
#include "ObjectHolder.hh"
#include <vector>

namespace dsMath {
enum class CompressionType;

template <typename DoubleType>
class ExternalPreconditioner : public Preconditioner<DoubleType>
{
    public:
        ExternalPreconditioner(size_t, PEnum::TransposeType_t);
        bool init (ObjectHolder, std::string &);
        dsMath::CompressionType GetRealMatrixCompressionType() const override;
        dsMath::CompressionType GetComplexMatrixCompressionType() const override;
        bool HasDeviceResult() const override;
        std::string GetDeviceResultToken() const override;
        std::string GetDeviceResultLocation() const override;
        ResultView<DoubleType> GetResultView() const override;

    protected:
        bool DerivedLUFactor(Matrix<DoubleType> *) override;
        void DerivedLUSolve(DoubleVec_t<DoubleType> &x, const DoubleVec_t<DoubleType> &b) const override;
        void DerivedLUSolve(ComplexDoubleVec_t<DoubleType> &x, const ComplexDoubleVec_t<DoubleType> &b) const override;

        ~ExternalPreconditioner();

    private:
        ExternalPreconditioner() = delete;
        ExternalPreconditioner(const ExternalPreconditioner &) = delete;
        ExternalPreconditioner &operator= (const ExternalPreconditioner &) = delete;

        struct FactorCallProfile {
          double total_seconds = 0.0;
          double build_seconds = 0.0;
          double interpreter_seconds = 0.0;
          double result_seconds = 0.0;
          double validate_seconds = 0.0;
          double other_seconds = 0.0;
        };
        struct SolveCallProfile {
          double total_seconds = 0.0;
          double build_seconds = 0.0;
          double interpreter_seconds = 0.0;
          double result_seconds = 0.0;
          double validate_seconds = 0.0;
          double device_meta_seconds = 0.0;
          double materialize_seconds = 0.0;
          double device_view_seconds = 0.0;
          double other_seconds = 0.0;
          int need_host_vector = 0;
          int device_view_enabled = 0;
        };

        mutable ObjectHolder command_handle_;
        mutable ObjectHolder command_data_;
        mutable ObjectHolder last_result_object_;
        mutable DeviceResultBuffer device_result_buffer_;
        mutable std::string last_device_result_token_;
        mutable std::string last_device_result_location_;
        mutable std::uint64_t result_generation_;
        mutable bool profile_enabled_ = false;
        mutable double factor_total_seconds_ = 0.0;
        mutable double factor_build_seconds_ = 0.0;
        mutable double factor_interpreter_seconds_ = 0.0;
        mutable double factor_result_seconds_ = 0.0;
        mutable double factor_validate_seconds_ = 0.0;
        mutable double factor_parse_seconds_ = 0.0;
        mutable double solve_total_seconds_ = 0.0;
        mutable double solve_build_seconds_ = 0.0;
        mutable double solve_interpreter_seconds_ = 0.0;
        mutable double solve_result_seconds_ = 0.0;
        mutable double solve_validate_seconds_ = 0.0;
        mutable double solve_device_meta_seconds_ = 0.0;
        mutable double solve_parse_seconds_ = 0.0;
        mutable double solve_materialize_seconds_ = 0.0;
        mutable double solve_device_view_seconds_ = 0.0;
        mutable size_t factor_calls_ = 0;
        mutable size_t solve_calls_ = 0;
        mutable std::vector<FactorCallProfile> factor_call_profiles_;
        mutable std::vector<SolveCallProfile> solve_call_profiles_;
        CompressionType compression_type_;
};
}

#endif
