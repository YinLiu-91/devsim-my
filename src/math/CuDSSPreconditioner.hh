/***
 * DEVSIM
 * SPDX-License-Identifier: Apache-2.0
 *
 * Draft interface for a future native cuDSS preconditioner.
 *
 * This file is intentionally not wired into the build yet. It captures the
 * lower-overhead C++ path that could replace the Python callback once the
 * design is stabilized.
 ***/

#ifndef DS_CUDSS_PRECONDITIONER_HH
#define DS_CUDSS_PRECONDITIONER_HH

#include "ObjectHolder.hh"
#include "Preconditioner.hh"

namespace dsMath {
enum class CompressionType;

template <typename DoubleType>
class CuDSSPreconditioner : public Preconditioner<DoubleType>
{
  public:
    CuDSSPreconditioner(size_t, PEnum::TransposeType_t);
    bool init(ObjectHolder, std::string &);
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

    ~CuDSSPreconditioner();

  private:
    CuDSSPreconditioner() = delete;
    CuDSSPreconditioner(const CuDSSPreconditioner &) = delete;
    CuDSSPreconditioner &operator=(const CuDSSPreconditioner &) = delete;

    mutable ObjectHolder command_handle_;
    mutable ObjectHolder command_data_;
    mutable ObjectHolder last_result_object_;
    mutable DeviceResultBuffer device_result_buffer_;
    CompressionType compression_type_;
    mutable bool symbolic_initialized_;
    mutable std::string last_device_result_token_;
    mutable std::string last_device_result_location_;

    bool profile_enabled_;
    mutable size_t factor_calls_;
    mutable size_t solve_calls_;
    mutable size_t analysis_calls_;
    mutable size_t refactor_calls_;
    mutable size_t host_materialize_mode_calls_;
    mutable size_t host_materialize_node_calls_;
    mutable size_t analysis_miss_first_factor_calls_;
    mutable size_t analysis_miss_symbolic_status_calls_;
    mutable size_t analysis_miss_hash_mismatch_calls_;
    mutable size_t analysis_miss_dim_change_calls_;
    mutable size_t analysis_miss_backend_mode_calls_;
    mutable size_t last_symbolic_nnz_;
    mutable bool last_symbolic_nnz_valid_;
    mutable double analysis_seconds_;
    mutable double refactor_seconds_;
    mutable double factor_seconds_;
    mutable double solve_seconds_;
    bool backend_fallback_to_callback_;
};

}

#endif
