/***
DEVSIM
Copyright 2013 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/

#ifndef DS_MATH_TYPES_HH
#define DS_MATH_TYPES_HH
#ifdef DEVSIM_EXTENDED_PRECISION
#include "Float128.hh"
#endif
#include "../common_api/ObjectHolder.hh"
#include <vector>
#include <complex>
#include <string>
#include <memory>
#include <functional>
#include <cstdint>
namespace dsMath
{

template <typename T>
struct ComplexTypeWrapper
{
};

template <>
struct ComplexTypeWrapper<double>
{
    using type = std::complex<double>;
};

#ifdef DEVSIM_EXTENDED_PRECISION
template <>
struct ComplexTypeWrapper<float128>
{
    using type = complex128;
};
#endif

template <typename DoubleType>
using ComplexDouble_t = typename ComplexTypeWrapper<DoubleType>::type;

template <typename DoubleType>
using ComplexDoubleVec_t = std::vector<ComplexDouble_t<DoubleType>>;

template <typename DoubleType>
using DoubleVec_t = std::vector<DoubleType>;

struct DeviceResultBuffer
{
    bool valid = false;
    std::string backend;
    size_t length = 0;
    std::uint64_t generation = 0;
    std::function<bool(std::vector<double> &)> copy_to_host_double;
    std::function<bool(const std::vector<size_t> &, std::vector<double> &)> copy_rows_to_host_double;
};

template <typename DoubleType>
struct ResultView
{
    const DoubleVec_t<DoubleType> *host_values = nullptr;
    const ObjectHolder *host_payload = nullptr;
    mutable std::shared_ptr<DoubleVec_t<DoubleType>> host_cached_values;
    const DeviceResultBuffer *device_result = nullptr;
    bool has_device_values = false;
    std::string device_token;
    std::string location = "host";
};

typedef std::vector<int>    IntVec_t;

}
#endif
