#pragma once
#include "hydraulis/impl/utils/cuda_utils.h"

namespace hydraulis {
namespace impl {

template<typename spec_a_t, typename spec_b_t>
struct kplus{
  __forceinline__ __host__ __device__ 
  spec_a_t operator() (spec_a_t a, spec_b_t b) {
    return a + b;
  }
};

template<typename spec_a_t, typename spec_b_t>
struct kminus{
  __forceinline__ __host__ __device__ 
  spec_a_t operator() (spec_a_t a, spec_b_t b) {
    return a - b;
  }
};

template<typename spec_a_t, typename spec_b_t>
struct kmultiplies{
  __forceinline__ __host__ __device__ 
  spec_a_t operator() (spec_a_t a, spec_b_t b) {
    return a * b;
  }
};

template<typename spec_a_t, typename spec_b_t>
struct kdivides{
  __forceinline__ __host__ __device__ 
  spec_a_t operator() (spec_a_t a, spec_b_t b) {
    return a / b;
  }
};


} // namespace impl
} // namespace hydraulis
