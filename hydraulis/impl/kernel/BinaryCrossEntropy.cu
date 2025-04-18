#include "hydraulis/core/ndarray.h"
#include "hydraulis/impl/stream/CUDAStream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/cuda_utils.h"
#include "hydraulis/impl/utils/cuda_math.h"
#include "hydraulis/impl/utils/offset_calculator.cuh"
#include "hydraulis/impl/kernel/Vectorized.cuh"

namespace hydraulis {
namespace impl {

template <typename spec_t>
struct kBCEGradFunctor {
  __device__ spec_t operator()(spec_t pred, spec_t label, spec_t grad_loss) {
    spec_t one = 1.0f;
    spec_t denominator = pred * (one - pred);
    return grad_loss * (pred - label) / MAX(denominator, spec_t(1e-12));
  }
};

template <>
struct kBCEGradFunctor<bfloat16> {
  __device__ bfloat16 operator()(bfloat16 pred, bfloat16 label, bfloat16 grad_loss) {
    bfloat16 one = 1.0f;
    bfloat16 denominator = pred * (one - pred);
    return grad_loss * (pred - label) / denominator;
  }
};

void BinaryCrossEntropyCuda(const NDArray& pred, const NDArray& label,
                            NDArray& loss, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, loss);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "BinaryCrossEntropyCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t, spec_t>(
        pred, label, loss, n_rows, stream,
        [] __device__ (spec_t pred, spec_t label) {
          spec_t v1 = hydraulis::cuda::cuda_log(pred);
          spec_t v2 = hydraulis::cuda::cuda_log(1 - pred);
          // clip to -100 following PyTorch
          spec_t min_value = -100;
          return -label * hydraulis::cuda::cuda_max(v1, min_value) - (1 - label) * hydraulis::cuda::cuda_max(v2, min_value);
        });
    });
  NDArray::MarkUsedBy({pred, label, loss}, stream);
}

void BinaryCrossEntropyGradientCuda(const NDArray& pred, const NDArray& label,
                                    const NDArray& grad_loss, NDArray& output,
                                    const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, grad_loss);
  HT_ASSERT_SAME_NDIM(pred, output);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "BinaryCrossEntropyGradientCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t, spec_t, spec_t>(
        pred, label, grad_loss, output, n_rows, stream,
        kBCEGradFunctor<spec_t>());
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hydraulis
