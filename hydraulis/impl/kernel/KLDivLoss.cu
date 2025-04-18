#include "hydraulis/core/ndarray.h"
#include "hydraulis/impl/stream/CUDAStream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/cuda_utils.h"
#include "hydraulis/impl/utils/cuda_math.h"
#include "hydraulis/impl/utils/offset_calculator.cuh"
#include "hydraulis/impl/kernel/Vectorized.cuh"

namespace hydraulis {
namespace impl {

void KLDivLossCuda(const NDArray& pred, const NDArray& label,
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
    pred->dtype(), spec_t, "KLDivLossCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t, spec_t>(
        pred, label, loss, n_rows, stream,
        [] __device__ (spec_t pred, spec_t label) {
          spec_t lglabel = hydraulis::cuda::cuda_log(label);
          // clip to -100 following PyTorch
          spec_t min_value = -100;
          return label * (hydraulis::cuda::cuda_max(lglabel, min_value) - pred);
        });
  });
  NDArray::MarkUsedBy({pred, label, loss}, stream);
}

void KLDivLossGradientCuda(const NDArray& pred, const NDArray& label,
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
    label->dtype(), spec_t, "KLDivLossGradientCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t, spec_t>(
        label, grad_loss, output, n_rows, stream,
        [] __device__ (spec_t label, spec_t grad_loss) {
          return - grad_loss * label;
        });
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hydraulis
