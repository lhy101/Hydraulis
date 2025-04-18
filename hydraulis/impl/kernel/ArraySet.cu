#include "hydraulis/core/ndarray.h"
#include "hydraulis/impl/stream/CUDAStream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/cuda_utils.h"
#include "hydraulis/impl/utils/offset_calculator.cuh"
#include "hydraulis/impl/kernel/Vectorized.cuh"

namespace hydraulis {
namespace impl {

void ArraySetCuda(NDArray& data, double value, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  if (data->dtype() == kFloat4 || data->dtype() == kNFloat4) {
    NDArray::MarkUsedBy({data}, stream);
    return;
  }
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "ArraySetCuda", [&]() {
      launch_loop_kernel<spec_t>(data, size, stream,
                                 [=] __device__ (int /*idx*/) -> spec_t {
                                   return static_cast<spec_t>(value);
                                 });
  });
  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hydraulis
