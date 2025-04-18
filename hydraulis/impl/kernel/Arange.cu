#include "hydraulis/core/ndarray.h"
#include "hydraulis/impl/stream/CUDAStream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/cuda_utils.h"
#include "hydraulis/impl/kernel/Vectorized.cuh"

namespace hydraulis {
namespace impl {

void ArangeCuda(double start, double step, NDArray& output, const Stream& stream) {

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "RangeCuda", [&]() {
      launch_loop_kernel<spec_t>(output, size, stream,
                                 [start, step] __device__ (int x) -> spec_t {
                                   return static_cast<spec_t>(start + step * size_t(x));
                                 });
  });
  NDArray::MarkUsedBy({output}, stream);
}

} // namespace impl
} // namespace hydraulis
