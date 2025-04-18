#include "hydraulis/core/ndarray.h"
#include "hydraulis/impl/stream/CUDAStream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/cuda_utils.h"
#include "hydraulis/impl/kernel/Vectorized.cuh"

namespace hydraulis {
namespace impl {

void RangeMaskCuda(const NDArray& input, int64_t min, int64_t max,
                  NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "RangeMaskCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t>(input, output, size, stream,
                                         [min, max] __device__ (spec_t x) -> spec_t {
                                           spec_t zero = 0;
                                           spec_t one = 1.0f;
                                           return ((static_cast<int64_t>(x) >= min) &&
                                                   (static_cast<int64_t>(x) <= max)) ? zero : one;
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hydraulis
