#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void PowCuda(const NDArray& input, double exponent, NDArray& output,
             const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "PowCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t>(input, output, size, stream,
                                         [exponent] __device__ (spec_t x) -> spec_t {
                                           return hetu::cuda::cuda_pow(x, static_cast<spec_t>(exponent));
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu