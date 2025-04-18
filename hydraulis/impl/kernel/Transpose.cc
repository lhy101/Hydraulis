#include "hydraulis/core/ndarray.h"
#include "hydraulis/core/stream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/omp_utils.h"
#include "hydraulis/impl/stream/CPUStream.h"

namespace hydraulis {
namespace impl {

// Out-of-place version of transpose and its gradient
/* It is replaced with in-place version. */
template <typename spec_t>
void transpose_cpu(const spec_t* input, spec_t* output, const int64_t* buf,
                   uint32_t ndims, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    const auto* in_strides = buf;
    const auto* out_strides = buf + ndims;
    const auto* perm = buf + ndims * 2;
    uint32_t i_idx = 0;
    uint32_t t = idx;
    for (uint32_t i = 0; i < ndims; ++i) {
      const uint32_t ratio = t / out_strides[i];
      t -= ratio * out_strides[i];
      i_idx += ratio * in_strides[perm[i]];
    }
    output[idx] = input[i_idx];
  }
}

void transpose_quantization(const uint8_t* input, uint8_t* output, const int64_t* buf,
                            uint32_t ndims, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size / 2; ++idx) {
    output[idx] = 0;
  }
  for (size_t idx = 0; idx < size; ++idx) {
    const auto* in_strides = buf;
    const auto* out_strides = buf + ndims;
    const auto* perm = buf + ndims * 2;
    uint32_t i_idx = 0;
    uint32_t t = idx;
    for (uint32_t i = 0; i < ndims; ++i) {
      const uint32_t ratio = t / out_strides[i];
      t -= ratio * out_strides[i];
      i_idx += ratio * in_strides[perm[i]];
    }
    int tmp = 0;
    if (i_idx % 2 == 0)
      tmp = input[i_idx / 2] >> 4;
    else 
      tmp = input[i_idx / 2] & (0x0F);
    if (idx % 2 == 0)
      output[idx / 2] += (tmp << 4);
    else 
      output[idx / 2] += tmp;
  }
}

void TransposeCpu(const NDArray& input, NDArray& output, const HTAxes& perm,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  auto ndim = static_cast<uint32_t>(input->ndim());
  auto ndim_ = static_cast<uint32_t>(output->ndim());
  HT_ASSERT(ndim == ndim_);
  HTShape buf(3 * ndim);
  int64_t in_stride = 1;
  int64_t out_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buf[i] = in_stride;
    buf[ndim + i] = out_stride;
    buf[ndim * 2 + i] = perm[i];
    in_stride *= input->shape(i);
    out_stride *= output->shape(i);
  }
  HT_ASSERT(in_stride == out_stride);
  size_t size = in_stride;
  if (size == 0)
    return;
  if (input->dtype() == kFloat4 || input->dtype() == kNFloat4) {
    auto _future = cpu_stream.EnqueueTask(
        [input, output, buf, ndim ,size]() {
        transpose_quantization(input->data_ptr<uint8_t>(), output->data_ptr<uint8_t>(),
                               buf.data(), ndim, size);
        }, "Transpose");
  }
  else {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "TransposeCpu", [&]() {
        cpu_stream.EnqueueTask(
          [input, output, buf, ndim ,size]() {
          transpose_cpu<spec_t>(input->data_ptr<spec_t>(),
                                output->data_ptr<spec_t>(), buf.data(), ndim, size);
          },"Transpose");
      });
  }
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hydraulis
