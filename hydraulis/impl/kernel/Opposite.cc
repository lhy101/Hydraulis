#include "hydraulis/core/ndarray.h"
#include "hydraulis/core/stream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/omp_utils.h"
#include "hydraulis/impl/stream/CPUStream.h"

namespace hydraulis {
namespace impl {

template <typename spec_t>
void opposite_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    output[idx] = -input[idx];
  }
}

template <typename spec_t>
void opposite_cpu(const spec_t* input, spec_t alpha, size_t size, spec_t* output,
                  int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hydraulis::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = -input[i_idx];
  }
}

template <typename spec_t>
void opposite_cpu(const spec_t* input, size_t size, spec_t* output,
                  int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                  const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hydraulis::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hydraulis::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = -input[i_idx];
  }
}

void OppositeCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "OppositeCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      if (input->is_contiguous() && output->is_contiguous()) {
        opposite_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                             output->data_ptr<spec_t>());
      }
      else {
        opposite_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                             output->data_ptr<spec_t>(), input->ndim(),
                             input->stride().data(), output->stride().data(),
                             input->shape().data());
      }
      },"Opposite");     
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hydraulis
