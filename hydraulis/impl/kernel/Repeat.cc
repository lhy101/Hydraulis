#include "hydraulis/core/ndarray.h"
#include "hydraulis/core/stream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/omp_utils.h"
#include "hydraulis/impl/stream/CPUStream.h"

namespace hydraulis {
namespace impl {

template <typename spec_t>
void repeat_cpu(const spec_t *input, spec_t *output, size_t size,
                const int64_t *stride_in, const int64_t *stride_out,
                const int64_t *dim, int ndim) {
  for (size_t idx = 0; idx < size; ++idx) {
    int index = 0;
    size_t ind = idx;
    for (int i = 0; i < ndim; i++) {
      int tmp_index = ind / stride_out[i];
      index += (tmp_index % dim[i]) * stride_in[i];
      ind -= tmp_index * stride_out[i];
    }
    output[idx] = input[index];
  }
}


void RepeatCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  int ndim = output->ndim();
  HTStride stride_tmp(ndim);
  HTShape shape_tmp(ndim);
  for (int i = 0; i < ndim; i++) {
      if (i < int(ndim - input->ndim())) {
          stride_tmp[i] = input->stride()[0];
          shape_tmp[i] = 1;
      } else {
          stride_tmp[i] = input->stride()[i - (ndim - input->ndim())];
          shape_tmp[i] = input->shape(i - (ndim - input->ndim()));
      }
  }
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RepeatCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, output, size, stride_tmp, shape_tmp, ndim]() {
        repeat_cpu<spec_t>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size, stride_tmp.data(), 
        output->stride().data(), shape_tmp.data(), ndim);
        },"Repeat");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void array_zero_set_cpu(spec_t* input, size_t size) {
  for (size_t idx = 0; idx < size; ++idx)
    input[idx] = 0;
}

template <typename spec_t>
void repeat_gradient_cpu(const spec_t *input, spec_t *output, size_t size,
                         const int64_t *stride_in, const int64_t *stride_out,
                         const int64_t *dim, int ndim) {
  for (size_t idx = 0; idx < size; ++idx) {
    int index = 0;
    size_t ind = idx;
    for (int i = 0; i < ndim; i++) {
      int tmp_index = ind / stride_out[i];
      index += (tmp_index % dim[i]) * stride_in[i];
      ind -= tmp_index * stride_out[i];
    }
    output[index] += input[idx];
  }
}

void RepeatGradientCpu(const NDArray& output, NDArray& input, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  int ndim = output->ndim();
  HTStride stride_tmp(ndim);
  HTShape shape_tmp(ndim);
  for (int i = 0; i < ndim; i++) {
      if (i < int(ndim - input->ndim())) {
          stride_tmp[i] = input->stride()[0];
          shape_tmp[i] = 1;
      } else {
          stride_tmp[i] = input->stride()[i - (ndim - input->ndim())];
          shape_tmp[i] = input->shape(i - (ndim - input->ndim()));
      }
  }
  if (size == 0)
    return;

  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RepeatGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, output, size, stride_tmp, shape_tmp, ndim]() {
        array_zero_set_cpu<spec_t>(
                input->data_ptr<spec_t>(), input->numel());
        repeat_gradient_cpu<spec_t>(
        output->data_ptr<spec_t>(), input->data_ptr<spec_t>(), size, stride_tmp.data(), 
        output->stride().data(), shape_tmp.data(), ndim);
        },"RepeatGradient");
      
    });
  NDArray::MarkUsedBy({input, output}, stream);
}


} // namespace impl
} // namespace hydraulis
