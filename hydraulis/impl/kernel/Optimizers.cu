#include "hydraulis/core/ndarray.h"
#include "hydraulis/impl/stream/CUDAStream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/ndarray_utils.h"
#include "hydraulis/impl/utils/cuda_utils.h"
#include "hydraulis/impl/utils/cuda_math.h"
#include "hydraulis/impl/utils/offset_calculator.cuh"
#include "hydraulis/impl/kernel/Vectorized.cuh"

namespace hydraulis {
namespace impl {

void SGDUpdateCuda(const NDArray& grad, NDArray& param, NDArray& velocity,
                   float lr, float momentum, bool nesterov,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_CUDA_DEVICE(param);
  HT_ASSERT_SAME_DEVICE(grad, param);
  HT_ASSERT_SAME_SHAPE(grad, param);
  if (momentum != 0) {
    HT_ASSERT_CUDA_DEVICE(velocity);
    HT_ASSERT_SAME_DEVICE(velocity, param);
    HT_ASSERT_SAME_SHAPE(velocity, param);
  }
  size_t size = grad->numel();
  if (size == 0)
    return;
  NDArray grad_;
  if (grad->dtype() != param->dtype())
    grad_ = NDArray::to(grad, param->device(), param->dtype(), stream.stream_index());
  else
    grad_ = grad;
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "SGDUpdateCuda", [&]() {
    if (momentum == 0) {
      launch_loop_kernel<spec_t, spec_t, spec_t>(param, grad, param, size, stream,
                                                 [lr] __device__ (spec_t param, spec_t grad) -> spec_t {
                                                   return param - static_cast<spec_t>(lr) * grad;
                                                 });
    } else {
      if (!nesterov) {
        launch_loop_kernel_multiple_outputs<std::tuple<spec_t, spec_t, spec_t>, thrust::tuple<spec_t, spec_t>>
                                          ({velocity, grad_, param}, {velocity, param}, size, stream,
                                            [=] __device__ (spec_t velocity, spec_t grad, spec_t param)
                                              -> thrust::tuple<spec_t, spec_t> {
                                                auto update_velocity = velocity * momentum - grad * lr;
                                                auto update_param = param + update_velocity;
                                                return thrust::tuple<spec_t, spec_t>{
                                                  update_velocity,
                                                  update_param
                                                };
                                            });
      } else {
        launch_loop_kernel_multiple_outputs<std::tuple<spec_t, spec_t, spec_t>, thrust::tuple<spec_t, spec_t>>
                                           ({velocity, grad_, param}, {velocity, param}, size, stream,
                                            [=] __device__ (spec_t velocity, spec_t grad, spec_t param)
                                              -> thrust::tuple<spec_t, spec_t> {
                                                float temp = lr * grad;
                                                auto update_velocity = momentum * (velocity - temp);
                                                auto update_param = param + (update_velocity - temp);
                                                return thrust::tuple<spec_t, spec_t>{
                                                  update_velocity,
                                                  update_param
                                                };
                                            });
      }
    }
  });
  NDArray::MarkUsedBy({grad, grad_, param, velocity}, stream);
}

void SGDUpdateWithGradScalerCuda(const NDArray& grad, const NDArray& infinite_count,
                                 NDArray& param, NDArray& velocity,
                                 float lr, float momentum, bool nesterov,
                                 const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_CUDA_DEVICE(param);
  HT_ASSERT_SAME_DEVICE(grad, param);
  HT_ASSERT_SAME_SHAPE(grad, param);
  if (momentum != 0) {
    HT_ASSERT_CUDA_DEVICE(velocity);
    HT_ASSERT_SAME_DEVICE(velocity, param);
    HT_ASSERT_SAME_SHAPE(velocity, param);
  }
  size_t size = grad->numel();
  if (size == 0)
    return;
  NDArray grad_;
  if (grad->dtype() != param->dtype())
    grad_ = NDArray::to(grad, param->device(), param->dtype(), stream.stream_index());
  else
    grad_ = grad;
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "SGDUpdateCuda", [&]() {
    auto infinite_count_host_arr = NDArray::to(infinite_count, kCPU, kFloat32, stream.stream_index());
    auto infinite_count_host = infinite_count_host_arr->data_ptr<float>()[0];
    if (infinite_count_host) {
      NDArray::MarkUsedBy({infinite_count_host_arr}, stream);
      return;
    }
    if (momentum == 0) {
      launch_loop_kernel<spec_t, spec_t, spec_t>(param, grad_, param, size, stream,
                                                 [=] __device__ (spec_t param, spec_t grad) -> spec_t {
                                                   return param - lr * grad;
                                                 });
    } else {
      if (!nesterov) {
        launch_loop_kernel_multiple_outputs<std::tuple<spec_t, spec_t, spec_t>, thrust::tuple<spec_t, spec_t>>
                                           ({velocity, grad_, param}, {velocity, param}, size, stream,
                                            [=] __device__ (spec_t velocity, spec_t grad, spec_t param)
                                              -> thrust::tuple<spec_t, spec_t> {
                                                auto update_velocity = momentum * velocity - lr * grad;
                                                auto update_param = param + update_velocity;
                                                return thrust::tuple<spec_t, spec_t>{
                                                  update_velocity,
                                                  update_param
                                                };
                                            });
      } else {
        launch_loop_kernel_multiple_outputs<std::tuple<spec_t, spec_t, spec_t>, thrust::tuple<spec_t, spec_t>>
                                           ({velocity, grad_, param}, {velocity, param}, size, stream,
                                            [=] __device__ (spec_t velocity, spec_t grad, spec_t param)
                                              -> thrust::tuple<spec_t, spec_t> {
                                                float temp = lr * grad;
                                                auto update_velocity = momentum * (velocity - temp);
                                                auto update_param = param + (update_velocity - temp);
                                                return thrust::tuple<spec_t, spec_t>{
                                                  update_velocity,
                                                  update_param
                                                };
                                            });
      }
      NDArray::MarkUsedBy({infinite_count_host_arr}, stream);
    }
  });
  NDArray::MarkUsedBy({grad, grad_, param, velocity}, stream);
}

void AdamCuda(const NDArray& grad, NDArray& param, NDArray& mean,
              NDArray& variance, NDArray& step, 
              float lr, float beta1, float beta2,
              float eps, float weight_decay, bool update_step,
              const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_CUDA_DEVICE(param);
  HT_ASSERT_CUDA_DEVICE(mean);
  HT_ASSERT_CUDA_DEVICE(variance);
  HT_ASSERT_SAME_DEVICE(grad, param);
  HT_ASSERT_SAME_DEVICE(grad, mean);
  HT_ASSERT_SAME_DEVICE(grad, variance);
  size_t size = grad->numel();
  if (size == 0)
    return;
  NDArray grad_;
  if (grad->dtype() != param->dtype())
    grad_ = NDArray::to(grad, param->device(), param->dtype(), stream.stream_index());
  else
    grad_ = grad;
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "AdamUpdateCuda", [&]() {
    int64_t cur_step = step->data_ptr<int64_t>()[0];
    launch_loop_kernel_multiple_outputs<std::tuple<spec_t, spec_t, spec_t, spec_t>, thrust::tuple<spec_t, spec_t, spec_t>>
                                       ({grad_, param, mean, variance}, {param, mean, variance}, size, stream,
                                        [=] __device__ (spec_t grad, spec_t param, spec_t mean, spec_t variance)
                                          -> thrust::tuple<spec_t, spec_t, spec_t> {
                                            auto update_mean = mean * beta1 + grad * (1 - beta1);
                                            auto update_variance = variance * beta2 + grad * grad * (1 - beta2);
                                            spec_t bias1 = spec_t(1 - hydraulis::cuda::cuda_pow(beta1, float(cur_step)));
                                            spec_t bias2 = hydraulis::cuda::cuda_sqrt(spec_t(1 - hydraulis::cuda::cuda_pow(beta2, float(cur_step))));
                                            auto update_param = param - (lr * (update_mean / bias1) / 
                                                      (hydraulis::cuda::cuda_sqrt(update_variance) / bias2 + eps));
                                            return thrust::tuple<spec_t, spec_t, spec_t>{
                                              update_param,
                                              update_mean,
                                              update_variance
                                            };
                                        });
  });
  if (update_step)
    step->data_ptr<int64_t>()[0] = (step->data_ptr<int64_t>()[0] + 1);
  NDArray::MarkUsedBy({grad, grad_, param, mean, variance, step}, stream);
}

} // namespace impl
} // namespace hydraulis
