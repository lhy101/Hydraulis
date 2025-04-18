#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Pool.h"

namespace hydraulis {
namespace graph {

class MaxPoolOpImpl;
class MaxPoolGradientOpImpl;

class MaxPoolOpImpl final : public PoolOpImpl {

 public:
  MaxPoolOpImpl(size_t kernel_H, size_t kernel_W, 
                size_t padding, size_t stride,
                OpMeta op_meta = OpMeta())
  : PoolOpImpl(quote(MaxPoolOp), kernel_H, kernel_W,
               padding, stride) {}

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return PoolOpImpl::operator==(rhs);
  }
};

class MaxPoolGradientOpImpl final : public PoolGradientOpImpl {
 public:
  MaxPoolGradientOpImpl(size_t kernel_H,
                        size_t kernel_W, size_t padding, size_t stride,
                        OpMeta op_meta = OpMeta())
  : PoolGradientOpImpl(quote(MaxPoolGradientOp), kernel_H, kernel_W,
                       padding, stride) {}

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return PoolGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeMaxPoolOp(Tensor input, size_t kernel_H, size_t kernel_W, 
                     size_t padding, size_t stride,
                     OpMeta op_meta = OpMeta());

Tensor MakeMaxPoolGradientOp(Tensor output, Tensor output_grad, Tensor input,
                             size_t kernel_H, size_t kernel_W, size_t padding,
                             size_t stride, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
