#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Unary.h"

namespace hydraulis {
namespace graph {

class LogsigmoidOpImpl;
class LogsigmoidOp;
class LogsigmoidGradientOpImpl;
class LogsigmoidGradientOp;

class LogsigmoidOpImpl final : public UnaryOpImpl {
 private:
  friend class LogsigmoidOp;
  struct constructor_access_key {};

 public:
  LogsigmoidOpImpl()
  : UnaryOpImpl(quote(LogsigmoidOp)){
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs); 
  }
};

Tensor MakeLogsigmoidOp(Tensor input, OpMeta op_meta = OpMeta());

class LogsigmoidGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  LogsigmoidGradientOpImpl()
  : UnaryGradientOpImpl(quote(LogsigmoidGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeLogsigmoidGradientOp(Tensor input, Tensor grad_output,
                                OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
