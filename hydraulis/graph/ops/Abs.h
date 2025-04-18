#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Unary.h"

namespace hydraulis {
namespace graph {

class AbsOpImpl;
class AbsOp;
class AbsGradientOpImpl;
class AbsGradientOp;

class AbsOpImpl final : public UnaryOpImpl {
 private:
  friend class AbsOp;
  struct constructor_access_key {};

 public:
  AbsOpImpl(bool inplace)
  : UnaryOpImpl(quote(AbsOp), inplace) {
  }

 protected:
  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeAbsOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeAbsInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class AbsGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  AbsGradientOpImpl()
  : UnaryGradientOpImpl(quote(AbsGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeAbsGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
