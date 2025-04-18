#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Unary.h"

namespace hydraulis {
namespace graph {

class FloorOpImpl;
class FloorOp;

class FloorOpImpl final : public UnaryOpImpl {
 private:
  friend class FloorOp;
  struct constructor_access_key {};

 public:
  FloorOpImpl(bool inplace)
  : UnaryOpImpl(quote(FloorOp), inplace) {
  }

 protected:
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeFloorOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeFloorInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
