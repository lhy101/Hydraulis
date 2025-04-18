#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Unary.h"

namespace hydraulis {
namespace graph {

class BoolOpImpl;
class BoolOp;

class BoolOpImpl final : public UnaryOpImpl {

 public:
  BoolOpImpl()
  : UnaryOpImpl(quote(BoolOp)) {
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  
 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeBoolOp(Tensor input, OpMeta op_meta = OpMeta());


} // namespace graph
} // namespace hydraulis
