#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"

namespace hydraulis {
namespace graph {

class MatDotOpImpl;
class MatDotOp;

class MatDotOpImpl final : public OpInterface {
 public:
  MatDotOpImpl(int64_t axes = 0)
  : OpInterface(quote(MatDotOp)) {
  }

  int64_t get_axes() const {
    return _axes;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;
  
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  int64_t _axes;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MatDotOpImpl&>(rhs);
      return (get_axes() == rhs_.get_axes()); 
    }
    return false;
  }
};

Tensor MakeMatDotOp(Tensor a, Tensor b, int64_t axes = 0,
                    OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
