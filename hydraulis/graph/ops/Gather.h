#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"

namespace hydraulis {
namespace graph {

class GatherOpImpl;
class GatherOp;
class GatherGradientOpImpl;
class GatherGradientOp;

class GatherOpImpl final : public OpInterface {

 public:
  GatherOpImpl(int64_t dim)
  : OpInterface(quote(GatherOp)), _dim(dim) {
  }

  int64_t get_dim() const {
    return _dim;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(inputs[1]->shape())
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  int64_t _dim;
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const GatherOpImpl&>(rhs);
      return (get_dim() == rhs_.get_dim());
    }
    return false;
  }
};

Tensor MakeGatherOp(Tensor input, int64_t dim, Tensor id, OpMeta op_meta = OpMeta());

class GatherGradientOpImpl final : public OpInterface {

 public:
  GatherGradientOpImpl(int64_t dim)
  : OpInterface(quote(GatherGradientOp)),
  _dim(dim) {
  }

  int64_t get_dim() const {
    return _dim;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;
                      
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  int64_t _dim;
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const GatherGradientOpImpl&>(rhs);
      return (get_dim() == rhs_.get_dim());
    }
    return false;
  }
};

Tensor MakeGatherGradientOp(Tensor grad_output, int64_t dim, Tensor id, Tensor input,
                            OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
