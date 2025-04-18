#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"

namespace hydraulis {
namespace graph {

class RepeatOpImpl;
class RepeatOp;
class RepeatGradientOpImpl;
class RepeatGradientOp;

class RepeatOpImpl final : public OpInterface {
 public:
  RepeatOpImpl(HTShape repeats)
  : OpInterface(quote(RepeatOp)),
  _repeats(repeats){
  }

  inline HTShape repeats() const{
    return _repeats;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape output_shape = repeats();
    HT_ASSERT(output_shape.size() >= inputs[0]->ndim())
    << output_shape << " has dim " << output_shape.size()
    << ", but " << inputs[0]->shape() << " has dim " << inputs[0]->ndim();
    for (size_t i = 0; i < inputs[0]->ndim(); ++i) {
      if (inputs[0]->shape(i) > 0)
      output_shape[i + output_shape.size() - inputs[0]->ndim()] *= inputs[0]->shape(i); 
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTShape _repeats;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RepeatOpImpl&>(rhs);
      return (repeats() == rhs_.repeats());
    }
    return false;
  }
};

Tensor MakeRepeatOp(Tensor input, HTShape repeats, OpMeta op_meta = OpMeta());

class RepeatGradientOpImpl final : public OpInterface {

 public:
  RepeatGradientOpImpl()
  : OpInterface(quote(RepeatGradientOp)) {
  }


 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[1]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override; 

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

Tensor MakeRepeatGradientOp(Tensor grad_output, Tensor input,
                            OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
