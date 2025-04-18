#pragma once

#include "hydraulis/graph/operator.h"

namespace hydraulis {
namespace graph {

class OptimizerUpdateOpInterface : public OpInterface {
 public:
  OptimizerUpdateOpInterface(OpType&& op_type, float learning_rate)
  : OpInterface(std::move(op_type)), _learning_rate(learning_rate) {
    HT_VALUE_ERROR_IF(_learning_rate < 0)
      << "Invalid learning rate: " << _learning_rate;
  }

  uint64_t op_indicator() const noexcept override {
    return OPTIMIZER_UPDATE_OP;
  }

  bool inplace_at(size_t input_position) const override {
    // By default, the first input is parameter, the second is gradient,
    // and the rest are optimizer states.
    return input_position != 1;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    // Question: should we check whether the param is trainable?
    HT_VALUE_ERROR_IF(!inputs.front()->producer()->is_parameter())
      << "The first input " << inputs.front() << " is not a parameter";
    return {inputs.front()->meta()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes.front()};
  }

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const override {
    // In place update
    return {inputs.front()};
  }

  bool DoMapToParallelDevices(Operator& op, const DeviceGroupUnion& placement_group_union) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ =
        reinterpret_cast<const OptimizerUpdateOpInterface&>(rhs);
      return learning_rate() == rhs_.learning_rate();
    }
    return false;
  }

  float learning_rate() const {
    return _learning_rate;
  }

 protected:
  float _learning_rate;
};

class SGDUpdateOpImpl final : public OptimizerUpdateOpInterface {
 public:
  SGDUpdateOpImpl(float learning_rate)
  : OptimizerUpdateOpInterface(quote(SGDUpdateOp), learning_rate) {}

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
};

class SGDUpdateWithGradScalerOpImpl final : public OptimizerUpdateOpInterface {
 public:
  SGDUpdateWithGradScalerOpImpl(float learning_rate)
  : OptimizerUpdateOpInterface(quote(SGDUpdateWithGradScalerOp), learning_rate) {}

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
};

class MomentumUpdateOpImpl final : public OptimizerUpdateOpInterface {
 public:
  MomentumUpdateOpImpl(float learning_rate, float momentum, bool nesterov)
  : OptimizerUpdateOpInterface(quote(MomemtumUpdateOp), learning_rate),
    _momentum(momentum),
    _nesterov(nesterov) {
    HT_VALUE_ERROR_IF(momentum < 0 || momentum > 1)
      << "Invalid momemtum: " << momentum;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MomentumUpdateOpImpl&>(rhs);
      return momentum() == rhs_.momentum() && nesterov() == rhs_.nesterov();
    }
    return false;
  }

  float momentum() const {
    return _momentum;
  }

  bool nesterov() const {
    return _nesterov;
  }

 protected:
  float _momentum;
  bool _nesterov;
};

class AdamOpImpl : public OptimizerUpdateOpInterface {
 public:
  AdamOpImpl(float learning_rate, const std::vector<bool>& multi_zero = {false},
             float beta1 = 0.9, float beta2 = 0.999, 
             float eps = 1e-8, float weight_decay = 0)
  : OptimizerUpdateOpInterface(quote(AdamOp), learning_rate),
    _multi_zero(multi_zero),
    _beta1(beta1),
    _beta2(beta2),
    _eps(eps),
    _weight_decay(weight_decay) {
    HT_VALUE_ERROR_IF(beta1 < 0 || beta1 > 1)
      << "Invalid beta1: " << beta1;
    HT_VALUE_ERROR_IF(beta2 < 0 || beta1 > 2)
      << "Invalid beta2: " << beta2;
  }

  uint64_t op_indicator() const noexcept override {
    return OPTIMIZER_UPDATE_OP | ADAM_OP;
  }  

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override; 

  void DoSpecialMergeStrategy(Operator& op, Operator& another_op) override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AdamOpImpl&>(rhs);
      return beta1() == rhs_.beta1() && 
             beta2() == rhs_.beta2() && 
             eps() == rhs_.eps() && 
             weight_decay() == rhs_.weight_decay();
    }
    return false;
  }

  const std::vector<bool>& multi_zero() const {
    return _multi_zero;
  }

  float beta1() const {
    return _beta1;
  }

  float beta2() const {
    return _beta2;
  }

  float eps() const {
    return _eps;
  }

  float weight_decay() const {
    return _weight_decay;
  }

  const NDArray& adam_step() const {
    return _adam_step;
  }

 protected:
  std::vector<bool> _multi_zero;
  float _beta1;
  float _beta2;
  float _eps;
  float _weight_decay;
  NDArray _adam_step;
};

Tensor MakeSGDUpdateOp(Tensor param, Tensor grad, float learning_rate,
                       OpMeta op_meta = OpMeta());

Tensor MakeSGDUpdateWithGradScalerOp(Tensor param, Tensor grad, Tensor infinite_count, float learning_rate,
                                     OpMeta op_meta = OpMeta());

Tensor MakeMomentumUpdateOp(Tensor param, Tensor grad, Tensor velocity,
                            float learning_rate, float momentum, bool nesterov,
                            OpMeta op_meta = OpMeta());

Tensor MakeAdamOp(Tensor param, Tensor grad, 
                  Tensor mean, Tensor variance,
                  float learning_rate, Tensor step, float beta1 = 0.9,
                  float beta2 = 0.999, float eps = 1e-8,
                  float weight_decay = 0,
                  OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis