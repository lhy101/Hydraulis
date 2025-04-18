#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Unary.h"

namespace hydraulis {
namespace graph {

class DropoutOpImpl;
class DropoutOp;
class DropoutGradientOpImpl;
class DropoutGradientOp;

class DropoutOpImpl final : public UnaryOpImpl {
 public:
  DropoutOpImpl(double keep_prob, bool inplace = false)
  : UnaryOpImpl(quote(DropoutOp), inplace),
    _keep_prob(keep_prob) {}

  inline double keep_prob() const {
    return _keep_prob;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    NDArrayMeta mask_meta = inputs[0]->meta();
    mask_meta.set_dtype(DataType::BOOL);
    return {output_meta, mask_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  double _keep_prob;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DropoutOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob() &&
              inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeDropoutOp(Tensor input, double keep_prob, OpMeta op_meta = OpMeta());

Tensor MakeDropoutInplaceOp(Tensor input, double keep_prob, OpMeta op_meta = OpMeta());

class DropoutGradientOpImpl final : public UnaryGradientOpImpl {
 public:
  DropoutGradientOpImpl(double keep_prob,
                        bool fw_inplace,
                        OpMeta op_meta = OpMeta())
  : UnaryGradientOpImpl(quote(DropoutGradientOp)),
    _keep_prob(keep_prob),
    _fw_inplace(fw_inplace) {
  }

  inline double keep_prob() const {
    return _keep_prob;
  }

  inline bool fw_inplace() const {
    return _fw_inplace;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  double _keep_prob;
  bool _fw_inplace;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DropoutGradientOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob() &&
              fw_inplace() == rhs_.fw_inplace());
    }
    return false;
  }
};

Tensor MakeDropoutGradientOp(Tensor grad_output, Tensor mask,
                             double keep_prob, bool fw_inplace,
                             OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
