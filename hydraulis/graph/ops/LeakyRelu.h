#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Unary.h"

namespace hydraulis {
namespace graph {

class LeakyReluOpImpl;
class LeakyReluOp;
class LeakyReluGradientOpImpl;
class LeakyReluGradientOp;

class LeakyReluOpImpl final : public UnaryOpImpl {
 private:
  friend class LeakyReluOp;
  struct constructor_access_key {};

 public:
  LeakyReluOpImpl(double alpha, bool inplace)
  : UnaryOpImpl(quote(LeakyReluOp), inplace), _alpha(alpha) {
  }

  inline double get_alpha() const {
    return _alpha;
  }

protected:
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  double _alpha;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LeakyReluOpImpl&>(rhs);
      return get_alpha() == rhs_.get_alpha(); 
    }
    return false;
  }
};

Tensor MakeLeakyReluOp(Tensor input, double alpha, OpMeta op_meta = OpMeta());

Tensor MakeLeakyReluInplaceOp(Tensor input, double alpha, OpMeta op_meta = OpMeta());

class LeakyReluGradientOpImpl final : public UnaryGradientOpImpl {
 private:
  friend class LeakyReluGradientOp;
  struct constructor_access_key {};

 public:
  LeakyReluGradientOpImpl(double alpha, bool is_result,
                          OpMeta op_meta = OpMeta())
  : UnaryGradientOpImpl(quote(LeakyReluGradientOp)),
    _alpha(alpha), _is_result(is_result) {
  }

  inline double get_alpha() const {
    return _alpha;
  }

  inline bool is_result() const {
    return _is_result;
  }

protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  double _alpha;
  bool _is_result;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LeakyReluGradientOpImpl&>(rhs);
      return (get_alpha() == rhs_.get_alpha() && is_result() == rhs_.is_result()); 
    }
    return false;
  }
};

Tensor MakeLeakyReluGradientOp(Tensor input, Tensor grad_output, double alpha,
                               bool is_result = false, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
 