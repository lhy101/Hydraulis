#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/graph/ops/Unary.h"

namespace hydraulis {
namespace graph {

class SiluOpImpl;
class SiluOp;
class SiluGradientOpImpl;
class SiluGradientOp;

class SiluOpImpl final : public UnaryOpImpl {
 private:
  friend class SiluOp;
  struct constructor_access_key {};

 public:
  SiluOpImpl()
  : UnaryOpImpl(quote(SiluOp)){
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

Tensor MakeSiluOp(Tensor input, OpMeta op_meta = OpMeta());

class SiluGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  SiluGradientOpImpl()
  : UnaryGradientOpImpl(quote(SiluGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeSiluGradientOp(Tensor input, Tensor grad_output,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
