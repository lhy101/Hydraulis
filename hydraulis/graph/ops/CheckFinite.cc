#include "hydraulis/graph/ops/CheckFinite.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

void CheckFiniteOpImpl::DoCompute(Operator& op, 
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hydraulis::impl::CheckFinite,
                               inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
}

TensorList CheckFiniteOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {Tensor()};
}

HTShapeList CheckFiniteOpImpl::DoInferShape(Operator& op, 
                                            const HTShapeList& input_shapes, 
                                            RuntimeContext& ctx) const {
  return {{1}};
}

Tensor MakeCheckFiniteOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<CheckFiniteOpImpl>(),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

void CheckNumericOpImpl::DoCompute(Operator& op, 
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hydraulis::impl::CheckNumeric,
                               inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
}

TensorList CheckNumericOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {Tensor()};
}

HTShapeList CheckNumericOpImpl::DoInferShape(Operator& op, 
                                            const HTShapeList& input_shapes, 
                                            RuntimeContext& ctx) const {
  return {{3}};
}

Tensor MakeCheckNumericOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<CheckNumericOpImpl>(),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
