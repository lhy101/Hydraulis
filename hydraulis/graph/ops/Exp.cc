#include "hydraulis/graph/ops/Exp.h"
#include "hydraulis/graph/ops/Arithmetics.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

NDArrayList ExpOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs,
                                 RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  NDArray::exp(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
  return outputs;
}

void ExpOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                          NDArrayList& outputs, RuntimeContext& ctx) const {
  NDArray::exp(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ExpOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeMulElewiseOp(op->output(0), grad_outputs.at(0),
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

Tensor MakeExpOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<ExpOpImpl>(false),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeExpInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<ExpOpImpl>(true),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
