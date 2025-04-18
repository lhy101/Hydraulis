#include "hydraulis/graph/ops/Log.h"
#include "hydraulis/graph/ops/Arithmetics.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

NDArrayList LogOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs,
                                 RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  NDArray::log(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
  return outputs;
}

void LogOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                          NDArrayList& outputs, RuntimeContext& ctx) const {
  NDArray::log(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList LogOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  HT_ASSERT(!inplace())
    << "This op doesn't support gradient for inplace.";
  return {op->requires_grad(0) ? MakeDivElewiseOp(grad_outputs.at(0), op->input(0),
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

Tensor MakeLogOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<LogOpImpl>(false),
        std::move(inputs),
        std::move(op_meta))->output(0);
}
Tensor MakeLogInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<LogOpImpl>(true),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
