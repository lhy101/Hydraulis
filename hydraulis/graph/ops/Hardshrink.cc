#include "hydraulis/graph/ops/Hardshrink.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

void HardshrinkOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::hardshrink(inputs.at(0), lambda(), 
                      op->instantiation_ctx().stream_index, 
                      outputs.at(0));
}

TensorList HardshrinkOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeHardshrinkGradientOp(op->output(0), grad_outputs.at(0),
                                 lambda(), op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void HardshrinkGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hydraulis::impl::HardshrinkGradient, inputs.at(0),
                               inputs.at(1), lambda(),
                               outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeHardshrinkOp(Tensor input, double lambda, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<HardshrinkOpImpl>(lambda),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeHardshrinkGradientOp(Tensor output, Tensor grad_output,
                         double lambda, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<HardshrinkGradientOpImpl>(lambda),
        {std::move(output), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
