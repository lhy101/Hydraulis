#include "hydraulis/graph/ops/Gather.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

void GatherOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs,
                             NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  NDArray::gather(inputs.at(0), inputs.at(1), get_dim(), 
                  op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList GatherOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeGatherGradientOp(grad_outputs.at(0), get_dim(), op->input(1), op->input(0),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList
GatherOpImpl::DoInferShape(Operator& op,
                           const HTShapeList& input_shapes,
                           RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes[0].size() == input_shapes[1].size());
  int64_t len = input_shapes[0].size();
  for (int64_t i = 0; i < len; ++i) {
    if (i != get_dim())
      HT_ASSERT(input_shapes[0][i] == input_shapes[1][i]);
  }
  return {input_shapes.at(1)};
}

void GatherOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_id = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_id.is_valid()) 
    << "GatherOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_id.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_pure_duplicate() && ds_id.check_pure_duplicate())
    << "Input tensor cannot be splited in any dimension!";
  outputs.at(0)->set_distributed_states(ds_id);    
}

void GatherGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hydraulis::impl::GatherGradient,
    inputs.at(0), inputs.at(1), outputs.at(0), get_dim(), op->instantiation_ctx().stream());
}

HTShapeList
GatherGradientOpImpl::DoInferShape(Operator& op,
                                   const HTShapeList& input_shapes,
                                   RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void GatherGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());
}

Tensor MakeGatherOp(Tensor input, int64_t dim, Tensor id, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<GatherOpImpl>(dim),
          {std::move(input), std::move(id)},
          std::move(op_meta))->output(0);
}

Tensor MakeGatherGradientOp(Tensor grad_output, int64_t dim, Tensor id, Tensor input,
                            OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<GatherGradientOpImpl>(dim),
          {std::move(grad_output), std::move(id), std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
