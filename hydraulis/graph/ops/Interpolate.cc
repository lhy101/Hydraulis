#include "hydraulis/graph/ops/Interpolate.h"
#include "hydraulis/graph/ops/Reshape.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

void InterpolateOpImpl::DoCompute(Operator& op,
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hydraulis::impl::Interpolate, inputs.at(0),
                                  outputs.at(0), align_corners(), op->instantiation_ctx().stream());
}

TensorList InterpolateOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeInterpolateGradientOp(grad_outputs.at(0), op->input(0), align_corners(), scale_factor(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
InterpolateOpImpl::DoInferShape(Operator& op, 
                                const HTShapeList& input_shapes, 
                                RuntimeContext& ctx) const {
  HTShape output = input_shapes[0];
  if (out_shape().size() == 2) {
    output[2] = out_shape()[0];
    output[3] = out_shape()[1];
  }
  else {
    HT_ASSERT(scale_factor() > 0);
    output[2] = output[2] * scale_factor();
    output[3] = output[3] * scale_factor();
  }
  return {output};
}

void InterpolateOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                       const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) << "InterpolateOpImpl: input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(2))
    << "InstanceNormOp only support split dimensions N&C in [N, C, H, W]!";  
  outputs.at(0)->set_distributed_states(ds_input);
}

void InterpolateGradientOpImpl::DoCompute(Operator& op,
                                          const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hydraulis::impl::InterpolateGradient,
    inputs.at(0), outputs.at(0), align_corners(), op->instantiation_ctx().stream());
}


HTShapeList
InterpolateGradientOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  return {input_shapes.at(1)};
}

void InterpolateGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                               const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(1)->get_distributed_states());
}

Tensor MakeInterpolateOp(Tensor input, const HTShape& outshape,
                         bool align_corners, double scale_factor,
                         OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
          std::make_shared<InterpolateOpImpl>(outshape, align_corners, scale_factor),
          std::move(inputs),
          std::move(op_meta))->output(0);
}

Tensor MakeInterpolateGradientOp(Tensor grad_output, Tensor input,
                                 bool align_corners, double scale_factor,
                                 OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<InterpolateGradientOpImpl>(align_corners, scale_factor),
          {std::move(grad_output), std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
