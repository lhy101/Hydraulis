#include "hydraulis/graph/ops/Repeat.h"
#include "hydraulis/graph/ops/Reshape.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

void RepeatOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
  //                              hydraulis::impl::Repeat, inputs.at(0),
  //                              outputs.at(0), op->instantiation_ctx().stream());
  NDArray::repeat(inputs.at(0), repeats(),
                  op->instantiation_ctx().stream_index,
                  outputs.at(0));
}

TensorList RepeatOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeRepeatGradientOp(grad_outputs.at(0), op->input(0),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
RepeatOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  HTShape output_shape = repeats();
  HT_ASSERT(output_shape.size() >= input_shapes[0].size());
  for (size_t i = 0; i < input_shapes[0].size(); ++i) {
    output_shape[i + output_shape.size() - input_shapes[0].size()] *= input_shapes[0][i]; 
  }
  return {output_shape};
}

void RepeatOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta) const {
  const DistributedStates& ds = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds.is_valid()) 
    << "RepeatOpImpl: distributed states for input tensor must be valid!";
  HT_ASSERT(ds.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";

  const auto& repeat = repeats();
  int32_t extra_dim_size = repeat.size() - inputs.at(0)->ndim();
  for (int dim = extra_dim_size; dim < repeat.size(); dim++) {
    if (repeat[dim] > 1) {
      HT_ASSERT(ds.get_dim(dim - extra_dim_size) == 1)
        << "The repeat dim " << dim - extra_dim_size << " shouldn't be split!";
    }
  }

  if (extra_dim_size == 0) {
    outputs.at(0)->set_distributed_states(ds);
  } else {
    std::unordered_map<int32_t, int32_t> new_states;
    std::vector<int32_t> new_order;
    for (auto& state : ds.get_states()) {
      new_states[state.first + extra_dim_size] = state.second;
    }
    for (auto& o : ds.get_order()) {
      new_order.push_back(o + extra_dim_size);      
    }
    outputs.at(0)->set_distributed_states({ds.get_device_num(), new_states, new_order});
  }
}

void RepeatGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs, NDArrayList& outputs,
                                     RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hydraulis::impl::RepeatGradient,
    inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList
RepeatGradientOpImpl::DoInferShape(Operator& op, 
                                   const HTShapeList& input_shapes, 
                                   RuntimeContext& ctx) const {
  return {input_shapes[1]};
}

void RepeatGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(1)->get_distributed_states());
}

Tensor MakeRepeatOp(Tensor input, HTShape repeats, OpMeta op_meta) {
    return Graph::MakeOp(
        std::make_shared<RepeatOpImpl>(repeats),
        {std::move(input)},
        std::move(op_meta))->output(0);
}

Tensor MakeRepeatGradientOp(Tensor grad_output, Tensor input,
                            OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<RepeatGradientOpImpl>(),
        {std::move(grad_output), std::move(input)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
