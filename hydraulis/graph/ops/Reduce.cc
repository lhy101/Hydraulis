#include "hydraulis/graph/ops/Arithmetics.h"
#include "hydraulis/graph/ops/Broadcast.h"
#include "hydraulis/graph/ops/Reduce.h"
#include "hydraulis/graph/headers.h"
#include "hydraulis/graph/ops/kernel_links.h"

namespace hydraulis {
namespace graph {

void ReduceOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  NDArray::reduce(inputs.at(0), reduction(), get_axes(), false,
                  op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ReduceOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeReduceGradientOp(grad_outputs.at(0), op->output(0), op->input(0), HTShape(), reduction(),
                                get_axes(), get_keepdims(), op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList ReduceOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  HTShapeList outputlist = {};
  HTShape input_shape = input_shapes.at(0);
  int ndim = input_shape.size();
  int64_t mean_multiplier = 1;
  HTShape axes = get_axes();
  int len = axes.size();
  HTKeepDims keepdims = get_keepdims();
  HTShape add_axes = {};
  for (int i = 0; i < len; ++i) {
    if (axes[i] < 0) {
      axes[i] += ndim;
    }
    HT_ASSERT(axes[i] >= 0 && axes[i] < ndim);
    mean_multiplier *= input_shape[axes[i]];
    if (keepdims[i] == true)
      input_shape[axes[i]] = 1;
    else {
      input_shape[axes[i]] = 0;
      add_axes.emplace_back(axes[i]);
    }
  }
  HTShape output_shape(0);
  for (int i = 0; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  if (output_shape.size() == 0)
    output_shape.emplace_back(1);
  outputlist = {output_shape};
  return outputlist;
}

DistributedStates ReduceOpImpl::StatesForDistributedReduce(
  const Tensor& input, 
  const HTShape& axes, 
  const HTKeepDims& keepdims) {
  const DistributedStates& ds_input = input->get_distributed_states();
  int32_t partial = ds_input.get_dim(-2);
  // device_num
  int32_t device_num = ds_input.get_device_num();
  // states
  std::unordered_map<int32_t, int32_t> states = ds_input.get_states();
  for (auto d : axes) {
    int32_t state_d = ds_input.get_dim(d); 
    if (state_d > 1) {
      partial *= state_d;
      states.erase(d);
    }
  }
  states[-2] = partial;
  std::vector<int32_t> sorted_keys;
  for (auto& pair : states) {
    if (pair.first >= 0) {
      sorted_keys.push_back(pair.first);
    }
  }
  std::sort(sorted_keys.begin(), sorted_keys.end());
  for (auto d : sorted_keys) {
    int32_t reduce_dimensions = 0;
    for (int i = 0; i < axes.size(); i++) {
      if (axes[i] < d && !keepdims[i]) {
        reduce_dimensions++;
      }
    }
    int32_t new_d = d - reduce_dimensions;
    if (new_d != d) {
      states[new_d] = states[d];
      states.erase(d);
    }
  }
  // order
  std::vector<int32_t> order = ds_input.get_order();
  int32_t dup_occur = 0;
  bool prev_dup = false;
  std::vector<int64_t> partial_candidate = axes;
  partial_candidate.push_back(-2);
  for (int i = order.size() - 1; i >= 0; i--) {
    if (std::find(partial_candidate.begin(), partial_candidate.end(), order[i]) != partial_candidate.end()) {
      if (!prev_dup) {
        dup_occur++;
      }
      prev_dup = true;
      if (order[i] != -2) {
        if (std::find(order.begin(), order.end(), -2) == order.end()) {
          order[i] = -2;
        } else {
          order.erase(order.begin() + i);
        }
      }
    } else {
      prev_dup = false;
    }
  }
  HT_ASSERT(dup_occur <= 1) << "Duplicate dimension and reduce dimensions must be consecutive!";
  for (int i = 0;i < order.size(); i++) {
    int32_t reduce_dimensions = 0;
    for (int j = 0; j < axes.size(); j++) {
      if (axes[j] < order[i] && !keepdims[j]) {
        reduce_dimensions++;
      }
    }
    order[i] -= reduce_dimensions;
  }
  return DistributedStates({device_num, states, order});
}

void ReduceOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ReduceOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Tensor input shouldn't be partial!";
  HTShape axes = get_axes();
  HTKeepDims keepdims = get_keepdims();  
  DistributedStates ds_output = StatesForDistributedReduce(inputs.at(0), axes, keepdims);
  outputs.at(0)->set_distributed_states(ds_output);
}

void ReduceOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                     TensorList& outputs, const OpMeta& op_meta) const {
  HT_ASSERT(inputs_hetero_dim.at(0) < 0)
    << "Currently not support complex hetero dim deducing";
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void ReduceGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     NDArrayList& outputs, RuntimeContext& ctx) const {
  if (reduction() == ReductionType::MEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(), hydraulis::impl::BroadcastShapeMul, inputs.at(0),
      get_const_value(), outputs.at(0), get_axes(), op->instantiation_ctx().stream());
  } else {
    auto axes = get_axes();
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                    hydraulis::impl::BroadcastShape, inputs.at(0),
                                    outputs.at(0), axes, op->instantiation_ctx().stream());
  }
}

HTShapeList ReduceGradientOpImpl::DoInferShape(Operator& op,
                                               const HTShapeList& input_shapes,
                                               RuntimeContext& ctx) const {
  return  {input_shapes.at(2)};
}

void ReduceGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());
}

void ReduceGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                             TensorList& outputs, const OpMeta& op_meta) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(2));
}

Tensor MakeReduceOp(Tensor input, ReductionType reduction, const HTAxes& axes,
                    const HTKeepDims& keepdims,
                    OpMeta op_meta) {
  HTAxes parsed_axes = axes;
  HTKeepDims parsed_keepdims = keepdims;
  if (parsed_axes.size() == 0) {
      parsed_axes.reserve(input->ndim());
      for (size_t i = 0; i < input->ndim(); ++i) {
        parsed_axes.push_back(i);
      }
    }
  parsed_axes = NDArrayMeta::ParseAxes(parsed_axes, input->ndim());
  HT_ASSERT(parsed_keepdims.size() == parsed_axes.size() || parsed_keepdims.size() == 1);
  if (parsed_keepdims.size() == 1) {
    int len = parsed_axes.size();
    bool keepdim = parsed_keepdims[0];
    for (int i = 1; i < len; ++i) {
      parsed_keepdims.emplace_back(keepdim);
    }
  }     
  return Graph::MakeOp(
          std::make_shared<ReduceOpImpl>(reduction, parsed_axes, parsed_keepdims),
          {std::move(input)},
          std::move(op_meta))->output(0);              
}

Tensor MakeReduceOp(Tensor input, const std::string& mode, const HTAxes& axes,
                    const HTKeepDims& keepdims,
                    OpMeta op_meta) {
  return MakeReduceOp(std::move(input), Str2ReductionType(mode), axes, keepdims, op_meta);
}

Tensor MakeReduceMeanOp(Tensor input, const HTAxes& axes,
                        const HTKeepDims& keepdims,
                        OpMeta op_meta) {
  return MakeReduceOp(std::move(input), ReductionType::MEAN, axes, keepdims, op_meta);     
}

Tensor MakeReduceSumOp(Tensor input, const HTAxes& axes,
                       const HTKeepDims& keepdims,
                       OpMeta op_meta) {
  return MakeReduceOp(std::move(input), ReductionType::SUM, axes, keepdims, op_meta);          
}

Tensor MakeReduceMaxOp(Tensor input, const HTAxes& axes,
                       const HTKeepDims& keepdims,
                       OpMeta op_meta) {
  return MakeReduceOp(std::move(input), ReductionType::MAX, axes, keepdims, op_meta);          
}

Tensor MakeReduceMinOp(Tensor input, const HTAxes& axes,
                       const HTKeepDims& keepdims,
                       OpMeta op_meta) {
  return MakeReduceOp(std::move(input), ReductionType::MIN, axes, keepdims, op_meta);          
}

Tensor MakeReduceProdOp(Tensor input, const HTAxes& axes,
                        const HTKeepDims& keepdims,
                        OpMeta op_meta) {
  return MakeReduceOp(std::move(input), ReductionType::PROD, axes, keepdims, op_meta);
}

Tensor MakeReduceGradientOp(Tensor input, Tensor ori_output, Tensor ori_input, const HTShape& shape,
                            ReductionType reduction, const HTAxes add_axes, const HTKeepDims& keepdims,
                            OpMeta op_meta){
  double const_value = 0;
  if (reduction == ReductionType::MEAN) {
    HTShape input_shape = ori_input->shape();
    int ndim = input_shape.size();
    int64_t mean_multiplier = 1;
    int len = add_axes.size();
    for (int i = 0; i < len; ++i) {
      HT_ASSERT(add_axes[i] >= 0 && add_axes[i] < ndim);
      mean_multiplier *= input_shape[add_axes[i]];
    }
    const_value = 1.0 / mean_multiplier;
  }
  return Graph::MakeOp(
          std::make_shared<ReduceGradientOpImpl>(shape, reduction, add_axes, keepdims, const_value),
          {std::move(input), std::move(ori_output), std::move(ori_input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hydraulis
