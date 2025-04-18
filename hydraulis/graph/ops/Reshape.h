#pragma once

#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/utils/tensor_utils.h"
#include "hydraulis/core/symbol.h"

namespace hydraulis {
namespace graph {

class ArrayReshapeOpImpl;
class ArrayReshapeOp;
class ArrayReshapeGradientOpImpl;
class ArrayReshapeGradientOp;

// reshape算子虽然依赖shape，但不需要symbolic方法
// 原因是其可以通过-1去推断维度的大小
// 2023.12.9修正，依然是需要symbolic shape的
// 因为可能会有seq_len以及batch_size同时发生变化的情况
class ArrayReshapeOpImpl final : public OpInterface {
 private:
  friend class ArrayReshapeOp;
  struct constructor_access_key {};

 public:
  // symbolic shape constructor
  ArrayReshapeOpImpl(const SyShape& output_shape, int64_t padding_axis = -1)
  : OpInterface(quote(ArrayReshapeOp)),
     _global_output_shape(output_shape), 
     _padding_axis(padding_axis), 
     _symbolic(true) { // default is global output shape, if distributed, then turn into local output shape
  }
  // fixed shape constructor
  ArrayReshapeOpImpl(const HTShape& output_shape, int64_t padding_axis = -1)
  : OpInterface(quote(ArrayReshapeOp)),
     _global_output_shape(output_shape.begin(), output_shape.end()), 
     _padding_axis(padding_axis), 
     _symbolic(false) { // default is global output shape, if distributed, then turn into local output shape
  }

  HTShape get_output_shape() const {
    return get_HTShape_from_SyShape(_global_output_shape);
  }

  const SyShape& get_symbolic_output_shape() const {
    return _global_output_shape;
  }

  // deprecated: only used in gpt inference, before symbolic shape is realized
  int64_t get_padding_axis() const {
    return _padding_axis;
  }

  bool symbolic() const {
    return _symbolic;
  }

  HTShape get_output_shape(const HTShape& input_shape) const {
    int numel = 1;
    for (auto d : input_shape) {
      numel *= d;
    }
    HTShape output_shape = get_output_shape();
    int index = -1;
    int numel_output = 1;
    for (int i = 0; i < output_shape.size(); i++) {
      if (output_shape[i] == -1) {
        HT_ASSERT(index == -1)
          << "not allow multi -1 appears in shape!";
        index = i; 
      } else {
        numel_output *= output_shape[i];
      }
    }
    if (index != -1) {
      output_shape[index] = numel / numel_output;
    }
    else {
      HT_ASSERT(numel_output == numel) << "ArrayReshapeOpImpl: the numel of input and output should be equal, "
        << "but input shape is " << input_shape << " and output shape is " << output_shape;
    }
    return output_shape;
  }

  // input_shape & output_shape should be global shape
  static DistributedStates get_output_ds(const HTShape& input_shape,
                                         const DistributedStates& ds_input, 
                                         const HTShape& output_shape) {
    int dim_i = input_shape.size() - 1;
    int dim_o = output_shape.size() - 1;
    std::unordered_map<int, int> dim_map;
    while (dim_i >= 0 && dim_o >= 0) {
      int last_dim_i = dim_i;
      int last_dim_o = dim_o;
      int i_size = input_shape[dim_i];
      int o_size = output_shape[dim_o];
      while (i_size != o_size) {
        if (i_size < o_size) {
          i_size *= input_shape[--dim_i];
        } else {
          o_size *= output_shape[--dim_o];
        }
      }
      while (dim_i >= 1 && input_shape[dim_i - 1] == 1) {
        dim_i--;
      }
      while (dim_o >= 1 && output_shape[dim_o - 1] == 1) {
        dim_o--;
      }
      // shape[dim_i~last_dim_i] == shape[dim_o~last_dim_o]
      // case 0: 1 to 1
      if (dim_i == last_dim_i && dim_o == last_dim_o) {
        if (ds_input.get_dim(dim_i) > 0) {
          dim_map[dim_i] = dim_o;
        }
      }
      // case 1: 1 to many
      else if (dim_i == last_dim_i && dim_o != last_dim_o) {
        if (ds_input.get_dim(dim_i) > 0) {
          dim_map[dim_i] = dim_o;
        }
      }
      // case 2: many to 1
      else if (dim_i != last_dim_i && dim_o == last_dim_o) {
        for (int d = dim_i + 1; d <= last_dim_i; d++) {
          HT_ASSERT(ds_input.get_dim(d) == 1)
            << "ReShapeOp: dimension " << d << " shouldn't be splited"
            << ", ds input is " << ds_input.ds_info() << ", input shape is " << input_shape
            << ", and output shape is " << output_shape;
        }
        if (ds_input.get_dim(dim_i) > 0) {
          dim_map[dim_i] = dim_o;
        }
      }
      // case 3: many to many
      else {
        for (int d = dim_i; d <= last_dim_i; d++) {
          HT_ASSERT(ds_input.get_dim(d) == 1 || d == dim_o)
            << "ReshapeOp: dimension " << d << " shouldn't be splited!";
        }
      }
      dim_i--;
      dim_o--;
    }
    dim_map[-1] = -1;
    std::unordered_map<int32_t, int32_t> states;
    std::vector<int32_t> order;
    for (int d : ds_input.get_order()) {
      order.push_back(dim_map[d]);
      states[dim_map[d]] = ds_input.get_dim(d);
    }
    DistributedStates ds_output({ds_input.get_device_num(), states, order});
    return ds_output;
  }

  HTShape get_local_output_shape(const HTShape& global_input_shape,
                                 const DistributedStates& input_ds) const {
    HTShape global_output_shape = get_output_shape(global_input_shape);
    HT_LOG_TRACE << "global_input_shape = " << global_input_shape
      << " input ds states = " << input_ds.get_states()
      << " global output shape = " << global_output_shape;
    DistributedStates output_ds = get_output_ds(global_input_shape, input_ds, global_output_shape);
    HTShape local_shape(global_output_shape.size());
    for (size_t d = 0; d < global_output_shape.size(); d++) {
      local_shape[d] = global_output_shape[d] / output_ds.get_dim(d);
    }
    return local_shape;
  }  

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape output_shape = get_output_shape();
    if (inputs[0]->has_distributed_states()) {
      output_shape = get_local_output_shape(inputs[0]->global_shape(), 
                                            inputs[0]->get_distributed_states());                                       
    }
    NDArrayMeta output_meta;
    if (inputs[0]->is_contiguous()) {
      output_meta = inputs[0]->meta();
      output_meta.view(output_shape);
    } else {
      output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                 .set_shape(output_shape)
                                 .set_device(inputs[0]->device());
    }
    return {output_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  // deprecated: only used in gpt inference, before symbolic shape is realized
  HTShapeList DoInferDynamicShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  SyShape _global_output_shape;
  bool _symbolic;

  int64_t _padding_axis; // deprecated

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ArrayReshapeOpImpl&>(rhs);
      return get_output_shape() == rhs_.get_output_shape();
    }
    return false;
  }
};

// fixed shape
Tensor MakeArrayReshapeOp(Tensor input, const HTShape& output_shape,
                          OpMeta op_meta = OpMeta());

// symbolic shape
Tensor MakeArrayReshapeOp(Tensor input, const SyShape& output_shape,
                          OpMeta op_meta = OpMeta());

// deprecated: only used in gpt inference, before symbolic shape is realized
Tensor MakeArrayReshapeOp(Tensor input, const HTShape& output_shape,
                          int64_t padding_axis, OpMeta op_meta = OpMeta());

class ArrayReshapeGradientOpImpl final : public OpInterface {

 public:
  // symbolic shape constructor
  ArrayReshapeGradientOpImpl(const SyShape& in_shape)
  : OpInterface(quote(ArrayReshapeGradientOp)), 
    _input_shape(in_shape),
    _symbolic(true) {
  }
  // fixed shape constructor
  ArrayReshapeGradientOpImpl(const HTShape& in_shape)
  : OpInterface(quote(ArrayReshapeGradientOp)), 
    _input_shape(in_shape.begin(), in_shape.end()),
    _symbolic(false) {
  }

  HTShape get_input_shape() const {
    return get_HTShape_from_SyShape(_input_shape);
  }

  const SyShape& get_symbolic_input_shape() const {
    return _input_shape;
  }

  bool symbolic() const {
    return _symbolic;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[1]->dtype())
                                           .set_shape(get_input_shape())
                                           .set_stride(inputs[1]->stride())
                                           .set_device(inputs[1]->device());
    return {output_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  SyShape _input_shape;
  bool _symbolic;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ArrayReshapeGradientOpImpl&>(rhs);
      return get_input_shape() == rhs_.get_input_shape();
    }
  }

};

// fixed shape
Tensor MakeArrayReshapeGradientOp(Tensor grad_output, Tensor ori_input, const HTShape& in_shape,
                                  OpMeta op_meta = OpMeta());

// symbolic shape
Tensor MakeArrayReshapeGradientOp(Tensor grad_output, Tensor ori_input, const SyShape& in_shape,
                                  OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hydraulis
