import hydraulis
import numpy as np
from queue import Queue
from .module import Module
import numbers

__all__ = [
    'HtMultiColumnParallelLinear', 
    'HtMultiRowParallelLinear', 
    'HtMultiParallelEmbedding',
    'HtMultiVocabParallelEmbedding',
    'HtMultiParallelLayerNorm',
    'HtMultiParallelRMSNorm',
]

def parallel_data_provider(global_data, ds_union, device_group_index, device_index):
    ds = ds_union.get_local(device_group_index)
    order, states = ds.order, ds.states
    local_map = hydraulis.map_to_local_data(ds, device_index)
    local_data = global_data.copy()
    for dim in order:
        if dim < 0:
            continue
        splits = states[dim]
        split_index = local_map[dim]
        start = int(split_index * (global_data.shape[dim] / splits))
        stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
        local_data = local_data.take(range(start, stop), axis=dim)
    return local_data
  
def parallel_multi_data_provider(global_data, ds_unions, device_group_unions):
    multi_local_data = []
    for i in range(len(ds_unions)):
        ds_union = ds_unions[i]
        device_group_union = device_group_unions[i]
        device_group_index, device_index = get_local_index(device_group_union)
        ds = ds_union.get_local(device_group_index)
        order, states = ds.order, ds.states
        local_map = hydraulis.map_to_local_data(ds, device_index)
        local_data = global_data.copy()
        for dim in order:
            if dim < 0:
                continue
            splits = states[dim]
            split_index = local_map[dim]
            start = int(split_index * (global_data.shape[dim] / splits))
            stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
            local_data = local_data.take(range(start, stop), axis=dim)
        multi_local_data.append(local_data)
    return multi_local_data

def get_local_index(device_group_union):
    local_device = hydraulis.local_device()
    device_group_index = -1
    device_index = -1
    for device_group_index in range(len(device_group_union)):
        device_group = device_group_union[device_group_index]
        if device_group.contains(local_device):
            device_index = device_group.get_index(local_device)
            break
    if device_group_index == len(device_group_union): # for pipeline parallel other stages
        device_group_index = -1
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_group_index, device_index

def get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    multi_ds_parallel_config = []
    for ds_parallel_config in ds_parallel_configs:
        config_queue = Queue()
        config_queue.put(ds_parallel_config)
        while (not config_queue.empty()):
            config = config_queue.get()
            if module_name in config:
                multi_ds_parallel_config.append(config[module_name])
                break
            else:
                for value in config.values():
                    if type(value) == dict:
                        if "range" in value and (_range < value["range"][0] or _range > value["range"][-1]):
                            continue
                        config_queue.put(value)
    assert len(multi_ds_parallel_config) == len(ds_parallel_configs), \
        f'ds_parallel_configs parse error, cannot find {module_name}' if _range == -1 else \
        f'ds_parallel_configs parse error, cannot find {module_name} with range id {_range}'   
    return multi_ds_parallel_config

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    ds_list = []
    dg_list = []
    if config['type'] == 'placeholder':
        hetero_dim = 0
    elif config['type'] == 'variable':
        hetero_dim = -1
    else:
        raise RuntimeError(f"unsupported type {config['type']}!")   
    hetero_sum = len(config['device_group_union'])
    if hetero_sum == 1:
        hetero_dim = -3
    for hetero_num in range(hetero_sum):
        dummy_num_devices = len(config['device_group_union'][hetero_num]) * hetero_sum
        zero = False
        split = {}
        for key, value in config['split'].items():
            assert len(value) == hetero_sum, "hetero sum mismatches"
            split[int(key)] = value[hetero_num]
        assert len(config['dup']) == hetero_sum, "hetero sum mismatches"
        states = {-1: config['dup'][hetero_num], **split}
        if config['type'] == 'placeholder':
            order = sorted(split.keys()) + [-1]
        elif config['type'] == 'variable':
            order = [-1] + sorted(split.keys())
            assert 'zero' in config, f"variable config must have zero!"
            zero = config['zero']
        else:
            raise RuntimeError(f"unsupported type {config['type']}!")
        ds = hydraulis.DistributedStates(dummy_num_devices, states, order, zero)
        all_devices = hydraulis.global_device_group()
        dg = hydraulis.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group_union'][hetero_num]])
        ds_list.append(ds)
        dg_list.append(dg)
    return hydraulis.DistributedStatesUnion(ds_list, hetero_dim), dg_list

class HtMultiParallelRMSNorm(Module):
    def __init__(self, normalized_shape, multi_ds_parallel_config, sp=False, dtype=hydraulis.float32, name='rmsnorm'):
        super(HtMultiParallelRMSNorm, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.sp = sp
        self.name = name
        self.ds_union_map = {'dup': [], 'split0': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            hetero_size = len(device_group_union)
            hetero_dim = ds_union_dup.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "ParallelLayerNorm only support hetero on dup"
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            ds_list_split0 = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: device_group_union[i].num_devices * hetero_size}, [0])
                for i in range(hetero_size)] # for sp data
            ds_union_split0 = hydraulis.DistributedStatesUnion(ds_list_split0, 0 if hetero_dim != -3 else -3)
            self.ds_union_map['dup'].append(ds_union_dup)
            self.ds_union_map['split0'].append(ds_union_split0)
            
        self.weight = hydraulis.parallel_parameter(eval(f'hydraulis.ones_initializer()'), 
                                              self.normalized_shape, self.ds_union_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')

    def forward(self, input_p):
        if self.sp:
            assert input_p.check_ds_hierarchy_equal(self.ds_union_map['split0']), \
                f'for sequence parallel, layernorm {self.name} need input fully sharded in dimension 0 for each element in the union, but found {input_p.ds_hierarchy}'
            # do sequence parallel layernorm: [bsz * seq_len // tp, hidden_size]
            # print(f'in sp, ln input shape = {input_p.shape}')
            output_rms = hydraulis.rms_norm(input_p, None, self.weight, None, is_rms_norm=True, \
                                       device_group_hierarchy=self.device_group_unions, name=self.name + '_sp')[0]
            # allgather will be auto done in later column parallel
        else:
            # [bsz * seq_len, hidden_size]
            # print(f'in no-sp, ln input shape = {input_p.shape}')
            output_rms = hydraulis.rms_norm(input_p, None, self.weight, None, is_rms_norm=True, \
                                       device_group_hierarchy=self.device_group_unions, name=self.name)[0]
        return output_rms   

class HtMultiParallelLayerNorm(Module):
    def __init__(self, normalized_shape, multi_ds_parallel_config, sp=False, eps=1e-5, dtype=hydraulis.float32, name='ln'):
        super(HtMultiParallelLayerNorm, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.sp = sp
        self.eps = eps
        self.name = name
        self.ds_union_map = {'dup': [], 'split0': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            hetero_size = len(device_group_union)
            hetero_dim = ds_union_dup.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "ParallelLayerNorm only support hetero on dup"
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            ds_list_split0 = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: device_group_union[i].num_devices * hetero_size}, [0])
                for i in range(hetero_size)] # for sp data
            ds_union_split0 = hydraulis.DistributedStatesUnion(ds_list_split0, 0 if hetero_dim != -3 else -3)
            self.ds_union_map['dup'].append(ds_union_dup)
            self.ds_union_map['split0'].append(ds_union_split0)
            
        self.weight = hydraulis.parallel_parameter(eval(f'hydraulis.ones_initializer()'), 
                                              self.normalized_shape, self.ds_union_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')
        self.bias = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                              self.normalized_shape, self.ds_union_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_bias')

    def forward(self, input_p):
        if self.sp:
            assert input_p.check_ds_hierarchy_equal(self.ds_union_map['split0']), \
                f'for sequence parallel, layernorm {self.name} need input fully sharded in dimension 0 for each element in the union, but found {input_p.ds_hierarchy}'
            # do sequence parallel layernorm: [bsz * seq_len // tp, hidden_size]
            # print(f'in sp, ln input shape = {input_p.shape}')
            output_ln = hydraulis.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, \
                                             self.eps, device_group_hierarchy=self.device_group_unions, name=self.name + '_sp')[0]
            # allgather will be auto done in later column parallel
        else:
            # [bsz * seq_len, hidden_size]
            # print(f'in no-sp, ln input shape = {input_p.shape}')
            output_ln = hydraulis.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, \
                                             self.eps, device_group_hierarchy=self.device_group_unions, name=self.name)[0]
        return output_ln   

class HtMultiParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                 init_method='xavier_normal_', dtype=hydraulis.float32, name='embedding'):
        super(HtMultiParallelEmbedding, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ds_union_map = {'dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            self.ds_union_map['dup'].append(ds_union_dup)
        
        # embedding_table should not be splited in any dimension!
        self.embedding_table = hydraulis.parallel_parameter(eval(f'hydraulis.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_union_map['dup'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_group_hierarchy=self.device_group_unions, name=f'{name}_table')
    
    def forward(self, input_p):
        return hydraulis.embedding_lookup(self.embedding_table, input_p, device_group_hierarchy=self.device_group_unions, name=self.name)
    
class HtMultiVocabParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                init_method='xavier_normal_', dtype=hydraulis.float32, name='vocab_embedding'):
        super(HtMultiVocabParallelEmbedding, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name

        self.ds_union_map = {'split0_dup': [], 'dup_split0': []}
        self.device_index = []
        self.device_group_unions = []
        self.vocab_start_index = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split0, device_group_union = config2ds(ds_parallel_config) # for embedding table
            self.device_group_unions.append(device_group_union)
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split0.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split0.get(i).get_dim(0) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split0.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "VocabParallelEmbedding only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union]) 
                , f'VocabParallelEmbedding get wrong ds_parallel_config: {ds_parallel_config}!')
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            ds_list_split0_dup = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: tp_union[i], 0: dcp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data
            ds_union_split0_dup = hydraulis.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
            self.ds_union_map['dup_split0'].append(ds_union_dup_split0)
            
            dup_group_idx = ds_union_dup_split0.get_local(device_group_index).get_dup_group_index(device_index)
            vocab_start_index = num_embeddings // tp_union[device_group_index] * dup_group_idx
            self.vocab_start_index.append(vocab_start_index)

        # embedding_table was splited in vocab dimension
        self.embedding_table = hydraulis.parallel_parameter(eval(f'hydraulis.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_union_map['dup_split0'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_group_hierarchy=self.device_group_unions, name=f'{name}_table')
    
    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            # sequence parallel: need use buffer for allgather to save activation
            tensor_split0_dup = hydraulis.comm(input_p, self.ds_union_map['split0_dup'])
            print(f"warning: vocab parallel embedding need extra communication for \
                    adapt input tensor ds hierarchy {input_p.ds_hierarchy} into {self.ds_union_map['split0_dup']}!")

        # walkaround: do offset inside embedding lookup op 
        # input_offset = tensor_split0_dup - self.vocab_start_index[0] # should do in embedding_lookup op for multi ds?
        lookup_split0_partial = hydraulis.embedding_lookup(self.embedding_table, tensor_split0_dup, self.vocab_start_index, 
                                                      device_group_hierarchy=self.device_group_unions, name=self.name+"_"+tensor_split0_dup.name)
        if lookup_split0_partial.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']): # pure dp
            output = lookup_split0_partial
        else:
            output = hydraulis.comm(lookup_split0_partial, self.ds_union_map['split0_dup'])
        return output

class HtMultiColumnParallelLinear(Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features, out_features, multi_ds_parallel_config,
                 bias=True, gather_output=True, init_method='xavier_normal_', 
                 dtype=hydraulis.float32, name='colp'):
        super(HtMultiColumnParallelLinear, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.name = name

        self.ds_union_map = {'dup_split0': [], 'split0_dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split1, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            zero = ds_parallel_config['zero']
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split1.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split1.get(i).get_dim(1) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split1.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "ColumnParallelLinear only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union])
                , f'ColumnParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!')        
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            # when dp=1 tp=8, weights: ds_dup_split0->ds_split0, data: ds_split0_dup->ds_dup
            # when dp=8 tp=1, weights: ds_dup_split0->ds_dup, data: ds_split0_dup->ds_split0
            ds_list_dup_split0 = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: dcp_union[i], 0: tp_union[i]}, [-1, 0], zero)
                for i in range(hetero_size)] # for weights with trans_b
            ds_list_split0_dup = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: tp_union[i], 0: dcp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data
            ds_union_dup_split0 = hydraulis.DistributedStatesUnion(ds_list_dup_split0, -1 if hetero_dim != -3 else -3) # for weights with trans_b
            ds_union_split0_dup = hydraulis.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3) # for data
            self.ds_union_map['dup_split0'].append(ds_union_dup_split0)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
        
        self.weight = hydraulis.parallel_parameter(eval(f'hydraulis.{init_method}initializer()'), 
                                              [out_features, in_features], 
                                              self.ds_union_map['dup_split0'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')
        if bias:
            self.bias = hydraulis.parallel_parameter(hydraulis.zeros_initializer(), [out_features], 
                                                self.ds_union_map['dup_split0'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_group_hierarchy=self.device_group_unions, name=f'{name}_bias')
        else:
            self.bias = None
      
    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hydraulis.comm(input_p, self.ds_union_map['split0_dup'])
            '''
            print(f"sp: column parallel linear need extra communication for \
                    adapt input tensor ds hierarchy {input_p.ds_hierarchy} into {self.ds_union_map['split0_dup']}!")
            '''
        
        if self.bias != None:
            tensor_split01 = hydraulis.linear(tensor_split0_dup, self.weight, self.bias, trans_a=False, trans_b=True, device_group_hierarchy=self.device_group_unions, name=self.name)
        else:
            tensor_split01 = hydraulis.linear(tensor_split0_dup, self.weight, trans_a=False, trans_b=True, device_group_hierarchy=self.device_group_unions, name=self.name)
        
        if not self.gather_output:
            output = tensor_split01
        else:
            if tensor_split01.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']): # pure dp
                output = tensor_split01
            else:
                output = hydraulis.comm(tensor_split01, self.ds_union_map['split0_dup'])

        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class HtMultiRowParallelLinear(Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """
    def __init__(self, in_features, out_features, 
                 multi_ds_parallel_config, sp=False, bias=True, 
                 init_method='xavier_normal_', 
                 dtype=hydraulis.float32, name='rowp'):
        super(HtMultiRowParallelLinear, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.sp = sp
        
        self.ds_union_map = {'dup_split0': [], 'dup_split1': [], 'dup': [], 'split0': [], 'split01': [], 'split0_dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split0, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            zero = ds_parallel_config['zero']
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split0.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split0.get(i).get_dim(0) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split0.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "RowParallelLinear only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union])
                , f'RowParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!')        
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            ds_list_dup_split1 = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: dcp_union[i], 1: tp_union[i]}, [-1, 1], zero)
                for i in range(hetero_size)] # for weight with trans_b
            ds_list_dup = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: dcp_union[i] * tp_union[i]}, [-1], zero)
                for i in range(hetero_size)] # for bias
            ds_list_split0 = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: dcp_union[i] * tp_union[i]}, [0])
                for i in range(hetero_size)] # for sp data
            ds_list_split01 = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: dcp_union[i], 1: tp_union[i]}, [0, 1])
                for i in range(hetero_size)] # for data split in dimension 1
            ds_list_split0_dup = [hydraulis.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: dcp_union[i], -1: tp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data reduce partial to dup
            ds_union_dup_split1 = hydraulis.DistributedStatesUnion(ds_list_dup_split1, -1 if hetero_dim != -3 else -3) # for weight with trans_b
            ds_union_dup = hydraulis.DistributedStatesUnion(ds_list_dup, -1 if hetero_dim != -3 else -3) # for bias
            ds_union_split0 = hydraulis.DistributedStatesUnion(ds_list_split0, 0 if hetero_dim != -3 else -3) # for sp data
            ds_union_split01 = hydraulis.DistributedStatesUnion(ds_list_split01, 0 if hetero_dim != -3 else -3) # for data split in dimension 1
            ds_union_split0_dup = hydraulis.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3) # for data reduce partial to dup
            self.ds_union_map['dup_split0'].append(ds_union_dup_split0)
            self.ds_union_map['dup_split1'].append(ds_union_dup_split1)
            self.ds_union_map['dup'].append(ds_union_dup)
            self.ds_union_map['split0'].append(ds_union_split0)
            self.ds_union_map['split01'].append(ds_union_split01)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
            
        self.weight = hydraulis.parallel_parameter(eval(f'hydraulis.{init_method}initializer()'), 
                                              [in_features, out_features], 
                                              self.ds_union_map['dup_split0'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')        
        if bias:
            self.bias = hydraulis.parallel_parameter(hydraulis.zeros_initializer(), [out_features], 
                                                self.ds_union_map['dup'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_group_hierarchy=self.device_group_unions, name=f'{name}_bias')
        else:
            self.bias = None

    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.ds_union_map['split01']):
            tensor_split01 = input_p
        else:
            tensor_split01 = hydraulis.comm(input_p, self.ds_union_map['split01']) # exists src_ds == dst_ds case, just ignore it in comm_op
        
        tensor_split0_partial = hydraulis.linear(tensor_split01, self.weight, trans_a=False, trans_b=False, device_group_hierarchy=self.device_group_unions, name=self.name)
         
        if self.sp:
            output = hydraulis.comm(tensor_split0_partial, self.ds_union_map['split0']) # reduce-scatter
        else:
            output = hydraulis.comm(tensor_split0_partial, self.ds_union_map['split0_dup']) # allreduce   
        
        output = output + self.bias if self.bias is not None else output
        
        return output
