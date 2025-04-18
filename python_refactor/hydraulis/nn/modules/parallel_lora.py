import hydraulis
from .module import Module
import numbers
from .parallel_ds import HtColumnParallelLinear, HtParallelEmbedding, HtParallelLayerNorm,\
                        HtRowParallelLinear, HtVocabParallelEmbedding

__all__ = [
    'LoRAColumnParallelLinear', 
    'LoRARowParallelLinear', 
    'LoRAParallelEmbedding',
    'LoRAVocabParallelEmbedding',
    'LoRAParallelLayerNorm',
    'LoRAModel',
]

def parallel_data_provider(global_data, ds, device_index):
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

def get_device_index(device_group):
    local_device = hydraulis.local_device()
    if device_group.contains(local_device):
        device_index = device_group.get_index(local_device)
    else: # for pipeline parallel other stages
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_index

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    num_devices = len(config['device_group'])
    zero = False
    split = {}
    for key, value in config['split'].items():
        split[int(key)] = value
    states = {-1: config['dup'], **split}
    if config['type'] == 'placeholder':
        order = sorted(split.keys()) + [-1]
    elif config['type'] == 'variable':
        order = [-1] + sorted(split.keys())
        assert 'zero' in config, f"variable config must have zero!"
        zero = config['zero']
    else:
        raise RuntimeError(f"unsupported type {config['type']}!")
    ds = hydraulis.DistributedStates(num_devices, states, order, zero)
    
    all_devices = hydraulis.global_device_group()
    device_group = hydraulis.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group']])
    return ds, device_group

class LoRAParallelLayerNorm(HtParallelLayerNorm):
    def __init__(self, ori_model, blocksize = 64, qtype = hydraulis.bfloat16,
                 map_absmax = {}, name='ln'):
        HtParallelLayerNorm.__init__(self, ori_model.normalized_shape, ori_model.ds_parallel_config,
                                     ori_model.eps, dtype=ori_model.dtype, name='ln', ori_model = ori_model)
        ds, self.device_group = config2ds(self.ds_parallel_config)
        device_index = get_device_index(self.device_group)
        self.blocksize = blocksize
        lora_dtype = self.dtype
        if (lora_dtype == hydraulis.float4) or (lora_dtype == hydraulis.nfloat4):
            lora_dtype = qtype
        # self.weight = ori_model.weight
        self.weighydraulis.set_requires_grad(False)
        # self.bias = ori_model.bias
        self.bias.set_requires_grad(False)
        if self.dtype == hydraulis.float4 or self.dtype == hydraulis.nfloat4:
            absmax_size = self.weighydraulis.global_numel() // blocksize
            if (self.weighydraulis.id not in map_absmax):
                parameter_dict = {"tensor_id": self.weighydraulis.id, "blocksize": blocksize}
                self.weight_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                             [absmax_size,], [ds], [device_index], 
                                                             dtype=hydraulis.float32, requires_grad=False,
                                                             parameter_dict = parameter_dict, 
                                                             device_groups=[self.device_group], 
                                                             name=f'{name}_weight_absmax')
                map_absmax[self.weighydraulis.id] = self.weight_absmax
            else:
                self.weight_absmax = map_absmax[self.weighydraulis.id]
            if (self.bias.id not in map_absmax):
                parameter_dict = {"tensor_id": self.bias.id, "blocksize": blocksize}
                self.bias_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                            [absmax_size,], [ds], [device_index], 
                                                            dtype=hydraulis.float32, requires_grad=False, 
                                                            parameter_dict = parameter_dict,
                                                            device_groups=[self.device_group], 
                                                            name=f'{name}_bias_absmax')
                map_absmax[self.bias.id] = self.bias_absmax
            else:
                self.bias_absmax = map_absmax[self.bias.id]

    def forward(self, input_p):
        return hydraulis.layer_norm(input_p, self.weight, self.bias, self.normalized_shape, 
                               self.eps, device_groups=[self.device_group], name=self.name)[0]

class LoRAParallelEmbedding(HtParallelEmbedding):
    def __init__(self, ori_model, rank, blocksize = 64, qtype = hydraulis.bfloat16,
                 map_absmax = {}, name='embedding_lora'):
        HtParallelEmbedding.__init__(self, ori_model.num_embeddings, ori_model.embedding_dim, 
                                     ori_model.ds_parallel_config,
                                     init_method='xavier_normal_', dtype=ori_model.dtype, name='embedding',
                                     ori_model = ori_model)
        self.rank = rank
        # self.embedding_table = ori_model.embedding_table
        self.embedding_table.set_requires_grad(False)
        ds_dup = hydraulis.DistributedStates(self.ds_parallel_config['dup'], {-1: self.ds_parallel_config['dup']}, [-1])
        if self.dtype == hydraulis.float4 or self.dtype == hydraulis.nfloat4:
            absmax_size = self.embedding_table.global_numel() // blocksize
            if (self.embedding_table.id not in map_absmax):
                parameter_dict = {"tensor_id": self.embedding_table.id, "blocksize": blocksize}
                self.embedding_table_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                                      [absmax_size,], [ds_dup], 
                                                                      [get_device_index(self.device_group)],
                                                                      dtype=hydraulis.float32, requires_grad=False, 
                                                                      parameter_dict = parameter_dict,
                                                                      device_groups=[self.device_group], 
                                                                      name=f'{name}_table_absmax')
                map_absmax[self.embedding_table.id] = self.embedding_table_absmax
            else:
                self.embedding_table_absmax = map_absmax[self.embedding_table.id]
    
    def forward(self, input_p):
        return HtParallelEmbedding.forward(self, input_p)
    
class LoRAVocabParallelEmbedding(HtVocabParallelEmbedding):
    def __init__(self, ori_model, rank, blocksize = 64,
                 lora_alpha = 1, qtype = hydraulis.bfloat16,
                 map_absmax = {}, name='vocab_embedding_lora'):
        HtVocabParallelEmbedding.__init__(self, ori_model.num_embeddings, ori_model.embedding_dim, 
                                          ori_model.ds_parallel_config,
                                          init_method='zeros_', dtype=ori_model.dtype, 
                                          name='vocab_embedding', ori_model = ori_model)
        self.rank = rank
        # self.embedding_table = ori_model.embedding_table
        self.embedding_table.set_requires_grad(False)
        if self.dtype == hydraulis.float4 or self.dtype == hydraulis.nfloat4:
            absmax_size = self.embedding_table.global_numel() // blocksize
            if (self.embedding_table.id not in map_absmax):
                parameter_dict = {"tensor_id": self.embedding_table.id, "blocksize": blocksize}
                self.embedding_table_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                                      [absmax_size,], [self.ds_map['dup']], 
                                                                      [get_device_index(self.device_group)],
                                                                      dtype=hydraulis.float32, requires_grad=False, 
                                                                      parameter_dict = parameter_dict,
                                                                      device_groups=[self.device_group], 
                                                                      name=f'{name}_table_absmax')
                map_absmax[self.embedding_table.id] = self.embedding_table_absmax
            else:
                self.embedding_table_absmax = map_absmax[self.embedding_table.id]
                
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        lora_dtype = self.dtype
        if (lora_dtype == hydraulis.float4) or (lora_dtype == hydraulis.nfloat4):
            lora_dtype = qtype
        self.lora_dtype = lora_dtype
        if self.lora_alpha != 0:
            self.embedding_tableA = HtVocabParallelEmbedding(self.num_embeddings, rank, self.ds_parallel_config,
                                                            init_method='zeros_', 
                                                            dtype=lora_dtype, name='vocab_embedding_loraA')
            self.embedding_tableB = HtRowParallelLinear(rank, self.embedding_dim, self.ds_parallel_config,
                                                        bias=False, init_method='normal_', 
                                                        dtype=lora_dtype, name='vocab_embedding_loraB')
        
    
    def forward(self, input_p):
        output = HtVocabParallelEmbedding.forward(self, input_p)
        # if self.lora_alpha != 0:
        #     output = output + self.embedding_tableB(self.embedding_tableA(input_p)) * self.lora_alpha
        return output
    
class LoRAColumnParallelLinear(HtColumnParallelLinear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, ori_model, rank, blocksize = 64,
                 lora_alpha = 1, qtype = hydraulis.bfloat16, 
                 map_absmax = {}, name='colp_lora'):
        HtColumnParallelLinear.__init__(self, ori_model.in_features, ori_model.out_features, 
                                        ori_model.ds_parallel_config,
                                        bias=False if (ori_model.bias is None) else True, 
                                        gather_output = ori_model.gather_output, 
                                        init_method='zeros_', dtype=ori_model.dtype, 
                                        name='colp', ori_model = ori_model)
        self.rank = rank     
        # self.weight = ori_model.weight
        self.weighydraulis.set_requires_grad(False)
        if self.dtype == hydraulis.float4 or self.dtype == hydraulis.nfloat4:
            absmax_size = self.weighydraulis.global_numel() // blocksize
            if (self.weighydraulis.id not in map_absmax):
                parameter_dict = {"tensor_id": self.weighydraulis.id, "blocksize": blocksize}
                self.weight_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                             [absmax_size], [self.ds_map['dup']], 
                                                             [get_device_index(self.device_group)], 
                                                             dtype=hydraulis.float32, requires_grad=False, 
                                                             parameter_dict = parameter_dict,
                                                             device_groups=[self.device_group], 
                                                             name=f'{name}_weight_absmax')
                map_absmax[self.weighydraulis.id] = self.weight_absmax
            else:
                self.weight_absmax = map_absmax[self.weighydraulis.id]

        lora_dtype = self.dtype
        if (lora_dtype == hydraulis.float4) or (lora_dtype == hydraulis.nfloat4):
            lora_dtype = qtype
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        if self.lora_alpha != 0:
            self.lora_A = HtColumnParallelLinear(self.in_features, rank, self.ds_parallel_config,
                                                 bias=False, gather_output=True, init_method='he_uniform_', 
                                                 dtype=lora_dtype, name='colp_lora_A')
            self.lora_B = HtColumnParallelLinear(rank, self.out_features, self.ds_parallel_config,
                                                 bias=False, gather_output=self.gather_output, init_method='zeros_', 
                                                 dtype=lora_dtype, name='colp_lora_B')
        if self.bias is not None:
            # self.bias = ori_model.bias
            self.bias.set_requires_grad(False)
            if self.dtype == hydraulis.float4 or self.dtype == hydraulis.nfloat4:
                absmax_size = self.bias.global_numel() // blocksize
                if (self.bias.id not in map_absmax):
                    parameter_dict = {"tensor_id": self.bias.id, "blocksize": blocksize}
                    self.bias_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                               [absmax_size], [self.ds_map['dup']], 
                                                               [get_device_index(self.device_group)],
                                                               dtype=hydraulis.float32, requires_grad=False,
                                                               parameter_dict = parameter_dict, 
                                                               device_groups=[self.device_group], 
                                                               name=f'{name}_bias_absmax')
                    map_absmax[self.bias.id] = self.bias_absmax
                else:
                    self.bias_absmax = map_absmax[self.bias.id]
      
    def forward(self, input_p):
        output = HtColumnParallelLinear.forward(self, input_p)  
        if self.lora_alpha != 0:
            output = output + self.lora_B(self.lora_A(input_p)) * self.lora_alpha

        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class LoRARowParallelLinear(HtRowParallelLinear):
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
    def __init__(self, ori_model, rank, blocksize = 64, 
                 lora_alpha = 1, qtype = hydraulis.bfloat16,
                 map_absmax = {}, name='rowp_lora'):
        HtRowParallelLinear.__init__(self, ori_model.in_features, ori_model.out_features, 
                                     ori_model.ds_parallel_config,
                                     bias=False if (ori_model.bias is None) else True,
                                     init_method='zeros_', dtype=ori_model.dtype, 
                                     name='rowp', ori_model = ori_model)
        self.rank = rank
        # self.weight = ori_model.weight
        self.weighydraulis.set_requires_grad(False)
        if self.dtype == hydraulis.float4 or self.dtype == hydraulis.nfloat4:
            absmax_size = self.weighydraulis.global_numel() // blocksize
            if (self.weighydraulis.id not in map_absmax):
                parameter_dict = {"tensor_id": self.weighydraulis.id, "blocksize": blocksize}
                self.weight_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                             [absmax_size], [self.ds_map['dup']],
                                                             [get_device_index(self.device_group)], 
                                                             dtype=hydraulis.float32, requires_grad=False, 
                                                             parameter_dict = parameter_dict,
                                                             device_groups=[self.device_group], 
                                                             name=f'{name}_weight_absmax')
                map_absmax[self.weighydraulis.id] = self.weight_absmax
            else:
                self.weight_absmax = map_absmax[self.weighydraulis.id]
        
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        lora_dtype = self.dtype
        if (lora_dtype == hydraulis.float4) or (lora_dtype == hydraulis.nfloat4):
            lora_dtype = qtype
        if self.lora_alpha != 0:
            # lora_b_ds_parallel_config = self.ds_parallel_config.copy()
            # lora_b_ds_parallel_config["split"] = {}
            # if "0" in self.ds_parallel_config["split"]:
            #     lora_b_ds_parallel_config["split"]["1"] = self.ds_parallel_config["split"]["0"]
            # if "1" in self.ds_parallel_config["split"]:
            #     lora_b_ds_parallel_config["split"]["0"] = self.ds_parallel_config["split"]["1"]
            
            # self.lora_A = HtRowParallelLinear(self.in_features, rank, self.ds_parallel_config,
            #                                 bias=False, init_method='he_uniform_', 
            #                                 dtype=lora_dtype, name=f'rowp_lora_A_{self.name}')
            # self.lora_B = HtColumnParallelLinear(rank, self.out_features, lora_b_ds_parallel_config,
            #                                     bias=False, gather_output=True, init_method='zeros_', 
            #                                     dtype=lora_dtype, name=f'colp_lora_B_{self.name}')  
            self.lora_A = HtRowParallelLinear(self.in_features, rank, self.ds_parallel_config,
                                              bias=False, init_method='he_uniform_', 
                                              dtype=lora_dtype, name='rowp_lora_A')
            self.lora_B = HtRowParallelLinear(rank, self.out_features, self.ds_parallel_config,
                                              bias=False, init_method='zeros_', 
                                              dtype=lora_dtype, name='rowp_lora_B')      
        if self.bias is not None:
            # self.bias = ori_model.bias
            self.bias.set_requires_grad(False)
            if self.dtype == hydraulis.float4 or self.dtype == hydraulis.nfloat4:
                absmax_size = self.bias.global_numel() // blocksize
                if (self.bias.id not in map_absmax):
                    parameter_dict = {"tensor_id": self.bias.id, "blocksize": blocksize}
                    self.bias_absmax = hydraulis.parallel_parameter(eval(f'hydraulis.zeros_initializer()'), 
                                                               [absmax_size], [self.ds_map['dup']],
                                                               [get_device_index(self.device_group)], 
                                                               dtype=hydraulis.float32, requires_grad=False,
                                                               parameter_dict = parameter_dict,
                                                               device_groups=[self.device_group], 
                                                               name=f'{name}_bias_absmax')
                    map_absmax[self.bias.id] = self.bias_absmax
                else:
                    self.bias_absmax = map_absmax[self.bias.id]

    def forward(self, input_p):
        output = HtRowParallelLinear.forward(self, input_p)  
        if self.lora_alpha != 0:
            output = output + self.lora_B(self.lora_A(input_p)) * self.lora_alpha
        return output
    
class LoRAModel(Module):
    def __init__(self, ori_model, rank, blocksize = 64, 
                 lora_alpha = 1, qtype = hydraulis.bfloat16, name='lora_model'):
        super(LoRAModel, self).__init__()
        self.model = ori_model
        self.forward = self.model.forward
        self.state_dict = self.model.state_dict
        self.load_state_dict = self.model.load_state_dict
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.blocksize = blocksize
        self.qtype = qtype
        self.id_to_absmax = {}
        self.prepare_for_lora(self.model)
            

    def prepare_for_lora(self, model):
        for key, model_ in model.named_modules():
            lora_model_ = None
            parent_model_ = None
            parent_key = ''
            if isinstance(model_, hydraulis.nn.HtParallelLayerNorm):
                parent_key = ".".join(key.split(".")[:-1])
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRAParallelLayerNorm(model_, 
                                                    blocksize = self.blocksize, 
                                                    qtype = self.qtype,
                                                    map_absmax = self.id_to_absmax)
            if isinstance(model_, hydraulis.nn.HtColumnParallelLinear):
                parent_key = ".".join(key.split(".")[:-1])
                alpha = 0
                if "attn" in parent_key:
                    alpha = self.lora_alpha
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRAColumnParallelLinear(model_, self.rank, 
                                                       blocksize = self.blocksize, 
                                                       lora_alpha = alpha, 
                                                       qtype = self.qtype,
                                                       map_absmax = self.id_to_absmax)
            if isinstance(model_, hydraulis.nn.HtRowParallelLinear):
                parent_key = ".".join(key.split(".")[:-1])
                alpha = 0
                if "attn" in parent_key:
                    alpha = self.lora_alpha
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRARowParallelLinear(model_, self.rank,
                                                    blocksize = self.blocksize, 
                                                    lora_alpha = alpha, 
                                                    qtype = self.qtype,
                                                    map_absmax = self.id_to_absmax)
            if isinstance(model_, hydraulis.nn.HtParallelEmbedding):
                parent_key = ".".join(key.split(".")[:-1])
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRAParallelEmbedding(model_, self.rank, 
                                                    blocksize = self.blocksize,
                                                    map_absmax = self.id_to_absmax)
            if isinstance(model_, hydraulis.nn.HtVocabParallelEmbedding):
                parent_key = ".".join(key.split(".")[:-1])
                parent_model_ = model.find_module(parent_key)
                lora_model_ = LoRAVocabParallelEmbedding(model_, self.rank, 
                                                         blocksize = self.blocksize,
                                                         lora_alpha = 0,
                                                         qtype = self.qtype,
                                                         map_absmax = self.id_to_absmax)
            if parent_model_ is not None:
                setattr(parent_model_, key.split(".")[-1], lora_model_)
            model_ = lora_model_  
        # for key, model_ in self.model.named_modules():
        #     print("key:", key, "model:", model_) 
            

        