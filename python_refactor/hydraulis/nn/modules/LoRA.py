import hydraulis
from hydraulis import Tensor
from .module import Module
import math

from typing import Any

__all__ = [
    'LoRALinear', 
]

class LoRALinear(Module):
    
    def __init__(self, in_features: int, out_features: int, rank: int, p: float = 0, bias: bool = True, device_group = None) -> None:
        with hydraulis.graph("define_and_run"):
            super(LoRALinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            # self.weight = hydraulis.nn.functional.kaiming_uniform_([out_features, in_features], a = math.sqrt(5), requires_grad=True, device_group = device_group)
            self.weight = hydraulis.nn.functional.kaiming_uniform_([out_features, in_features], 
                                                              a = math.sqrt(5), requires_grad=False, 
                                                              device_group = device_group)
            self.weightA = hydraulis.nn.functional.kaiming_uniform_([rank, in_features], 
                                                               a = math.sqrt(5), requires_grad=True, 
                                                               device_group = device_group)
            self.weightB = hydraulis.zeros([out_features, rank], requires_grad=True, device_group = device_group)
            if p < 0 or p > 1:
                raise ValueError("dropout probability has to be between 0 and 1, "
                                "but got {}".format(p))
            self.p = 1 - p
            if bias:
                fan_in, _ = hydraulis.nn.functional._calculate_fan_in_and_fan_out(self.weighydraulis.shape)
                bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias = hydraulis.rand([out_features], -bound, bound, requires_grad=True, device_group = device_group)
            else:
                self.register_parameter('bias', None)
    
    
    def forward(self, input: Tensor, weight: Tensor = None) -> Tensor:
        if self.p < 1:
            input = hydraulis.dropout(input, self.p, False)
        if self.bias is not None:
            return hydraulis.matmul(input, self.weight, trans_b=True) + \
                    hydraulis.matmul(hydraulis.matmul(input, self.weightA, trans_b=True), 
                                self.weightB, trans_b=True) + self.bias
        else:
            return hydraulis.matmul(input, self.weight, trans_b=True) + \
                    hydraulis.matmul(hydraulis.matmul(input, self.weightA, trans_b=True), 
                                self.weightB, trans_b=True)


class LoRAEmbedding(Module):
    
    def __init__(self, num_embeddings, embedding_dim, rank, device_group = None) -> None:
        with hydraulis.graph("define_and_run"):
            super(LoRAEmbedding, self).__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = hydraulis.nn.functional.xavier_normal_([num_embeddings, embedding_dim], 
                                                            requires_grad=False, device_group = device_group)
            self.weightA = hydraulis.zeros([num_embeddings, rank], requires_grad=True, device_group = device_group)
            self.weightB = hydraulis.randn([embedding_dim, rank], 0, 1, requires_grad=True, device_group = device_group)
    
    def forward(self, input: Tensor) -> Tensor:
        return hydraulis.embedding_lookup(self.weight, input) + \
               hydraulis.matmul(hydraulis.embedding_lookup(self.weightA, input), self.weightB, trans_b=True)
