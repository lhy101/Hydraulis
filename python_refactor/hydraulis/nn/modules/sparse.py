import hydraulis
from hydraulis import Tensor
from .module import Module
import math

from typing import Any

__all__ = [
    'Embedding', 
]

class Embedding(Module):
    
    def __init__(self, num_embeddings, embedding_dim, device_groups = None) -> None:
        with hydraulis.graph("define_and_run"):
            super(Embedding, self).__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = hydraulis.nn.functional.xavier_normal_([num_embeddings, embedding_dim], requires_grad=True, device_groups=device_groups)
    
    def forward(self, input: Tensor) -> Tensor:
        return hydraulis.embedding_lookup(self.weight, input)
