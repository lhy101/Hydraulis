import hydraulis
from hydraulis import Tensor
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'ReLU', 
    'Sigmoid', 
    'Tanh', 
    'LeakyReLU',
    'NewGeLU'
]


class ReLU(Module):

    def __init__(self, inplace: bool = False):
        with hydraulis.graph("define_and_run"):
            super(ReLU, self).__init__()
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hydraulis.graph("define_and_run"):
            if self.inplace:
                return hydraulis.relu_(input)
            else:
                return hydraulis.relu(input)


class Sigmoid(Module):

    def __init__(self, inplace: bool = False):
        with hydraulis.graph("define_and_run"):
            super(Sigmoid, self).__init__()
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hydraulis.graph("define_and_run"):
            if self.inplace:
                return hydraulis.sigmoid_(input)
            else:
                return hydraulis.sigmoid(input)


class Tanh(Module):

    def __init__(self, inplace: bool = False):
        with hydraulis.graph("define_and_run"):
            super(Tanh, self).__init__()
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hydraulis.graph("define_and_run"):
            if self.inplace:
                return hydraulis.tanh_(input)
            else:
                return hydraulis.tanh(input)


class LeakyReLU(Module):

    def __init__(self, negative_slope: float = 0.1, inplace: bool = False):
        with hydraulis.graph("define_and_run"):
            super(LeakyReLU, self).__init__()
            self.negative_slope = negative_slope
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hydraulis.graph("define_and_run"):
            if self.inplace:
                return hydraulis.leakyrelu_(input, self.negative_slope)
            else:
                return hydraulis.leakyrelu(input, self.negative_slope)
      
        
class NewGeLU(Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self, inplace: bool = False):
        with hydraulis.graph("define_and_run"):
            super(NewGeLU, self).__init__()
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hydraulis.graph("define_and_run"):
            #  0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
            # TODO: implement hydraulis.pow(input, 3.0) to replace input * input * input, or implement a cuda kernel
            return 0.5 * input * (1.0 + hydraulis.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * (input * input * input))))