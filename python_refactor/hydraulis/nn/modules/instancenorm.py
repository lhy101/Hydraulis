import hydraulis
from hydraulis import Tensor
from .module import Module
import math
from .batchnorm import NormBase
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'InstanceNorm',
]

class InstanceNorm(NormBase):

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        with hydraulis.graph("define_and_run"):
            super(InstanceNorm, self).__init__(num_features, eps, 0)

    def forward(self, input: Tensor) -> Tensor:
        return hydraulis.instance_norm(input, self.eps)[0]

