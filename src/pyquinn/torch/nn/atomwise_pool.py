from abc import ABC, abstractmethod

import torch.nn as nn
import torch


class AtomwisePool(nn.Module, ABC):

    _modules = {}

    @staticmethod
    def register(name: str):
        """Registers an AtomWisePool subclass of this abstract class. You can
        add a custom AtomwisePool module by using the python decorator syntax 
        as follows:

        .. code-block:: python

            from pyquinn.torch import AtomwisePool

            @AtomwisePool.register('my_pool_name')
            class MyAtomwisePool(AtomwisePool):
            
                def forward(...):
                    ...

        Args:
            name (str): AtomwisePool class to be registered.
        """
        def decorator(subclass):
            if name not in AtomwisePool._modules:
                AtomwisePool._modules[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def from_name(
            module_name: str, 
            in_features: int,
            out_features: int | None = None,
            z_max: int = 118,
            module_kwargs: dict | None = None):
        
        if module_name not in AtomwisePool._modules:
            raise ValueError(f'Unknown AtomwisePool Type: {module_name}')

        module_kwargs = {} if module_kwargs is None else module_kwargs

        return AtomwisePool._modules[module_name](
            in_features=in_features, 
            out_features=out_features,
            z_max=z_max,
            **module_kwargs)

    def __init__(
            self, 
            out_features: int, 
            in_features: int | None = None,
            z_max: int = 118):

        super().__init__()

        self.out_features = out_features
        self.in_features = in_features if in_features is not None else 0
        self.z_max = z_max

    @abstractmethod
    def forward(self,
                atomic_numbers,
                features,
                **kwargs) -> tuple:

        raise NotImplementedError(
            "This method should be implemented in a "
            "subclass.")


@AtomwisePool.register('arithmetic')
class ArithmeticPool(AtomwisePool):

    _aggr_ops = {
        'sum': torch.sum,
        'mean': torch.mean,
        'min': torch.min,
        'max': torch.max
    }

    def __init__(
            self,
            aggr='sum',
            out_features: int | None = None,
            in_features: int | None = None,
            z_max: int = 118):

        if aggr not in self._aggr_ops:
            oplist = list(self._aggr_ops.keys())
            raise ValueError(
                f'Invalid aggr value "{aggr}"; '
                f'expected one of {oplist}')

        self.aggr = aggr
    
    def forward(
            self,
            atomic_numbers: torch.IntTensor,
            features: torch.Tensor | None = None) -> torch.Tensor:

        return self._aggr_ops[self.aggr](features, dim=0)