from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pyquinn.torch.util import get_activation_fn

class EdgewiseConv(nn.Module, ABC):

    _modules = {}

    @staticmethod
    def register(name: str):
        """Registers an EdgewiseConv subclass of this abstract class. You can
        add a custom EdgewiseConv module by using the python decorator syntax as
        follows:

        .. code-block:: python

            from pyquinn.torch import EdgewiseConv

            @AtomwiseConv.register('my_conv_name')
            class MyEdgewiseConv(EdgewiseConv):
            
                def forward(...):
                    ...

        Args:
            name (str): AtomwiseConv class to be registered.
        """
        def decorator(subclass):
            if name not in EdgewiseConv._modules:
                EdgewiseConv._modules[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def from_name(
            module_name: str, 
            in_features: int | None = None,
            out_features: int | None = None,
            in_edge_features: int | None = None,
            out_edge_features: int | None = None,
            z_max: int = 118,
            module_kwargs: dict = {}):
        
        if module_name not in EdgewiseConv._modules:
            raise ValueError(f'Unknown EdgewiseConv Type: {module_name}')

        return EdgewiseConv._modules[module_name](
            out_features=out_features,
            out_edge_features=out_edge_features,
            in_features=in_features,
            in_edge_features=in_edge_features,
            z_max=z_max,
            **module_kwargs)

    @abstractmethod
    def __init__(
            self,
            out_features: int,
            out_edge_features: int,
            in_features: int,
            in_edge_features: int,
            z_max: int = 118):

        self.out_features = out_features
        self.out_edge_features = out_edge_features
        self.in_features = in_features
        self.in_edge_features = in_edge_features
        
    @abstractmethod
    def forward(
            self,
            atomic_numbers,
            neighbors,
            neighbor_distances):

        raise NotImplementedError(
            "This method should be implemented in a "
            "subclass.")
