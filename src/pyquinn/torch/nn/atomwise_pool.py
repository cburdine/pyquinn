from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
        module_kwargs: dict | None = None,
    ):

        if module_name not in AtomwisePool._modules:
            raise ValueError(f"Unknown AtomwisePool Type: {module_name}")

        module_kwargs = {} if module_kwargs is None else module_kwargs

        return AtomwisePool._modules[module_name](
            in_features=in_features, out_features=out_features, **module_kwargs
        )

    def __init__(
        self, out_features: int | None = None, in_features: int | None = None
    ):

        super().__init__()

        self.out_features = out_features
        self.in_features = in_features

    @abstractmethod
    def forward(self, atomic_numbers, features, **kwargs) -> tuple:

        raise NotImplementedError(
            "This method should be implemented in a " "subclass."
        )


@AtomwisePool.register("arithmetic")
class ArithmeticPool(AtomwisePool):

    _pool_ops = {
        "sum": torch.sum,
        "mean": torch.mean,
        "min": torch.min,
        "max": torch.max,
    }

    def __init__(
        self,
        aggr="sum",
        out_features: int | None = None,
        in_features: int | None = None,
    ):

        super().__init__(
            out_features=out_features,
            in_features=in_features,
        )

        if aggr not in ArithmeticPool._pool_ops:
            oplist = list(ArithmeticPool._pool_ops.keys())
            raise ValueError(
                f'Invalid aggr value "{aggr}"; ' f"expected one of {oplist}"
            )

        self.pool_op = ArithmeticPool._pool_ops[aggr]

    def forward(
        self,
        atomic_numbers: torch.IntTensor,
        features: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:

        return self.pool_op(features, dim=0)
