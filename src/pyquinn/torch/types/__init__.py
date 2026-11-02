"""This package contains PyTorch Backend types."""

from pyquinn.torch.types.atom_graph import TorchAtomGraph
from pyquinn.torch.types.bandstructure import TorchBandStructure
from pyquinn.torch.types.dataset import TorchAtomGraphDataset
from pyquinn.torch.types.bandstructure_dataset import TorchBandStructureDataset
from pyquinn.torch.types.operator import TorchOperator
from pyquinn.torch.types.operator_dataset import TorchOperatorDataset

_module = "pyquinn.torch.types"
_module_symbols = (
    TorchAtomGraph,
    TorchBandStructure,
    TorchOperator,
    TorchAtomGraphDataset,
    TorchBandStructureDataset,
    TorchOperatorDataset,
)

for _symbol in _module_symbols:
    _symbol.__module__ = _module
