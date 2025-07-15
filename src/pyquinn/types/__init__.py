"""
This package contains basic types supported in the PyQuinn framework
"""

from pyquinn.types.atom_graph import AtomGraph
from pyquinn.types.bandstructure import BandStructure
from pyquinn.types.dataset import AtomGraphDataset
from pyquinn.types.bandstructure_dataset import BandStructureDataset


_module = "pyquinn.types"
_module_symbols = (
    AtomGraph,
    BandStructure,
    AtomGraphDataset,
    BandStructureDataset
)

for _symbol in _module_symbols:
    _symbol.__module__ = _module