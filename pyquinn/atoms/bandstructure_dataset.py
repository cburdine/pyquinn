from abc import ABC, abstractmethod

from pyquinn.atoms.atom_graph import AtomGraph
from pyquinn.atoms.dataset import AtomGraphDataset
from pyquinn.atoms.bandstructure import BandStructure

import numpy as np

SUPPORTED_BANDSTRUCTURE_DATASET_BACKENDS = []
""" The full list of dataset backends supported by default"""


class BandStructureDatasetBase:
    """ Base class for BandStructureDataset instances"""


class BandStructureDataset(BandStructureDatasetBase):

    _backends = {}

    @classmethod
    def register_backend(cls, name: str):

        def decorator(subclass):
            if name not in cls._backends:
                SUPPORTED_BANDSTRUCTURE_DATASET_BACKENDS.append(name)
            cls._backends[name] = subclass
            return subclass
        return decorator

    def __init__(
            self,
            graphs: list[AtomGraph] | list[str] = [],
            band_structures: list[BandStructure] | list[str] = [],
            names: list[str] | None = None):

        self.graphs = []
        self.names = []
        self.band_structures = []

        self.extend(
            graphs=graphs, 
            band_structures=band_structures, 
            names=names)

    def append(
            self,
            graph: AtomGraph | str,
            bands: BandStructure | str,
            name: str | None = None):

        n_graphs = len(self.graphs)
        name = AtomGraphDataset._resolve_graph_name(n_graphs, graph, name)

        self.graphs.append(graph)
        self.band_structures.append(bands)
        self.names.append(name)        

    def extend(
            self,
            graphs: list[AtomGraph] | list[str],
            band_structures: list[BandStructure] | list[str],
            names: list[str] | None = None):

        if names is None:
            names = [None]*len(graphs)

        if len(graphs) != len(names):
            raise ValueError(f'Expected len(graphs) == len(names); '
                             f'got: {len(graphs)} != {len(names)}.')
        if len(graphs) != len(band_structures):
            raise ValueError(f'Expected len(graphs) == len(band_structures); '
                             f'got: {len(graphs)} != {len(band_structures)}.')

        # resolve the name of each graph:
        n_graphs = len(self.graphs)
        names = [
            AtomGraphDataset._resolve_graph_name(n_graphs+i, g, gn)
            for i, (g, gn) in enumerate(zip(graphs, names))
        ]

        self.graphs.extend(graphs)
        self.names.extend(names)

    def _validate_names(self):

        names_dict = {}
        for i, name in enumerate(self.names):
            if name in names_dict:
                raise ValueError(
                    f'Found more than one occurence of graph with name '
                    f'"{name}" at indices {names_dict[name]} and {i}')
            else:
                names_dict[name] = i

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[AtomGraph, BandStructure]:

        # retrieve graph
        graph = self.graphs[idx]
        if isinstance(graph, str):
            graph = AtomGraph.load(graph)

        # retrieve bands
        bands = self.band_structures[idx]
        if isinstance(bands, str):
            bands = BandStructure.load(bands)

        return graph, bands

    def items(self):
        for name, graph, bands in zip(
                self.names, self.graphs, self.band_structures):

            resolved_graph = AtomGraph.load(graph) \
                if isinstance(graph, str) else graph
            resolved_bands = BandStructure.load(bands) \
                if isinstance(bands, str) else bands

            yield name, (resolved_graph, resolved_bands)

    def compile(
            self,
            cache_dir: str,
            backend: str = 'torch',
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            max_k_pts: int | None = None,
            max_energies: int | None = None,
            bands_backend_kwargs: dict = {},
            **backend_kwargs):

        if backend not in BandStructureDataset._backends:
            raise ValueError(f'Unknown Backend Type: {backend}')

        # verify all names are unique:
        self._validate_names()

        return BandStructureDataset._backends[backend].compile(
            dataset=self,
            cache_dir=cache_dir,
            cutoffs=cutoffs,
            masses=masses,
            symbols=symbols,
            max_k_pts=max_k_pts,
            max_energies=max_energies,
            **backend_kwargs,
        )


class BackendBandStructureDataset(ABC, BandStructureDatasetBase):

    def __init__(
            self,
            graphs: list[str],
            band_structures: list[str],
            names: list[str],
            cache_dir: str):

        self.graphs = graphs
        self.band_structures = band_structures
        self.names = names
        self.cache_dir = cache_dir

    @classmethod
    @abstractmethod
    def compile(
            cls,
            dataset: BandStructureDataset,
            graphs: list[str],
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            max_k_pts: int | None = None,
            max_energies: int | None = None,
            bands_backend_kwargs: dict = {},
            **backend_kwargs):
        pass

    @abstractmethod
    def save(self, cache_dir: str):
        pass

    @classmethod
    @abstractmethod
    def load(self, cache_dir: str):
        pass

# import supported BackendBandStructureDatasets
import pyquinn.torch.bandstructure_dataset # noqa