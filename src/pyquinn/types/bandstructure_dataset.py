from abc import ABC, abstractmethod

from pyquinn.types import (
    AtomGraph, AtomGraphDataset, BandStructure
)
from pyquinn.types.backend import _load_backend_object
from pyquinn.io.stdout import progress_bar

from glob import glob
import os

SUPPORTED_BANDSTRUCTURE_DATASET_BACKENDS = {
    'torch': 'pyquinn.torch.types.TorchBandStructureDataset'
}
""" The supported backend BandStructureDataset objects"""


class BandStructureDatasetBase:
    """ Base class for BandStructureDataset instances"""


class BandStructureDataset(BandStructureDatasetBase):

    _backends = {}

    @staticmethod
    def register_backend(name: str):

        def decorator(subclass):
            BandStructureDataset._backends[name] = subclass
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
        self.band_structures.extend(band_structures)
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
            max_bands: int | None = None,
            bands_backend_kwargs: dict | None = None,
            **backend_kwargs):

        if backend not in BandStructureDataset._backends:
            if backend in SUPPORTED_BANDSTRUCTURE_DATASET_BACKENDS:
                path = SUPPORTED_BANDSTRUCTURE_DATASET_BACKENDS[backend]
                BandStructureDataset._backends[backend] = \
                    _load_backend_object(backend, path)
            else:
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
            max_bands=max_bands,
            **backend_kwargs,
        )

    @staticmethod
    def _save_bands(
            bands: str | BandStructure,
            name: str,
            dir: str):

        # load graph (if necessary)
        bands = BandStructure.load(bands) \
            if isinstance(bands, str) else bands

        # save graph to cache dir:
        bands_path = os.path.join(dir, f'{name}.pkl.bands')
        bands.save(bands_path)

    def save(self, dir: str):

        for name, (graph, bands) in progress_bar(self.items(), total=len(self)):
            AtomGraphDataset._save_graph(graph=graph, name=name, dir=dir)
            BandStructureDataset._save_bands(bands=bands, name=name, dir=dir)

    @staticmethod
    def load(dir: str, in_memory=False):

        graph_files = glob(os.path.join(dir, '*.pkl'))
        
        names = sorted(
            AtomGraphDataset._parse_graph_name(file)
            for file in graph_files
        )

        # parse all matching files:
        graphs = []
        band_structures = []
        for name in names:
            graph_file = os.path.join(dir, f'{name}.pkl')
            band_file = os.path.join(dir, f'{name}.pkl.bands')

            if os.path.exists(graph_file) and os.path.exists(band_file):
                graphs.append(
                    AtomGraph.load(graph_file) if in_memory else graph_file)
                band_structures.append(
                    BandStructure.load(band_file) if in_memory else band_file)

        return BandStructureDataset(
            graphs=graphs,
            band_structures=band_structures,
            names=names)


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

    @staticmethod
    @abstractmethod
    def compile(
            dataset: BandStructureDataset,
            graphs: list[str],
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            max_k_pts: int | None = None,
            max_bands: int | None = None,
            bands_backend_kwargs: dict | None = None,
            **backend_kwargs):
        pass

    @abstractmethod
    def save(self, cache_dir: str):
        pass

    @staticmethod
    @abstractmethod
    def load(cache_dir: str):
        pass