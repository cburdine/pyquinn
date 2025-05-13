from abc import ABC, abstractmethod
import os

from pyquinn.types import AtomGraph
from pyquinn.types.backend import _load_backend_object

from tqdm import tqdm
from glob import glob

SUPPORTED_DATASET_BACKENDS = {
    'torch': 'pyquinn.torch.types.TorchAtomGraphDataset'
}
""" The supported backend AtomGraphDataset objects"""


class AtomGraphDatasetBase:
    """ Base class for AtomGraphDataset instances"""        


class AtomGraphDataset(AtomGraphDatasetBase):

    _backends = {}

    @staticmethod
    def register_backend(name: str):

        def decorator(subclass):
            AtomGraphDataset._backends[name] = subclass
            return subclass
        return decorator

    def __init__(
            self,
            graphs: list[AtomGraph] | list[str] = [],
            names: list[str] | None = None):

        self.graphs = []
        self.names = []

        self.extend(graphs=graphs, names=names)

    def append(self, graph: AtomGraph | str, name: str | None = None):

        n_graphs = len(self.graphs)
        name = AtomGraphDataset._resolve_graph_name(n_graphs, graph, name)

        self.graphs.append(graph)
        self.names.append(name)

    def extend(
            self, 
            graphs: list[AtomGraph] | list[str],
            names: list[str] | None = None):

        if names is None:
            names = [None]*len(graphs)

        if len(graphs) != len(names):
            raise ValueError(f'Expected len(graphs) == len(names); '
                             f'got: {len(graphs)} != {len(names)}.')

        # resolve the name of each graph:
        n_graphs = len(self.graphs)
        names = [
            AtomGraphDataset._resolve_graph_name(n_graphs+i, g, gn)
            for i, (g, gn) in enumerate(zip(graphs, names))
        ]

        self.graphs.extend(graphs)
        self.names.extend(names)

    @staticmethod
    def _parse_graph_name(graph_filepath) -> str:
        base_file = os.path.split(graph_filepath)[-1]
        name = os.path.splitext(base_file)[0]
        return name

    @staticmethod
    def _resolve_graph_name(
            idx: int, 
            graph: AtomGraph, 
            graph_name: str | None = None):

        if graph_name is None:
            if isinstance(graph, str):
                graph_name = AtomGraphDataset._parse_graph_name(graph)
            else:
                graph_name = f'graph_{idx:05d}'

        return graph_name

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

    def __getitem__(self, idx: int) -> AtomGraph:

        # retrieve graph
        graph = self.graphs[idx]
        if isinstance(graph, str):
            graph = AtomGraph.load(graph)

        return graph

    def __iter__(self):
        for graph in self.graphs:
            yield AtomGraph.load(graph) \
                if isinstance(graph, str) else graph

    def items(self):
        for name, graph in zip(self.names, self.graphs):
            resolved_graph = AtomGraph.load(graph) \
                if isinstance(graph, str) else graph
            yield name, resolved_graph

    def compile(
            self,
            cache_dir: str,
            backend: str = 'torch',
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            **backend_kwargs) -> 'BackendAtomGraphDataset':

        if backend not in AtomGraphDataset._backends:
            if backend in SUPPORTED_DATASET_BACKENDS:
                path = SUPPORTED_DATASET_BACKENDS[backend]
                AtomGraphDataset[backend] = _load_backend_object(backend, path)
            else:
                raise ValueError(f'Unknown Backend Type: {backend}')

        # verify all names are unique:
        self._validate_names()

        # compile graphs:
        return AtomGraphDataset._backends[backend].compile(
            dataset=self,
            cache_dir=cache_dir,
            cutoffs=cutoffs,
            masses=masses,
            symbols=symbols,
            **backend_kwargs
        )

    @staticmethod
    def _save_graph(
            graph: str | AtomGraph,
            name: str,
            dir: str):

        # load graph (if necessary)
        graph = AtomGraph.load(graph) \
            if isinstance(graph, str) else graph

        # save graph to cache dir:
        graph_path = os.path.join(dir, f'{name}.pkl')
        graph.save(graph_path)

    def save(self, dir: str):

        for name, graph in tqdm(self.items(), total=len(self)):
            AtomGraphDataset._save_graph(graph=graph, name=name, dir=dir)

    @staticmethod
    def load(dir: str):

        graphs = glob(os.path.join(dir, '*.pkl'))
        names = [
            AtomGraphDataset._parse_graph_name(file)
            for file in graphs
        ]

        return AtomGraphDataset(
            graphs=graphs,
            names=names)


class BackendAtomGraphDataset(ABC, AtomGraphDatasetBase):

    def __init__(
            self,
            graphs: list[str],
            names: list[str],
            cache_dir: str | None = None):

        self.graphs = graphs
        self.names = names
        self.cache_dir = cache_dir

    @staticmethod
    @abstractmethod
    def compile(
            dataset: AtomGraphDataset,
            cache_dir: str,
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            **backend_kwargs):
        pass

    @abstractmethod
    def save(self, cache_dir: str):
        pass

    @staticmethod
    @abstractmethod
    def load(cache_dir: str):
        pass

# import supported BackendAtomGraphDatasets
#import pyquinn.torch.types.dataset # noqa