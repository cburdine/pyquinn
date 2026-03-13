from abc import ABC, abstractmethod
import os

from pyquinn.types import (
    AtomGraph, AtomGraphDataset, Operator
)

from pyquinn.types.atom_graph import BackendAtomGraph
from pyquinn.types.operator import BackendOperator

from pyquinn.types.backend import _load_backend_object
from pyquinn.io.stdout import progress_bar

from glob import glob

SUPPORTED_OPERATOR_DATASET_BACKENDS = {
    'torch': 'pyquinn.torch.types.TorchOperatorDataset'
}
""" The supported backend OperatorDataset objects"""

class OperatorDatasetBase:
    """ Base class for OperatorDataset instances"""

class OperatorDataset(OperatorDatasetBase):

    _backends = {}

    @staticmethod
    def register_backend(name: str):

        def decorator(subclass):
            OperatorDataset._backends[name] = subclass
            return subclass
        return decorator

    def __init__(
            self,
            graphs: list[AtomGraph] | list[str] | None = None,
            operators: list[Operator] | list[str] | None = None,
            names: list[str] | None = None):

        if graphs is None:
            graphs = []

        if operators is None:
            operators = []

        self.graphs = []
        self.names = []
        self.operators = []

        self.extend(
            graphs=graphs,
            operators=operators,
            names=names
        )

    def append(
            self,
            graph: AtomGraph | str,
            operator: Operator | str,
            name: str | None = None):
        
        n_graphs = len(self.graphs)
        name = AtomGraphDataset._resolve_graph_name(n_graphs, graph, name)

        self.graphs.append(graph)
        self.operators.append(operator)
        self.names.append(name)

    def extend(
            self,
            graphs: list[AtomGraph] | list[str],
            operators: list[Operator] | list[str],
            names: list[str] | None = None):

        if names is None:
            names = [None]*len(graphs)

        if len(graphs) != len(names):
            raise ValueError(f'Expected len(graphs) == len(names); '
                             f'got: {len(graphs)} != {len(names)}.')
        if len(graphs) != len(operators):
            raise ValueError(f'Expected len(graphs) == len(operators); '
                             f'got: {len(graphs)} != {len(operators)}.')

        # resolve the name of each graph:
        n_graphs = len(self.graphs)
        names = [
            AtomGraphDataset._resolve_graph_name(n_graphs+i, g, gn)
            for i, (g, gn) in enumerate(zip(graphs, names))
        ]

        self.graphs.extend(graphs)
        self.operators.extend(operators)
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

    def __getitem__(self, idx: int) -> tuple[AtomGraph, Operator]:

        # retrieve graph
        graph = self.graphs[idx]
        if isinstance(graph, str):
            graph = AtomGraph.load(graph)

        # retrieve operator
        operator = self.operators[idx]
        if isinstance(operator, str):
            operator = Operator.load(operator)

        return graph, operator

    def items(self):

        for name, graph, operator in zip(
                self.names, self.graphs, self.operators):

            resolved_graph = AtomGraph.load(graph) \
                if isinstance(graph, str) else graph
            resolved_op = Operator.load(operator) \
                if isinstance(operator, str) else operator

            yield name, (resolved_graph, resolved_op)

    def compile(
            self,
            backend: str = 'torch',
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            cache_dir: str | None = None,
            verbosity: int = 1,
            **backend_kwargs):

        if backend not in OperatorDataset._backends:
            if backend in SUPPORTED_OPERATOR_DATASET_BACKENDS:
                path = SUPPORTED_OPERATOR_DATASET_BACKENDS[backend]
                OperatorDataset._backends[backend] = \
                    _load_backend_object(backend, path)
            else:
                raise ValueError(f'Unknown Backend Type: {backend}')

        # verify all names are unique:
        self._validate_names()

        return OperatorDataset._backends[backend].compile(
            dataset=self,
            cutoffs=cutoffs,
            masses=masses,
            symbols=symbols,
            verbosity=verbosity,
            **backend_kwargs
        )

    @staticmethod
    def _save_operator(
        operator: str | Operator,
        name: str,
        dir: str):

        # load operator (if necessary):
        operator = Operator.load(operator) \
            if isinstance(operator, str) else operator
        
        # save operator to cache dir:
        op_path = os.path.join(dir, f'{name}.pkl.op')
        operator.save(op_path)

    def save(self, dir: str):

        for name, (graph, operator) in progress_bar(
                self.items(), 
                total=len(self)):

            AtomGraphDataset._save_graph(
                graph=graph,
                name=name,
                dir=dir)
            OperatorDataset._save_operator(
                operator=operator, 
                name=name, 
                dir=dir)

    @staticmethod
    def load(dir: str, in_memory=False):
        graph_files = glob(os.path.join(dir, '*.pkl'))

        names = sorted(
            AtomGraphDataset._parse_graph_name(file)
            for file in graph_files
        )

        # parse all matching files:
        graphs = []
        operators = []
        for name in names:
            graph_file = os.path.join(dir, f'{name}.pkl')
            op_file = os.path.join(dir, f'{name}.pkl.op')

            if os.path.exists(graph_file) and os.path.exists(op_file):
                graphs.append(
                    AtomGraph.load(graph_file) if in_memory else graph_file)
                operators.append(
                    Operator.load(op_file) if in_memory else op_file)

        return OperatorDataset(
            graphs=graphs,
            operators=operators,
            names=names)


class BackendOperatorDataset(ABC, OperatorDatasetBase):

    def __init__(
        self,
        graphs: list[str] | list[BackendAtomGraph],
        operators: list[str] | list[BackendOperator],
        names: list[str],
        cache_dir: str | None = None):
    
        self.graphs = graphs
        self.operators = operators
        self.names = names
        self.cache_dir = cache_dir

    @staticmethod
    @abstractmethod
    def compile(
            graphs: list[str],
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            max_k_pts: int | None = None,
            max_bands: int | None = None,
            operator_backend_kwargs: dict | None = None,
            verbosity: int = 1,
            **backend_kwargs):
        pass

    @abstractmethod
    def save(self, cache_dir: str):
        pass

    @staticmethod
    @abstractmethod
    def load(cache_dir: str):
        pass


            
