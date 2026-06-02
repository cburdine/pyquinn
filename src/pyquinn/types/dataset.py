import os
from abc import ABC, abstractmethod
from glob import glob

from pyquinn.io.stdout import progress_bar
from pyquinn.types import AtomGraph
from pyquinn.types.backend import _load_backend_object

SUPPORTED_DATASET_BACKENDS = {
    "torch": "pyquinn.torch.types.TorchAtomGraphDataset"
}
""" The supported backend AtomGraphDataset objects"""


class AtomGraphDatasetBase:
    """Base class for AtomGraphDataset instances"""


class AtomGraphDataset(AtomGraphDatasetBase):
    """Represents a dataset of :class:`~pyquinn.types.AtomGraph` objects, each
    containing inputs (`AtomGraph.features`)
    and labels to be predicted (`AtomGraph.outputs`, `AtomGraph.graph_outputs`)

    An AtomGraphDataset contains the following members:

    - `graphs`: The list of AtomGraphs (or strings containing paths to saved
        AtomGraph objects)
    - `names` (optional): The list of respective AtomGraph names. These names
        must be unique.

    Example:
        Here is an example of how an `AtomGraph` object can be constructed programatically
        then compiled to the PyTorch (`torch`) backend.

        .. code-block:: python

            from pyquinn.types import AtomGraph

            graphs = [
                AtomGraph(
                    atoms=[elem, 'Cl'],
                    positions=[
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0]
                    ])
                for elem in ('Li', 'Na', 'K')
            ]

            dataset = AtomGraphDataset(graphs)

            torch_dataset = dataset.compile(backend='torch')

            for torch_graph in torch_dataset:
                print(torch_graph.atomic_numbers)

        .. warning:: By default,
            :meth:`~pyquinn.torch.types.AtomGraphDataset.compile()` compiles all
            AtomGraphs in into BackendAtomGraphs in main memory. To compile the
            dataset to a directory stored on disk and load atomgraphs directly
            from disk you must set the `cache_dir` to a directory on your system.

            .. code-block:: python

                backend_dataset = torch.compile(cache_dir='/path/to/dataset/directory')

        An AtomGraphDataset can also be loaded from a directory of AtomGraph objects
        saved with :meth:`~pyquinn.types.AtomGraph.save`, using the
        :meth:`~pyquinn.types.AtomGraphDataset.load()` function. In this case, the
        names of the AtomGraph files in the directory are stored as the `names` values.

    """

    _backends = {}

    @staticmethod
    def register_backend(name: str):
        """Registers an :class:`~pyquinn.types.dataset.BackendDataset` subclass.

        This should be invoked as a decorator of the subclass. For example:

        .. code-block:: python

            @AtomGraphDataset.register_backend('my_backend_name')
            class MyBackendAtomGraphDataset(BackendAtomGraphDataset):
                ...


        Args:
            name (str): BackendAtomGraphDataset class to be registered.
        """

        def decorator(subclass):
            AtomGraphDataset._backends[name] = subclass
            return subclass

        return decorator

    def __init__(
        self,
        graphs: list[AtomGraph] | list[str] | None = None,
        names: list[str] | None = None,
    ):
        """Creates an AtomGraphDataset from a list of AtomGraphs or saved AtomGraph paths.

        Args:
            graphs (list[AtomGraph] | list[str] | None, optional): A list of AtomGraphs
                or saved AtomGraph paths. Defaults to None.
            names (list[str] | None, optional): A list of unique AtomGraph names.
                Defaults to None.
        """

        self.graphs = []
        self.names = []

        if graphs is not None:
            self.extend(graphs=graphs, names=names)

    def append(self, graph: AtomGraph | str, name: str | None = None):
        """Adds an AtomGraph object or a path to a saved AtomGraph object to this
        dataset.

        Args:
            graph (AtomGraph | str): The AtomGraph to be added. Can be a
                string with the full path to a saved AtomGraph.
            name (str | None, optional): The unique name of this AtomGraph. Defaults to None.
        """

        n_graphs = len(self.graphs)
        name = AtomGraphDataset._resolve_graph_name(n_graphs, graph, name)

        self.graphs.append(graph)
        self.names.append(name)

    def extend(
        self,
        graphs: list[AtomGraph] | list[str],
        names: list[str] | None = None,
    ):
        """Adds a list of AtomGraphs or saved AtomGraph paths to this dataset.

        Args:
            graphs (list[AtomGraph] | list[str]): A list of AtomGraphs
                or saved AtomGraph paths.
            names (list[str] | None, optional): A list of unique AtomGraph names.
                Defaults to None.

        Raises:
            ValueError: _description_
        """

        if names is None:
            names = [None] * len(graphs)

        if len(graphs) != len(names):
            raise ValueError(
                f"Expected len(graphs) == len(names); "
                f"got: {len(graphs)} != {len(names)}."
            )

        # resolve the name of each graph:
        n_graphs = len(self.graphs)
        names = [
            AtomGraphDataset._resolve_graph_name(n_graphs + i, g, gn)
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
        idx: int, graph: AtomGraph, graph_name: str | None = None
    ):

        if graph_name is None:
            if isinstance(graph, str):
                graph_name = AtomGraphDataset._parse_graph_name(graph)
            else:
                graph_name = f"graph_{idx:05d}"

        return graph_name

    def _validate_names(self):
        names_dict = {}
        for i, name in enumerate(self.names):
            if name in names_dict:
                raise ValueError(
                    f"Found more than one occurence of graph with name "
                    f'"{name}" at indices {names_dict[name]} and {i}'
                )
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
            yield AtomGraph.load(graph) if isinstance(graph, str) else graph

    def items(self):
        """Returns an iterator of key-value pairs over this dataset,
        similar to Python's `dict.items()` function. In this case, the
        keys and values are the names and graphs, respectively.

        Yields:
            tuple[str, AtomGraph]: Pairs of names and AtomGraphs.
        """
        for name, graph in zip(self.names, self.graphs):
            resolved_graph = (
                AtomGraph.load(graph) if isinstance(graph, str) else graph
            )
            yield name, resolved_graph

    def compile(
        self,
        backend: str = "torch",
        cutoffs: dict | list[float] | None = None,
        masses: dict | list[float] | None = None,
        symbols: dict | list[str] | None = None,
        cache_dir: str | None = None,
        verbosity: int = 1,
        **backend_kwargs,
    ) -> "BackendAtomGraphDataset":
        """Returns a Compiled form of this AtomGraphDataset that targets a specific backend.
        A compiled AtomGraphDataset can then be manipulated using backend-specific methods.

        The additional arguments passed via `**backend_kwargs` are passed to the
        :meth:`pyquinn.types.AtomGraph.compile` function along with the specified keyword
        arguments.

        .. warning:: By default,
            :meth:`~pyquinn.torch.types.AtomGraphDataset.compile()` compiles all
            AtomGraphs in into BackendAtomGraphs in main memory. To compile the
            dataset to a directory stored on disk and load atomgraphs directly
            from disk you must set the `cache_dir` to a directory on your system.

        Args:
            backend (str, optional): The Backend to target. Defaults to 'torch'.
            cutoffs (dict | list[float] | None, optional): Cutoff radii. Defaults to None.
            masses (dict | list[float] | None, optional): Atomic masses. Defaults to None.
            symbols (dict | list[str] | None, optional): Element substitutions
                made prior to compilation. Defaults to None.
            cache_dir (str | None, optional): Path to a directory used to cache
                the compiled dataset. Defaults to None.

        Raises:
            ValueError: If the backend is unknown, or compilation fails.

        Returns:
            BackendAtomGraphDataset: An AtomGraphDataset compiled for the backend.
        """

        if backend not in AtomGraphDataset._backends:
            if backend in SUPPORTED_DATASET_BACKENDS:
                path = SUPPORTED_DATASET_BACKENDS[backend]
                AtomGraphDataset[backend] = _load_backend_object(backend, path)
            else:
                raise ValueError(f"Unknown Backend Type: {backend}")

        # verify all names are unique:
        self._validate_names()

        # compile graphs:
        return AtomGraphDataset._backends[backend].compile(
            dataset=self,
            cutoffs=cutoffs,
            masses=masses,
            symbols=symbols,
            cache_dir=cache_dir,
            verbosity=verbosity,
            **backend_kwargs,
        )

    @staticmethod
    def _save_graph(graph: str | AtomGraph, name: str, dir: str):

        # load graph (if necessary)
        graph = AtomGraph.load(graph) if isinstance(graph, str) else graph

        # save graph to cache dir:
        graph_path = os.path.join(dir, f"{name}.pkl")
        graph.save(graph_path)

    def save(self, dir: str):

        for name, graph in progress_bar(self.items(), total=len(self)):
            AtomGraphDataset._save_graph(graph=graph, name=name, dir=dir)

    @staticmethod
    def load(dir: str, in_memory: bool = False):

        graph_files = glob(os.path.join(dir, "*.pkl"))
        names = sorted(
            AtomGraphDataset._parse_graph_name(file) for file in graph_files
        )

        # parse all matching files:
        graphs = []
        for name in names:
            graph_file = os.path.join(dir, f"{name}.pkl")

            if os.path.exists(graph_file):
                graphs.append(
                    AtomGraph.load(graph_file) if in_memory else graph_file
                )

        return AtomGraphDataset(graphs=graphs, names=names)


class BackendAtomGraphDataset(ABC, AtomGraphDatasetBase):

    def __init__(
        self, graphs: list[str], names: list[str], cache_dir: str | None = None
    ):

        self.graphs = graphs
        self.names = names
        self.cache_dir = cache_dir

    @staticmethod
    @abstractmethod
    def compile(
        dataset: AtomGraphDataset,
        cutoffs: dict | list[float] | None = None,
        masses: dict | list[float] | None = None,
        symbols: dict | list[str] | None = None,
        cache_dir: str | None = None,
        verbosity: int = 1,
        **backend_kwargs,
    ):
        pass

    @abstractmethod
    def save(self, cache_dir: str):
        pass

    @staticmethod
    @abstractmethod
    def load(cache_dir: str):
        pass


# import supported BackendAtomGraphDatasets
# import pyquinn.torch.types.dataset # noqa
