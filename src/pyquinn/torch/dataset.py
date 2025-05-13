from pyquinn.atoms.dataset import AtomGraphDataset, BackendAtomGraphDataset
from pyquinn.atoms.atom_graph import AtomGraph
from pyquinn.torch.atom_graph import TorchAtomGraph

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import os
from glob import glob


@AtomGraphDataset.register_backend('torch')
class TorchAtomGraphDataset(BackendAtomGraphDataset, Dataset):

    @staticmethod
    def _compile_graph(
            graph: AtomGraph | str,
            name: str,
            cache_dir: str,
            **compile_kwargs):

        # load graph (if necessary)
        graph = AtomGraph.load(graph) \
            if isinstance(graph, str) else graph

        # compile graph
        torch_graph = graph.compile(
            backend='torch',
            **compile_kwargs)

        torch_graph_path = os.path.join(
            cache_dir, f'{name}.pt')
        torch_graph.save(torch_graph_path)

        return torch_graph_path

    @staticmethod
    def _save_graph(
            graph: str | TorchAtomGraph,
            name: str,
            cache_dir: str,
            device: torch.device = 'cpu'):

        # load graph (if necessary)
        torch_graph = TorchAtomGraph.load(
            graph, device=device) \
            if isinstance(graph, str) else graph

        # save graph to cache dir:
        torch_graph_path = os.path.join(
            cache_dir, f'{name}.pt')
        torch_graph.save(torch_graph_path)

    @staticmethod
    def compile(
            dataset: AtomGraphDataset,
            cache_dir: str,
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            device: torch.device = 'cpu',
            dtype: torch.dtype = torch.float32,
            idx_dtype: torch.dtype = torch.int64):

        torch_graphs = [
            TorchAtomGraphDataset._compile_graph(
                graph=graph,
                name=name,
                cache_dir=cache_dir,
                cutoffs=cutoffs,
                masses=masses,
                symbols=symbols,
                device=device,
                dtype=dtype,
                idx_dtype=idx_dtype)
            for name, graph in tqdm(
                dataset.items(), total=len(dataset))
        ]

        return TorchAtomGraphDataset(
            graphs=torch_graphs,
            names=dataset.names,
            cache_dir=cache_dir,
            device=device,
        )

    def __init__(
            self,
            graphs: list[str],
            names: list[str],
            cache_dir: str,
            device: torch.device = 'cpu'):

        super().__init__(
            graphs=graphs, 
            names=names,
            cache_dir=cache_dir)

        self.device = device

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = TorchAtomGraph.load(
            filename=self.graphs[index],
            device=self.device)

        return graph

    def items(self):
        for i in range(len(self)):
            yield self.names[i], self[i]

    @staticmethod
    def load(
            cache_dir: str,
            device: torch.device = 'cpu'):

        graphs = glob(os.path.join(cache_dir, '*.pt'))
        names = [
            AtomGraphDataset._parse_graph_name(file)
            for file in graphs
        ]

        return TorchAtomGraphDataset(
            graphs=graphs,
            names=names,
            device=device,
        )

    def save(self, cache_dir: str):

        _dataset_iter = tqdm(zip(
            self.names, self.graphs
        ), total=len(self.names))

        for name, graph in _dataset_iter:
            TorchAtomGraphDataset._save_graph(
                graph=graph,
                name=name,
                cache_dir=cache_dir,
                device=self.device)