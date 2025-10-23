from pyquinn.types import Operator, AtomGraphDataset
from pyquinn.types.operator_dataset import (
    OperatorDataset,
    BackendOperatorDataset
)
from pyquinn.torch.types import (
    TorchAtomGraphDataset,
    TorchOperator,
    TorchAtomGraph
)
from pyquinn.io.stdout import progress_bar

import os
from glob import glob

import torch

@OperatorDataset.register_backend('torch')
class TorchOperatorDataset(
        BackendOperatorDataset, 
        torch.utils.data.Dataset):

    @staticmethod
    def _compile_operator(
            operator: Operator | str,
            name: str | None = None,
            cache_dir: str | None = None,
            **compile_kwargs) -> TorchOperator | str:

        # load operator (if necessary)
        operator = Operator.load(operator) \
            if isinstance(operator, str) else operator
            
        # compile operator
        torch_operator = operator.compile(
            backend='torch',
            **compile_kwargs)

        if cache_dir is not None:
            torch_operator_path = os.path.join(
                cache_dir, f'{name}.pt.bands')
            torch_operator.save(torch_operator_path)
            torch_operator = torch_operator_path

        return torch_operator

    @staticmethod
    def _save_operator(
        operator: TorchOperator | str,
        name: str,
        cache_dir: str,
        device: torch.device = 'cpu'):


        # load operator (if necessary)
        operator = TorchOperator.load(operator, device=device) \
            if isinstance(operator, str) else operator

        # save operator to cache dir
        torch_operator_path = os.path.join(
            cache_dir, f'{name}.pt.bands')
        operator.save(torch_operator_path)


    @staticmethod
    def compile(
        dataset: OperatorDataset,
        cache_dir: str | None = None,
        cutoffs: dict | list[float] | None = None,
        masses: dict | list[float] | None = None,
        symbols: dict | list[str] | None = None,
        max_k_pts: int | None = None,
        max_bands: int | None = None,
        device: torch.device = 'cpu',
        dtype: torch.dtype = torch.float32,
        idx_dtype: torch.dtype = torch.int64):
    
        torch_graphs = []
        torch_operators = []
        names = []

        for name, (graph, operator) in progress_bar(dataset.items(), total=len(dataset)):

            # compile and add graph
            torch_graphs.append(
                TorchAtomGraphDataset._compile_graph(
                    graph=graph,
                    name=name,
                    cache_dir=cache_dir,
                    cutoffs=cutoffs,
                    masses=masses,
                    symbols=symbols,
                    device=device,
                    dtype=dtype,
                    idx_dtype=idx_dtype))
            
            # compile and add operator
            torch_operators.append(
                TorchOperatorDataset._compile_operator(
                    operator=operator,
                    name=name,
                    cache_dir=cache_dir,
                    device=device,
                    dtype=dtype))

            names.append(name)

        return TorchOperatorDataset(
            graphs=torch_graphs,
            operators=torch_operators,
            names=names,
            cache_dir=cache_dir,
            device=device
        )

    def __init__(
            self,
            graphs: list[str] | list[TorchAtomGraph],
            operators: list[str] | list[TorchOperator],
            names: list[str],
            cache_dir: str,
            device: torch.device = 'cpu'):

        super().__init__(
            graphs=graphs,
            operators=operators,
            names=names,
            cache_dir=cache_dir,
        )

        self.device = device

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):

        graph = self.graphs[index]
        operator = self.operators[index]

        if isinstance(graph, str):
            graph = TorchAtomGraph.load(
                filename=graph,
                device=self.device)

        if isinstance(operator, str):
            operator = TorchOperator.load(
                filename=operator,
                device=self.device)

        return graph, operator

    @staticmethod
    def load(
            cache_dir: str,
            in_memory: bool = False,
            device: torch.device = 'cpu'):

        graphs = glob(os.path.join(cache_dir, '*.pt'))
        operators = glob(ps.path.join(cache_dir, '*.pt.op'))
        names = [
            AtomGraphDataset._parse_graph_name(file)
            for file in graphs
        ]

        if in_memory:
            graphs = [
                TorchAtomGraph.load(filename=file, device=device)
                for file in graphs
            ]
            operators = [
                TorchOperator.load(filename=file, device=device)
                for file in operators
            ]

        return TorchOperatorDataset(
            graphs=graphs,
            operators=operators,
            names=names,
            cache_dir=cache_dir,
            device=device
        )

    def save(self, cache_dir: str):

        _dataset_iter = progress_bar(zip(
            self.names, self.graphs, self.operators
        ), total=len(self.names))

        for name, graph, operator in _dataset_iter:
            TorchAtomGraphDataset._save_graph(
                graph=graph,
                name=name,
                cache_dir=cache_dir,
                device=self.device)

            TorchOperatorDataset._save_operator(
                operator=operator,
                name=name,
                cache_dir=cache_dir,
                device=self.device)



        
        