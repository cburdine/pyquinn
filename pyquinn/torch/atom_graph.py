from pyquinn.atoms.atom_graph import AtomGraph, BackendAtomGraph
from pyquinn.atoms.geometry import build_dense_neighbor_list
import torch


@AtomGraph.register_backend('torch')
class TorchAtomGraph(BackendAtomGraph):

    @staticmethod
    def compile(
            graph: AtomGraph,
            atomic_numbers: list,
            cutoffs: list,
            masses: list,
            device: torch.device = 'cpu',
            dtype: torch.dtype = torch.float32,
            idx_dtype: torch.dtype = torch.int64):
        
        atomic_numbers = torch.tensor(
            atomic_numbers,
            device=device, dtype=idx_dtype)

        positions = torch.tensor(
            graph.positions,
            device=device, dtype=dtype)
        masses = torch.tensor(
            masses,
            device=device, dtype=dtype)
        unit_cell = torch.tensor(
            graph.unit_cell,
            device=device, dtype=dtype)
        periodic = torch.tensor(
            graph.periodic,
            device=device, dtype=torch.bool)
        features = torch.tensor(
            graph.features,
            device=device, dtype=dtype)
        outputs = torch.tensor(
            graph.outputs,
            device=device, dtype=dtype)
        graph_outputs = torch.tensor(
            graph.graph_outputs,
            device=device, dtype=dtype)
        cutoffs = torch.tensor(
            cutoffs, 
            device=device, dtype=dtype)

        return TorchAtomGraph(
            atomic_numbers=atomic_numbers,
            positions=positions,
            masses=masses,
            unit_cell=unit_cell,
            periodic=periodic,
            features=features,
            outputs=outputs,
            graph_outputs=graph_outputs,
            cutoffs=cutoffs,
            compute_neighbors=True)

    def __init__(self,
                 atomic_numbers: torch.IntTensor,
                 positions: torch.Tensor,
                 masses: torch.Tensor,
                 unit_cell: torch.Tensor,
                 periodic: torch.BoolTensor,
                 features: torch.Tensor,
                 outputs: torch.Tensor,
                 graph_outputs: torch.Tensor,
                 cutoffs: torch.Tensor,
                 compute_neighbors: bool = True):

        super().__init__(
            atomic_numbers=atomic_numbers,
            positions=positions,
            masses=masses,
            unit_cell=unit_cell,
            periodic=periodic,
            features=features,
            outputs=outputs,
            graph_outputs=graph_outputs,
            cutoffs=cutoffs,
            compute_neighbors=compute_neighbors
        )

    def update_neighbors(self):

        neighbors, unit_cell_images = \
            build_dense_neighbor_list(
                positions=self.positions.detach().numpy(),
                unit_cell=self.unit_cell.detach().numpy(),
                periodic=self.periodic.detach().numpy(),
                cutoffs=self.cutoffs.detach().numpy()
            )

        self.neighbors = torch.tensor(
            neighbors, dtype=torch.int64)
        self.unit_cell_images = torch.tensor(
            unit_cell_images, dtype=torch.int64)

    def detach(self):
        atom_graph = TorchAtomGraph(
            atomic_numbers=self.atomic_numbers.detach(),
            positions=self.positions.detach(),
            masses=self.masses.detach(),
            unit_cell=self.unit_cell.detach(),
            periodic=self.periodic.detach(),
            features=self.features.detach(),
            cutoffs=self.cutoffs.detach(),
            compute_neighbors=False,
        )

        atom_graph.neighbors = self.neighbors.detach()
        atom_graph.unit_cell_images = self.unit_cell_images.detach()

        return atom_graph

    def to(self, **kwargs):
        
        atom_graph = TorchAtomGraph(
            atomic_numbers=self.atomic_numbers.to(**kwargs),
            positions=self.positions.to(**kwargs),
            masses=self.masses.to(**kwargs),
            unit_cell=self.unit_cell.to(**kwargs),
            periodic=self.periodic.to(**kwargs),
            features=self.features.to(**kwargs),
            cutoffs=self.cutoffs.to(**kwargs),
            compute_neighbors=False,
        )

        atom_graph.neighbors = self.neighbors.to(**kwargs)
        atom_graph.unit_cell_images = self.unit_cell_images.to(**kwargs)

        return atom_graph

    def extract(self):
        return AtomGraph(
            atoms=self.atomic_numbers.detach().numpy(),
            positions=self.positions.detach().numpy(),
            unit_cell=self.unit_cell.detach().numpy(),
            periodic=self.periodic.detach().numpy(),
            features=self.features.detach().numpy(),
            outputs=self.outputs.detach().numpy(),
            graph_outputs=self.graph_outputs.detach().numpy()
        )
    
    @staticmethod
    def load(
            filename: str, 
            device: torch.device | None = None) -> 'TorchAtomGraph':
        """Helper function to load a TorchAtomGraph from a pickle
        file. For TorchAtomGraphs, the file extension `*.pt` 
        is recommended.

        Args:
            filename (str): The name of the file to load.
            device (torch.device): The device that the AtomGraph
                is loaded to. If None, the previous device on which
                the tensor was allocated is used. Defaults to None.

        Returns:
            TorchAtomGraph: The loaded TorchAtomGraph
        """
        
        graph = torch.load(filename, map_location=device)

        if not isinstance(graph, TorchAtomGraph):
            raise IOError(
                f'Expected an instance of TorchAtomGraph; '
                f'got: {type(graph)}')

        return graph

    def save(self, filename: str):
        """Helper function to save a TorchAtomGraph to a pickle
        file. For TorchAtomGraphs, the file extension `*.pt` 
        is recommended.

        Args:
            filename (str): The name of the file to save.
        """

        torch.save(self, filename)
