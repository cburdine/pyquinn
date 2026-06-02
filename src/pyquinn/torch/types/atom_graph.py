import torch

from pyquinn.types.atom_graph import AtomGraph, BackendAtomGraph
from pyquinn.util.geometry import (
    build_dense_neighbor_list,
    monkhorst_pack_mesh,
    supercell_lattice_points,
)


@AtomGraph.register_backend("torch")
class TorchAtomGraph(BackendAtomGraph):

    @staticmethod
    def compile(
        graph: AtomGraph,
        atomic_numbers: list,
        cutoffs: list,
        masses: list,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        idx_dtype: torch.dtype = torch.int64,
    ):

        atomic_numbers = torch.tensor(
            atomic_numbers, device=device, dtype=idx_dtype
        )

        positions = torch.tensor(graph.positions, device=device, dtype=dtype)
        masses = torch.tensor(masses, device=device, dtype=dtype)
        unit_cell = torch.tensor(graph.unit_cell, device=device, dtype=dtype)
        periodic = torch.tensor(graph.periodic, device=device, dtype=torch.bool)
        features = torch.tensor(graph.features, device=device, dtype=dtype)
        outputs = torch.tensor(graph.outputs, device=device, dtype=dtype)
        graph_outputs = torch.tensor(
            graph.graph_outputs, device=device, dtype=dtype
        )
        cutoffs = torch.tensor(cutoffs, device=device, dtype=dtype)

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
            compute_neighbors=True,
        )

    def __init__(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        masses: torch.Tensor,
        unit_cell: torch.Tensor,
        periodic: torch.BoolTensor,
        features: torch.Tensor,
        outputs: torch.Tensor,
        graph_outputs: torch.Tensor,
        cutoffs: torch.Tensor,
        compute_neighbors: bool = True,
    ):

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
            compute_neighbors=compute_neighbors,
        )

    def update_neighbors(self):

        neighbors, unit_cell_images = build_dense_neighbor_list(
            positions=self.positions.detach().cpu().numpy(),
            unit_cell=self.unit_cell.detach().cpu().numpy(),
            periodic=self.periodic.detach().cpu().numpy(),
            cutoffs=self.cutoffs.detach().cpu().numpy(),
        )

        self.neighbors = torch.tensor(
            neighbors, device=self.positions.device, dtype=torch.int64
        )
        self.unit_cell_images = torch.tensor(
            unit_cell_images, device=self.positions.device, dtype=torch.int64
        )

    def wrap(self):
        images = torch.floor(self.positions @ self.unit_rcell().T)
        self.positions -= images @ self.unit_cell

    def unit_rcell(self, periodicity: bool = True) -> torch.Tensor:

        if periodicity:
            delta = self.periodic.to(torch.float32)

            # Note: the 'gelsd' driver seems to be the only one that
            # returns deterministic results when the unit cell is
            # perturbed by machine epsilon values. This may change
            # with future versions of PyTorch.
            rcell = torch.linalg.lstsq(
                self.unit_cell.T, torch.diag(delta), driver="gelsd"
            )[0]
        else:
            rcell = torch.linalg.pinv(self.unit_cell.T)

        return rcell

    def band_path(
        self, path: str | None = None, n_points: int = 2048
    ) -> torch.Tensor:

        from ase import Atoms

        cell_np = self.unit_cell.detach().cpu().numpy()
        graph_cell = Atoms(cell=cell_np).cell
        frac_k_pts = torch.tensor(
            graph_cell.bandpath(path=path, npoints=n_points).kpts,
            device=self.positions.device,
            dtype=self.positions.dtype,
        )

        return frac_k_pts

    def unit_cell_mesh(self, size: tuple[int] | int = 5) -> torch.Tensor:
        mesh = torch.tensor(
            monkhorst_pack_mesh(size),
            device=self.positions.device,
            dtype=self.positions.dtype,
        )

        return mesh @ self.unit_cell

    def unit_rcell_mesh(self, size: tuple[int] | int = 5) -> torch.Tensor:
        mesh = torch.tensor(
            monkhorst_pack_mesh(size),
            device=self.positions.device,
            dtype=self.positions.dtype,
        )

        return mesh

    def frac_positions(self) -> torch.tensor:
        return self.positions @ torch.linalg.pinv(self.unit_cell)

    def supercell(self, size: torch.IntTensor) -> "BackendAtomGraph":

        size_np = size.detach().cpu().numpy()
        matrix_np, lattice_points_np = supercell_lattice_points(size_np)

        matrix = torch.tensor(
            matrix_np, device=self.unit_cell.device, dtype=self.unit_cell.dtype
        )

        supercell = matrix @ self.unit_cell

        lattice_points = torch.tensor(
            lattice_points_np,
            device=self.unit_cell.device,
            dtype=self.unit_cell.dtype,
        )

        N = lattice_points.shape[0]
        n_atoms = N * self.positions.shape[0]
        _positions = self.positions.view(1, -1, 3)
        _offsets = (lattice_points @ self.unit_cell).view(-1, 1, 3)
        supercell_positions = (_positions + _offsets).reshape(n_atoms, 3)

        return TorchAtomGraph(
            atomic_numbers=self.atomic_numbers.repeat(N),
            positions=supercell_positions,
            masses=self.masses.repeat(N),
            unit_cell=supercell,
            periodic=self.periodic,
            features=self.features.repeat(
                (N,) + (1,) * (self.features.ndim - 1)
            ),
            outputs=self.outputs.repeat((N,) + (1,) * (self.features.ndim - 1)),
            graph_outputs=self.graph_outputs.repeat(
                (N,) + (1,) * (self.features.ndim - 1)
            ),
            cutoffs=self.cutoffs.repeat(N),
        )

    def detach(self):
        atom_graph = TorchAtomGraph(
            atomic_numbers=self.atomic_numbers.detach(),
            positions=self.positions.detach(),
            masses=self.masses.detach(),
            unit_cell=self.unit_cell.detach(),
            periodic=self.periodic.detach(),
            features=self.features.detach(),
            outputs=self.outputs.detach(),
            graph_outputs=self.graph_outputs.detach(),
            cutoffs=self.cutoffs.detach(),
            compute_neighbors=False,
        )

        atom_graph.neighbors = self.neighbors.detach()
        atom_graph.unit_cell_images = self.unit_cell_images.detach()

        return atom_graph

    def copy(self):
        atom_graph = TorchAtomGraph(
            atomic_numbers=self.atomic_numbers.clone(),
            positions=self.positions.clone(),
            masses=self.masses.clone(),
            unit_cell=self.unit_cell.clone(),
            periodic=self.periodic.clone(),
            features=self.features.clone(),
            outputs=self.outputs.clone(),
            graph_outputs=self.graph_outputs.clone(),
            cutoffs=self.cutoffs.clone(),
            compute_neighbors=False,
        )

        atom_graph.neighbors = self.neighbors.clone()
        atom_graph.unit_cell_images = self.unit_cell_images.clone()

        return atom_graph

    def clone(self):
        return self.copy()

    def to(self, *args, **kwargs):

        # short circuit to() function
        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            return self

        atom_graph = TorchAtomGraph(
            atomic_numbers=self.atomic_numbers.to(*args, **kwargs),
            positions=self.positions.to(*args, **kwargs),
            masses=self.masses.to(*args, **kwargs),
            unit_cell=self.unit_cell.to(*args, **kwargs),
            periodic=self.periodic.to(*args, **kwargs),
            features=self.features.to(*args, **kwargs),
            outputs=self.outputs.to(*args, **kwargs),
            graph_outputs=self.graph_outputs.to(*args, **kwargs),
            cutoffs=self.cutoffs.to(*args, **kwargs),
            compute_neighbors=False,
        )

        atom_graph.neighbors = self.neighbors.to(*args, **kwargs)
        atom_graph.unit_cell_images = self.unit_cell_images.to(*args, **kwargs)

        return atom_graph

    def extract(self):
        return AtomGraph(
            atoms=self.atomic_numbers.detach().cpu().numpy(),
            positions=self.positions.detach().cpu().numpy(),
            unit_cell=self.unit_cell.detach().cpu().numpy(),
            periodic=self.periodic.detach().cpu().numpy(),
            features=self.features.detach().cpu().numpy(),
            outputs=self.outputs.detach().cpu().numpy(),
            graph_outputs=self.graph_outputs.detach().cpu().numpy(),
        )

    @staticmethod
    def load(
        filename: str, device: torch.device | None = None
    ) -> "TorchAtomGraph":
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

        graph = torch.load(filename, weights_only=True, map_location=device)

        if not isinstance(graph, TorchAtomGraph):
            raise OSError(
                f"Expected an instance of TorchAtomGraph; "
                f"got: {type(graph)}"
            )

        return graph

    def save(self, filename: str):
        """Helper function to save a TorchAtomGraph to a pickle
        file. For TorchAtomGraphs, the file extension `*.pt`
        is recommended.

        Args:
            filename (str): The name of the file to save.
        """

        torch.save(self, filename)


# add TorchAtomGraph as safe to serialize
torch.serialization.add_safe_globals([TorchAtomGraph])
