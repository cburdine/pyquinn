import unittest

import torch

from pyquinn.torch.analysis.negf import TorchNEGFDevice
from pyquinn.torch.types.atom_graph import TorchAtomGraph


class TorchNEGFDeviceTest(unittest.TestCase):

    def _dummy_graph(self) -> TorchAtomGraph:
        # Minimal TorchAtomGraph instance; we avoid neighbor computation and
        # set neighbors/images manually where needed.
        dev = torch.device("cpu")
        atomic_numbers = torch.tensor([1], dtype=torch.int64, device=dev)
        positions = torch.zeros((1, 3), dtype=torch.float32, device=dev)
        masses = torch.ones((1,), dtype=torch.float32, device=dev)
        unit_cell = torch.eye(3, dtype=torch.float32, device=dev)
        periodic = torch.tensor([True, True, True], dtype=torch.bool, device=dev)
        features = torch.zeros((1, 1), dtype=torch.float32, device=dev)
        outputs = torch.zeros((1, 1), dtype=torch.float32, device=dev)
        graph_outputs = torch.zeros((1, 1), dtype=torch.float32, device=dev)
        cutoffs = torch.ones((1,), dtype=torch.float32, device=dev)

        g = TorchAtomGraph(
            atomic_numbers=atomic_numbers,
            positions=positions,
            masses=masses,
            unit_cell=unit_cell,
            periodic=periodic,
            features=features,
            outputs=outputs,
            graph_outputs=graph_outputs,
            cutoffs=cutoffs,
            compute_neighbors=False,
        )
        # Provide a trivial neighbor list.
        g.neighbors = torch.tensor([[0]], dtype=torch.int64)
        g.unit_cell_images = torch.zeros((1, 1, 3), dtype=torch.int64)
        return g

    def test_resolve_k_pts_default(self):
        dev = TorchNEGFDevice(model=None, site=None, boundaries=[])  # type: ignore[arg-type]
        energies = torch.linspace(-1.0, 1.0, 8)
        k_pts = dev._resolve_k_pts(None, energies)
        self.assertEqual(k_pts.shape, (1, 3))
        self.assertTrue(torch.allclose(k_pts, torch.zeros_like(k_pts)))

    def test_boundary_orientation_validation(self):
        g = self._dummy_graph()

        # Orientation with more than one nonzero component should fail.
        dev = TorchNEGFDevice(model=None, site=None, boundaries=[])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dev._validate_boundary_atomgraph(g, torch.tensor([1, 1, 0]))

    def test_boundary_second_neighbor_validation(self):
        g = self._dummy_graph()
        # Put an image offset of 2 along x; orientation [1,0,0] should error.
        g.unit_cell_images = torch.tensor([[[2, 0, 0]]], dtype=torch.int64)

        dev = TorchNEGFDevice(model=None, site=None, boundaries=[])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dev._validate_boundary_atomgraph(g, torch.tensor([1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
