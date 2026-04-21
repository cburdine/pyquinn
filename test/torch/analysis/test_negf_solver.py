import unittest

import torch

from pyquinn.torch.analysis.negf import (
    NEGFSolver,
    ElasticDephasingScatterer,
)


class _FakeGraph:

    def __init__(self, size: int, device: torch.device | None = None, is_system: bool = False):
        self.size = size
        self.device = torch.device('cpu') if device is None else device
        self.is_system = is_system


class _FakeDevice:

    def __init__(self, site_size: int, lead_sizes: list[int], gamma: float = 0.5):
        self.device = _FakeGraph(site_size)
        self.leads = [_FakeGraph(size) for size in lead_sizes]
        self.orientations = [torch.zeros(3, dtype=torch.int64) for _ in lead_sizes]
        self.gamma = gamma

    def system_graph(self) -> _FakeGraph:
        total = self.device.size + sum(lead.size for lead in self.leads)
        return _FakeGraph(size=total, is_system=True)


class _FakeModel:

    def __init__(self, site_size: int, lead_sizes: list[int], hop: float = 0.5):
        self.site_size = site_size
        self.lead_sizes = lead_sizes
        self.hop = hop

    def orbital_indices(self, graph: _FakeGraph) -> tuple[torch.IntTensor, torch.IntTensor]:
        idx = torch.arange(graph.size, dtype=torch.int64, device=graph.device)
        return idx, idx

    def hamiltonian_matrix(self, graph: _FakeGraph, k_pts: torch.Tensor | None) -> torch.Tensor:
        K = 1 if k_pts is None else k_pts.shape[0]
        dtype = torch.complex64
        device = k_pts.device if k_pts is not None else graph.device
        mat = torch.zeros((K, graph.size, graph.size), dtype=dtype, device=device)
        if graph.is_system:
            site = self.site_size
            lead = self.lead_sizes[0]
            mat[:, :site, site:site+lead] = self.hop
            mat[:, site:site+lead, :site] = self.hop
        return mat


def _build_solver(site_size: int = 2, lead_sizes: list[int] | None = None, tol: float = 1e-8) -> NEGFSolver:
    if lead_sizes is None:
        lead_sizes = [2]
    device = _FakeDevice(site_size=site_size, lead_sizes=lead_sizes)
    model = _FakeModel(site_size=site_size, lead_sizes=lead_sizes)
    return NEGFSolver(
        negf_device=device,
        model=model,
        mixing_alpha=0.2,
        max_iter=60,
        tol=tol,
    )


class TorchNEGFSolverTest(unittest.TestCase):

    def test_solver_runs_and_shapes(self):
        solver = _build_solver(tol=1.0)
        energies = torch.linspace(-1.0, 1.0, 128)
        k_pts = torch.zeros((1, 3))
        scatterers = [ElasticDephasingScatterer(D=0.1)]

        result = solver.solve(
            energies=energies,
            k_pts=k_pts,
            kt=0.2,
            scatterers=scatterers,
        )

        self.assertEqual(result.g_ret.shape, (1, 128, 2, 2))
        self.assertEqual(result.g_lesser.shape, (1, 128, 2, 2))
        self.assertEqual(result.g_greater.shape, (1, 128, 2, 2))
        self.assertIsNotNone(result.lead_self_energies)
        self.assertEqual(result.lead_self_energies.shape[0], 1)

    def test_scatterers_contribute_self_energy(self):
        solver = _build_solver(tol=1.0)
        energies = torch.linspace(-1.0, 1.0, 64)
        k_pts = torch.zeros((1, 3))

        no_scatter = solver.solve(energies=energies, k_pts=k_pts, kt=0.2, scatterers=[])
        lesser_se = no_scatter._tensor(no_scatter.lesser_self_energies)
        self.assertTrue(torch.allclose(lesser_se, torch.zeros_like(lesser_se)))

        scatter = solver.solve(
            energies=energies,
            k_pts=k_pts,
            kt=0.2,
            scatterers=[ElasticDephasingScatterer(D=0.2)],
        )
        lesser_se_scatter = scatter._tensor(scatter.lesser_self_energies)
        self.assertFalse(torch.allclose(lesser_se_scatter, torch.zeros_like(lesser_se_scatter)))

    def test_adaptive_mixing_decreases_on_oscillation(self):
        solver = _build_solver(tol=1.0)
        solver.adaptive_mixing = True
        solver.mixing_alpha_min = 1e-3
        solver.mixing_alpha_decay = 0.5

        alpha0 = 0.2
        # Residue decreases then increases -> oscillation trigger.
        history = [1.0, 0.5, 0.7]
        alpha1 = solver._adaptive_mixing_alpha(history, alpha0)
        self.assertLess(alpha1, alpha0)
        self.assertGreaterEqual(alpha1, solver.mixing_alpha_min)

        # No oscillation: monotone decrease -> alpha unchanged.
        history2 = [1.0, 0.8, 0.6]
        alpha2 = solver._adaptive_mixing_alpha(history2, alpha0)
        self.assertEqual(alpha2, alpha0)


if __name__ == "__main__":
    unittest.main()
