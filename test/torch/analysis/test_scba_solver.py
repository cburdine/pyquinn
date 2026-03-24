import unittest

import torch

from pyquinn.torch.analysis.negf import (
    TorchSCBASolver,
    ElasticDephasingScatterer,
)


class _FakeDevice:
    # Minimal interface needed by TorchSCBASolver.

    def __init__(self, N: int, gamma: float = 0.5):
        self.boundaries = [object()]  # length defines # leads
        self._N = N
        self._gamma = gamma

    def site_self_energies(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            boundary_gfs=None,
            **kwargs):
        # H_site: [K, N, N]
        if k_pts is None:
            K = 1
        else:
            K = k_pts.shape[0]

        N = self._N
        dev = energies.device
        dtype = torch.complex64

        H_site = torch.zeros((K, N, N), dtype=dtype, device=dev)
        H_site[:, torch.arange(N), torch.arange(N)] = torch.linspace(-0.2, 0.2, N, device=dev).to(dtype)

        # One lead self-energy, constant in energy: Σ^R = - i gamma/2 * I
        Sigma_lead = (-0.5j * self._gamma) * torch.eye(N, dtype=dtype, device=dev)
        Sigma_lead = Sigma_lead.view(1, 1, 1, N, N).expand(1, K, energies.numel(), N, N)

        return H_site, Sigma_lead


class TorchSCBASolverTest(unittest.TestCase):

    def test_solver_runs_and_shapes(self):
        device = _FakeDevice(N=2, gamma=0.7)
        solver = TorchSCBASolver(device=device, mixing_alpha=0.2, max_iter=50, tol=1e-8)

        energies = torch.linspace(-1.0, 1.0, 128)

        # Include one scatterer so Σ_scatt is nonzero.
        scatterers = [ElasticDephasingScatterer(D=0.1)]

        G_ret, G_less, G_great = solver.solve(
            energies=energies,
            kt=0.2,
            scatterers=scatterers,
        )

        self.assertEqual(G_ret.shape, (1, 128, 2, 2))
        self.assertEqual(G_less.shape, (1, 128, 2, 2))
        self.assertEqual(G_great.shape, (1, 128, 2, 2))

    def test_legacy_D_matches_scatterer(self):
        device = _FakeDevice(N=2, gamma=0.7)
        energies = torch.linspace(-1.0, 1.0, 96)

        solver1 = TorchSCBASolver(device=device, mixing_alpha=0.2, max_iter=60, tol=1e-8)
        solver2 = TorchSCBASolver(device=device, mixing_alpha=0.2, max_iter=60, tol=1e-8)

        out1 = solver1.solve(energies=energies, kt=0.2, D=0.1)
        out2 = solver2.solve(energies=energies, kt=0.2, scatterers=[ElasticDephasingScatterer(0.1)])

        for a, b in zip(out1, out2):
            self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
