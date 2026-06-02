import unittest

import torch

from pyquinn.torch.analysis.negf import (
    NEGFSolver,
    TorchNEGFDevice,
    greens_function,
)
from pyquinn.torch.models.bandstructure_model import TorchBandStructureModel
from pyquinn.torch.nn.lcao.lcao_hamiltonian import TBHamiltonian
from pyquinn.torch.types.atom_graph import TorchAtomGraph


def _hydrogen_chain_graph(n_atoms: int, a: float, x0: float, periodic_x: bool) -> TorchAtomGraph:
    dev = torch.device("cpu")

    positions = torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev)
    positions[:, 0] = x0 + a * torch.arange(n_atoms, dtype=torch.float32, device=dev)

    unit_cell = torch.diag(torch.tensor([a, 10.0 * a, 10.0 * a], dtype=torch.float32, device=dev))
    periodic = torch.tensor([periodic_x, False, False], dtype=torch.bool, device=dev)

    atomic_numbers = torch.ones((n_atoms,), dtype=torch.int64, device=dev)  # Hydrogen (Z=1)
    masses = torch.ones((n_atoms,), dtype=torch.float32, device=dev)
    features = torch.zeros((n_atoms, 1), dtype=torch.float32, device=dev)
    outputs = torch.zeros((n_atoms, 1), dtype=torch.float32, device=dev)
    graph_outputs = torch.zeros((n_atoms, 1), dtype=torch.float32, device=dev)

    # Choose cutoffs so nearest neighbors at distance a are included
    # (ASE neighborlist uses cutoffs[i]+cutoffs[j]).
    cutoffs = torch.full((n_atoms,), 0.6 * a, dtype=torch.float32, device=dev)

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


class HydrogenChainIntegrationTest(unittest.TestCase):
    def test_device_and_solver_run_on_hydrogen_chain(self):
        # Deterministic-ish default weights.
        torch.manual_seed(0)

        # Simple TB model (not physically calibrated; just plumbing coverage).
        model = TorchBandStructureModel(bands_model=TBHamiltonian())
        model.eval()

        a = 1.0

        # Finite "device" region: 3 atoms.
        site = _hydrogen_chain_graph(n_atoms=3, a=a, x0=0.0, periodic_x=False)

        # Two leads: single-atom unit cells, positioned to couple to the site ends.
        left = _hydrogen_chain_graph(n_atoms=1, a=a, x0=-a, periodic_x=True)
        right = _hydrogen_chain_graph(n_atoms=1, a=a, x0=3 * a, periodic_x=True)

        dev = TorchNEGFDevice(
            device=site,
            leads=[left, right],
            orientations=[torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0])],
        )

        energies = torch.linspace(-2.0, 2.0, 64)
        k_pts = torch.zeros((1, 3), dtype=torch.float32)

        # Exercise device assembly.
        system = dev.system_graph()
        self.assertEqual(system.atomic_numbers.numel(), 3 + 1 + 1)

        solver = NEGFSolver(
            negf_device=dev,
            model=model,
            mixing_alpha=0.2,
            max_iter=10,
            tol=1e-8,
        )

        H_site, Sigma_leads = solver.solve_hamiltonian_self_energies(
            energies=energies,
            k_pts=k_pts,
            eta=1e-3,
        )
        self.assertIsNotNone(Sigma_leads)
        self.assertEqual(H_site.shape[0], 1)  # K
        self.assertEqual(H_site.shape[-1], 3)
        self.assertEqual(Sigma_leads.shape[0], 2)  # two boundaries
        self.assertEqual(Sigma_leads.shape[1], k_pts.shape[0])  # K
        self.assertEqual(Sigma_leads.shape[2], energies.numel())  # E

        if Sigma_leads is not None:
            G_site = greens_function(
                energies=energies,
                hamiltonian_matrix=H_site,
                self_energies=Sigma_leads,
                eta=1e-3,
            )
        else:
            G_site = greens_function(
                energies=energies,
                hamiltonian_matrix=H_site,
                eta=1e-3,
            )
        self.assertEqual(G_site.shape[0], 1)
        self.assertEqual(G_site.shape[1], energies.numel())
        self.assertTrue(torch.isfinite(G_site.real).all())
        self.assertTrue(torch.isfinite(G_site.imag).all())

        # Exercise solver path end-to-end (no scatterers => should converge immediately).
        result = solver.solve(
            energies=energies,
            k_pts=k_pts,
            eta=1e-3,
            kt=0.05,
        )

        self.assertEqual(result.g_ret.shape, (1, energies.numel(), 3, 3))
        self.assertEqual(result.g_lesser.shape, (1, energies.numel(), 3, 3))
        self.assertEqual(result.g_greater.shape, (1, energies.numel(), 3, 3))

        # LDOS / DOS correctness checks
        ldos = result.ldos()
        self.assertEqual(ldos.shape, (energies.numel(), 3))

        dos = result.dos()
        self.assertEqual(dos.shape, (energies.numel(),))

        # Manual reference (single k-point -> Nk = 1).
        g_ret = result._tensor(result.g_ret)
        manual_ldos = (-1 / torch.pi) * torch.diagonal(g_ret.imag, dim1=-2, dim2=-1).sum(-3)
        manual_dos = (-1 / torch.pi) * torch.diagonal(g_ret.imag, dim1=-2, dim2=-1).sum(dim=(-3, -1))

        self.assertTrue(torch.allclose(ldos, manual_ldos, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(dos, manual_dos, atol=1e-6, rtol=1e-5))

        # k-resolved path should match the standard spectral-density formulas
        # for a single k-point.
        ldos_k = result.ldos(k_resolved=True)
        self.assertEqual(ldos_k.shape, (1, energies.numel(), 3))
        self.assertTrue(
            torch.allclose(
                ldos_k,
                (-1 / torch.pi) * torch.diagonal(g_ret.imag, dim1=-2, dim2=-1),
                atol=1e-6,
                rtol=1e-5,
            )
        )

        dos_k = result.dos(k_resolved=True)
        self.assertEqual(dos_k.shape, (1, energies.numel()))
        self.assertTrue(
            torch.allclose(
                dos_k,
                (-1 / torch.pi) * torch.diagonal(g_ret.imag, dim1=-2, dim2=-1).sum(dim=-1),
                atol=1e-6,
                rtol=1e-5,
            )
        )

        # At equilibrium the lead current should vanish.
        I_terminal = result.current(in_idx=0, out_idx=1, k_resolved=False)
        self.assertTrue(torch.isfinite(I_terminal))
        self.assertTrue(torch.allclose(I_terminal, torch.zeros_like(I_terminal), atol=1e-5, rtol=1e-5))

        # Bond-current matrix should be antisymmetric with a (near) zero diagonal.
        J_bond = result.bond_current_density(k_resolved=False)  # [E, N, N]
        self.assertEqual(J_bond.shape, (energies.numel(), 3, 3))
        self.assertTrue(torch.isfinite(J_bond).all())
        self.assertTrue(
            torch.allclose(
                J_bond + J_bond.transpose(-2, -1),
                torch.zeros_like(J_bond),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.diagonal(J_bond, dim1=-2, dim2=-1),
                torch.zeros_like(torch.diagonal(J_bond, dim1=-2, dim2=-1)),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_equilibrium_cut_current_matches_terminal_current(self):
        torch.manual_seed(0)

        model = TorchBandStructureModel(bands_model=TBHamiltonian())
        model.eval()

        a = 1.0
        site = _hydrogen_chain_graph(n_atoms=3, a=a, x0=0.0, periodic_x=False)
        left = _hydrogen_chain_graph(n_atoms=1, a=a, x0=-a, periodic_x=True)
        right = _hydrogen_chain_graph(n_atoms=1, a=a, x0=3 * a, periodic_x=True)

        dev = TorchNEGFDevice(
            device=site,
            leads=[left, right],
            orientations=[torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0])],
        )

        energies = torch.linspace(-2.0, 2.0, 64)

        # use a large redundant k-mesh to check for correct averaging/summation conventions
        k_pts = torch.zeros((10, 3), dtype=torch.float32)

        solver = NEGFSolver(
            negf_device=dev,
            model=model,
            mixing_alpha=0.2,
            max_iter=10,
            tol=1e-8,
        )

        result = solver.solve(
            energies=energies,
            k_pts=k_pts,
            eta=1e-3,
            kt=0.05,
        )

        J_cut = result.cut_current_density(
            in_orbitals=[0],
            out_orbitals=[1],
            k_resolved=False,
        )
        I_cut = torch.trapz(J_cut, energies, dim=-1)
        I_terminal = result.current(in_idx=0, out_idx=1, k_resolved=False)

        self.assertTrue(torch.allclose(I_cut, torch.zeros_like(I_cut), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(I_cut, I_terminal, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
