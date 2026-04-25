import unittest

import torch

from pyquinn.torch.nn import SKHamiltonian, E3SKHamiltonian, PODHamiltonian
from pyquinn.torch.nn.hamiltonian import ExponentialSKCoefficients
from pyquinn.torch.types.operator import build_nonperiodic_matrix


class TestSKHamiltonian(unittest.TestCase):

    def test_sk_hamiltonian_matrix_shapes_and_symmetry(self):
        torch.manual_seed(0)

        model = SKHamiltonian(
            eigensolver='torch_eigh',
            orbital_shells={6: [0, 1]},  # carbon sp
            z_max=6,
        )

        atomic_numbers = torch.tensor([6, 6, 6], dtype=torch.int64)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ],
            dtype=torch.int64,
        )

        H = model.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        # 3 sites x (s + p_x + p_y + p_z)
        self.assertEqual(tuple(H.shape), (12, 12))
        self.assertTrue(torch.allclose(H, H.mT, atol=1e-6))

        onsite, couplings, coupling_values, _images = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )
        H_from_entries = build_nonperiodic_matrix(onsite, couplings, coupling_values)
        self.assertTrue(torch.allclose(H, H_from_entries, atol=1e-6))

    def test_sk_hamiltonian_slater_koster_angular_factors_sp(self):
        model = SKHamiltonian(
            eigensolver='torch_eigh',
            orbital_shells={6: [0, 1]},
            z_max=6,
        )

        # Force constant SK integrals (no distance decay, unit element scaling).
        with torch.no_grad():
            model.sk_coefficients.coeff[:] = torch.tensor([-2.0, 1.5, 0.8, -0.3], dtype=model.sk_coefficients.coeff.dtype)
            model.sk_coefficients.raw_decay.fill_(-20.0)  # softplus ~ 0 => exp(-beta r) ~ 1
            if getattr(model.sk_coefficients, 'element_scale', None) is not None:
                model.sk_coefficients.element_scale.weight.fill_(1.0)

        atomic_numbers = torch.tensor([6, 6], dtype=torch.int64)
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)  # bond along +x
        neighbors = torch.tensor([[1], [0]], dtype=torch.int64)

        H = model.hamiltonian_matrix(atomic_numbers, positions, neighbors)

        # Orbital order per site: [s, p_x, p_y, p_z]
        s0, px0, py0, pz0 = 0, 1, 2, 3
        s1, px1, py1, pz1 = 4, 5, 6, 7

        Vss, Vsp, Vsig, Vpi = -2.0, 1.5, 0.8, -0.3

        self.assertTrue(torch.allclose(H[s0, s1], torch.tensor(Vss, dtype=H.dtype), atol=1e-6))

        # s (site0) -> p (site1)
        self.assertTrue(torch.allclose(H[s0, px1], torch.tensor(Vsp, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[s0, py1], torch.tensor(0.0, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[s0, pz1], torch.tensor(0.0, dtype=H.dtype), atol=1e-6))

        # p (site0) -> s (site1)
        self.assertTrue(torch.allclose(H[px0, s1], torch.tensor(-Vsp, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[py0, s1], torch.tensor(0.0, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[pz0, s1], torch.tensor(0.0, dtype=H.dtype), atol=1e-6))

        # p-p (along x): sigma along x, pi along y/z
        self.assertTrue(torch.allclose(H[px0, px1], torch.tensor(Vsig, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[py0, py1], torch.tensor(Vpi, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[pz0, pz1], torch.tensor(Vpi, dtype=H.dtype), atol=1e-6))

        # Off-diagonal p-p vanish for bond along x.
        self.assertTrue(torch.allclose(H[px0, py1], torch.tensor(0.0, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[px0, pz1], torch.tensor(0.0, dtype=H.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(H[py0, pz1], torch.tensor(0.0, dtype=H.dtype), atol=1e-6))

        # Symmetry check for a couple of entries.
        self.assertTrue(torch.allclose(H, H.mT, atol=1e-6))

    def test_sk_hamiltonian_matrix_k_resolved(self):
        torch.manual_seed(0)

        model = SKHamiltonian(
            eigensolver='torch_eigh',
            orbital_shells={6: [0, 1]},
            z_max=6,
        )

        with torch.no_grad():
            model.sk_coefficients.coeff[:] = torch.tensor([-2.0, 1.5, 0.8, -0.3], dtype=model.sk_coefficients.coeff.dtype)
            model.sk_coefficients.raw_decay.fill_(-20.0)  # softplus ~ 0 => exp(-beta r) ~ 1
            if getattr(model.sk_coefficients, 'element_scale', None) is not None:
                model.sk_coefficients.element_scale.weight.fill_(1.0)

        atomic_numbers = torch.tensor([6, 6], dtype=torch.int64)
        positions = torch.tensor([[0.05, 0.0, 0.0], [0.95, 0.0, 0.0]], dtype=torch.float32)
        neighbors = torch.tensor([[1], [0]], dtype=torch.int64)
        unit_cell = torch.eye(3, dtype=torch.float32)
        unit_cell_images = torch.tensor([[[-1, 0, 0]], [[1, 0, 0]]], dtype=torch.int64)

        k_pts = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float32)

        H_k = model.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
        )

        self.assertEqual(tuple(H_k.shape), (2, 8, 8))
        self.assertTrue(torch.allclose(H_k[0], H_k[0].mT.conj(), atol=1e-6))
        self.assertTrue(torch.allclose(H_k[1], H_k[1].mT.conj(), atol=1e-6))

        onsite, couplings, coupling_values, _images = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )
        H_np = build_nonperiodic_matrix(onsite, couplings, coupling_values).to(dtype=H_k.dtype)
        self.assertTrue(torch.allclose(H_k[0], H_np, atol=1e-6))

        self.assertGreater(float((H_k[1] - H_k[0]).abs().max().item()), 1e-8)

    def test_sk_hamiltonian_can_fit_ssh_chain_carbon_s(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

        # 1D SSH chain: two sites per unit cell, one s orbital per site.
        # Analytical bandstructure:
        #   E(k) = ±|t1 + t2 exp(i 2π k)|
        # with k in fractional coordinates (0..1).
        r1 = 1.0
        r2 = 1.5
        a = r1 + r2

        t1 = -2.0
        t2 = -1.0

        model = SKHamiltonian(
            eigensolver='torch_eigh',
            orbital_shells={6: [0]},
            sk_coefficients=ExponentialSKCoefficients(z_max=6, element_dependent=False),
            z_max=6,
        )

        # Keep onsite energies fixed at 0 so the fit targets only hopping.
        model.onsite_energies.requires_grad_(False)

        k = torch.linspace(0.0, 0.5, 33, dtype=torch.float32).view(-1, 1)
        k_pts = torch.cat((k, torch.zeros_like(k), torch.zeros_like(k)), dim=-1)

        # Target energies (sorted per k): [-E, +E]
        E = torch.sqrt(
            t1 * t1 + t2 * t2 + 2.0 * t1 * t2 * torch.cos(2.0 * torch.pi * k.squeeze(-1))
        ).to(torch.float32)
        target = torch.stack((-E, E), dim=-1)

        atomic_numbers = torch.tensor([6, 6], dtype=torch.int64)
        positions = torch.tensor([[0.0, 0.0, 0.0], [r1, 0.0, 0.0]], dtype=torch.float32)

        # Two neighbors per site to capture intra-cell (t1) and inter-cell (t2).
        neighbors = torch.tensor([[1, 1], [0, 0]], dtype=torch.int64)
        unit_cell = torch.tensor([[a, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float32)
        unit_cell_images = torch.zeros((2, 2, 3), dtype=torch.int64)
        unit_cell_images[0, 1, 0] = -1  # site 0 sees site 1 in previous cell (distance r2)
        unit_cell_images[1, 1, 0] = 1   # site 1 sees site 0 in next cell (distance r2)

        params = [model.sk_coefficients.coeff, model.sk_coefficients.raw_decay]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad(set_to_none=True)
            H_k = model.hamiltonian_matrix(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                k_pts=k_pts,
            )
            pred = torch.linalg.eigvalsh(H_k).to(torch.float32)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            return loss

        opt.step(closure)

        H_k = model.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
        )
        pred = torch.linalg.eigvalsh(H_k).to(torch.float32)

        self.assertLess(float(torch.max(torch.abs(pred - target)).item()), 2e-2)


class TestE3SKHamiltonian(unittest.TestCase):

    def test_e3sk_hamiltonian_matrix_shapes_and_symmetry(self):
        torch.manual_seed(0)

        model = E3SKHamiltonian(
            eigensolver='torch_eigh',
            orbital_shells={6: [0, 1]},
            z_max=6,
            l_max=1,
            r_max=4.0,
            r_points=16,
            edge_hidden=32,
            edge_layers=2,
        )

        atomic_numbers = torch.tensor([6, 6, 6], dtype=torch.int64)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ],
            dtype=torch.int64,
        )

        H = model.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        # 3 sites x (s + p_x + p_y + p_z)
        self.assertEqual(tuple(H.shape), (12, 12))
        self.assertTrue(torch.allclose(H, H.mT, atol=1e-6))

    def test_e3sk_hamiltonian_can_fit_ssh_chain_carbon_s(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

        r1 = 1.0
        r2 = 1.5
        a = r1 + r2

        t1 = -2.0
        t2 = -1.0

        model = E3SKHamiltonian(
            eigensolver='torch_eigh',
            orbital_shells={6: [0]},
            z_max=6,
            l_max=0,
            r_max=4.0,
            r_points=32,
            edge_hidden=64,
            edge_layers=2,
        )

        model.onsite_energies.requires_grad_(False)

        k = torch.linspace(0.0, 0.5, 33, dtype=torch.float32).view(-1, 1)
        k_pts = torch.cat((k, torch.zeros_like(k), torch.zeros_like(k)), dim=-1)

        E = torch.sqrt(
            t1 * t1 + t2 * t2 + 2.0 * t1 * t2 * torch.cos(2.0 * torch.pi * k.squeeze(-1))
        ).to(torch.float32)
        target = torch.stack((-E, E), dim=-1)

        atomic_numbers = torch.tensor([6, 6], dtype=torch.int64)
        positions = torch.tensor([[0.0, 0.0, 0.0], [r1, 0.0, 0.0]], dtype=torch.float32)

        neighbors = torch.tensor([[1, 1], [0, 0]], dtype=torch.int64)
        unit_cell = torch.tensor([[a, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float32)
        unit_cell_images = torch.zeros((2, 2, 3), dtype=torch.int64)
        unit_cell_images[0, 1, 0] = -1
        unit_cell_images[1, 1, 0] = 1

        params = list(model.edge_nns.parameters())
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=250, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad(set_to_none=True)
            H_k = model.hamiltonian_matrix(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                k_pts=k_pts,
            )
            pred = torch.linalg.eigvalsh(H_k).to(torch.float32)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            return loss

        opt.step(closure)

        H_k = model.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
        )
        pred = torch.linalg.eigvalsh(H_k).to(torch.float32)

        self.assertLess(float(torch.max(torch.abs(pred - target)).item()), 2e-2)


class TestPODHamiltonian(unittest.TestCase):

    def test_pod_hamiltonian_reduces_dimension(self):
        torch.manual_seed(0)

        model = PODHamiltonian(
            eigensolver='torch_eigh',
            orbital_shells={6: [0, 1]},
            z_max=6,
            l_max=1,
            bands_per_atom=2,
            r_max=4.0,
            r_points=16,
            edge_hidden=32,
            edge_layers=2,
        )

        atomic_numbers = torch.tensor([6, 6, 6], dtype=torch.int64)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ],
            dtype=torch.int64,
        )

        onsite, couplings, values, images = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        # 3 atoms * 2 bands/atom
        self.assertEqual(int(onsite.shape[0]), 6)
        self.assertEqual(tuple(couplings.shape)[-1], 2)
        self.assertEqual(int(values.shape[0]), int(couplings.shape[0]))
        self.assertEqual(tuple(images.shape), (int(couplings.shape[0]), 3))

        H = build_nonperiodic_matrix(onsite, couplings, values)
        self.assertEqual(tuple(H.shape), (6, 6))
        self.assertTrue(torch.allclose(H, H.mT, atol=1e-5))
