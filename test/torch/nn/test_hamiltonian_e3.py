import unittest

import torch
import e3nn.o3 as o3

from pyquinn.torch.nn import E3Hamiltonian
from pyquinn.torch.types.operator import build_nonperiodic_matrix


def _orbital_irreps_for_graph(model: E3Hamiltonian, atomic_numbers: torch.IntTensor) -> o3.Irreps:
    irreps = o3.Irreps()
    for z in atomic_numbers.tolist():
        l_values = model.orbital_shell_idxs[z]
        for l in l_values.tolist():
            if l < 0:
                continue
            irreps += o3.Irreps([(1, (int(l), (-1) ** int(l)))])
    return irreps


class TestE3Hamiltonian(unittest.TestCase):

    def test_e3_hamiltonian_matrix_shapes_and_symmetry(self):
        model = E3Hamiltonian(
            eigensolver='torch_eigh',
            spin='paired',
            rank=4,
            orbital_shells={
                6: [0, 1],
                8: [0],
            },
        )

        atomic_numbers = torch.tensor([6, 8, 8], dtype=torch.int64)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
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

        # Expected orbital count from shells.
        H_size = 0
        for z in atomic_numbers.tolist():
            for l in model.orbital_shell_idxs[z].tolist():
                if l >= 0:
                    H_size += 2 * int(l) + 1

        self.assertEqual(tuple(H.shape), (H_size, H_size))
        self.assertTrue(torch.allclose(H, H.mT, atol=1e-5))

        onsite, couplings, coupling_values = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )
        H_from_entries = build_nonperiodic_matrix(onsite, couplings, coupling_values)
        self.assertTrue(torch.allclose(H, H_from_entries, atol=1e-6))

    def test_e3_hamiltonian_equivariance(self):
        torch.manual_seed(0)

        model = E3Hamiltonian(
            eigensolver='torch_eigh',
            spin='paired',
            rank=6,
            orbital_shells={
                6: [0, 1],
            },
        )

        atomic_numbers = torch.tensor([6, 6], dtype=torch.int64)
        positions = torch.tensor(
            [[0.1, 0.2, -0.3], [1.0, -0.4, 0.7]],
            dtype=torch.float32,
        )
        neighbors = torch.tensor([[1], [0]], dtype=torch.int64)

        R = o3.rand_matrix().to(dtype=torch.float32)

        H = model.hamiltonian_matrix(atomic_numbers, positions, neighbors)
        H_rot = model.hamiltonian_matrix(atomic_numbers, positions @ R.mT, neighbors)

        irreps = _orbital_irreps_for_graph(model, atomic_numbers)
        D = irreps.D_from_matrix(R).to(dtype=H.dtype)

        H_expected = D @ H @ D.mT

        self.assertTrue(torch.allclose(H_rot, H_expected, atol=2e-3, rtol=2e-3))

    def test_e3_hamiltonian_matrix_k_resolved(self):
        torch.manual_seed(0)

        model = E3Hamiltonian(
            eigensolver='torch_eigh',
            spin='paired',
            rank=4,
            orbital_shells={6: [0]},
        )

        # Two-atom unit cell with a neighbor across the periodic boundary.
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

        self.assertEqual(tuple(H_k.shape), (2, 2, 2))
        self.assertTrue(torch.allclose(H_k[0], H_k[0].mT.conj(), atol=1e-6))
        self.assertTrue(torch.allclose(H_k[1], H_k[1].mT.conj(), atol=1e-6))

        # k=0 should match the non-periodic assembly (phase=1).
        onsite, couplings, coupling_values = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )
        H_np = build_nonperiodic_matrix(onsite, couplings, coupling_values).to(dtype=H_k.dtype)
        self.assertTrue(torch.allclose(H_k[0], H_np, atol=1e-6))

        # Non-zero k should change the Hamiltonian via a Bloch phase (even if tiny at init).
        self.assertGreater(float((H_k[1] - H_k[0]).abs().max().item()), 1e-8)

    def test_e3_hamiltonian_can_fit_ssh_chain_carbon_sp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

        n_sites = 4
        atomic_numbers = torch.full((n_sites,), 6, dtype=torch.int64)
        # Slightly non-collinear 1D ring geometry so the equivariant network can
        # generate non-trivial components across the p_x/p_y/p_z subspace, and
        # we can test periodic images for the boundary bond.
        positions = torch.tensor(
            [
                [0.0, 0.00, 0.00],
                [1.0, 0.15, 0.05],
                [2.0, -0.10, 0.12],
                [3.0, 0.20, -0.08],
            ],
            dtype=torch.float32,
        )
        neighbors = torch.tensor(
            [
                [1, 3],
                [0, 2],
                [1, 3],
                [2, 0],
            ],
            dtype=torch.int64,
        )

        # Periodic boundary in x with length 4.0. Only the boundary neighbor uses an image.
        unit_cell = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=torch.float32,
        )
        unit_cell_images = torch.zeros((n_sites, 2, 3), dtype=torch.int64)
        unit_cell_images[0, 1, 0] = -1  # site 0 sees site 3 in previous cell
        unit_cell_images[3, 1, 0] = 1   # site 3 sees site 0 in next cell

        # Orbital order per site: [s, p_x, p_y, p_z]
        n_orb_per_site = 4
        n_orb = n_sites * n_orb_per_site

        def orbital_idx(site: int, orb: int) -> int:
            return site * n_orb_per_site + orb

        # Equivariant SSH-like tight-binding target (alternating hoppings) for carbon s and p orbitals.
        # Uses simple Slater-Koster-style directional couplings so the target itself is E(3)-equivariant.
        e_s = -1.0
        e_p = 0.5
        t_ss_1, t_ss_2 = -2.0, -1.0
        t_sp_1, t_sp_2 = 1.0, 0.4
        t_pp_sigma_1, t_pp_sigma_2 = 1.5, 0.8
        t_pp_pi_1, t_pp_pi_2 = -0.3, -0.2

        H_target = torch.zeros((n_orb, n_orb), dtype=torch.complex64)
        mask = torch.zeros((n_orb, n_orb), dtype=torch.bool)

        for i in range(n_sites):
            s = orbital_idx(i, 0)
            H_target[s, s] = e_s
            mask[s, s] = True
            for p in (1, 2, 3):
                idx = orbital_idx(i, p)
                H_target[idx, idx] = e_p
                mask[idx, idx] = True

        eye3 = torch.eye(3, dtype=torch.float32)

        for i in range(n_sites):
            j = (i + 1) % n_sites
            parity_even = (i % 2 == 0)

            t_ss = t_ss_1 if parity_even else t_ss_2
            t_sp = t_sp_1 if parity_even else t_sp_2
            t_sigma = t_pp_sigma_1 if parity_even else t_pp_sigma_2
            t_pi = t_pp_pi_1 if parity_even else t_pp_pi_2

            slot = 0 if int(neighbors[i, 0].item()) == j else 1
            image = unit_cell_images[i, slot]
            d = (positions[j] + image.to(dtype=positions.dtype) @ unit_cell) - positions[i]
            u = d / d.norm()
            P = u.view(3, 1) @ u.view(1, 3)
            t_pp = t_pi * eye3 + (t_sigma - t_pi) * P

            s_i = orbital_idx(i, 0)
            s_j = orbital_idx(j, 0)

            # s-s hopping.
            H_target[s_i, s_j] = t_ss
            H_target[s_j, s_i] = t_ss
            mask[s_i, s_j] = True
            mask[s_j, s_i] = True

            # s-p hopping: H[s_i, p_j] ~ +t_sp * u; H[p_i, s_j] ~ -t_sp * u.
            for mu in range(3):
                p_i = orbital_idx(i, 1 + mu)
                p_j = orbital_idx(j, 1 + mu)

                val_sipj = t_sp * float(u[mu].item())
                val_pisj = -t_sp * float(u[mu].item())

                H_target[s_i, p_j] = val_sipj
                H_target[p_j, s_i] = val_sipj
                mask[s_i, p_j] = True
                mask[p_j, s_i] = True

                H_target[p_i, s_j] = val_pisj
                H_target[s_j, p_i] = val_pisj
                mask[p_i, s_j] = True
                mask[s_j, p_i] = True

            # p-p hopping block: 3x3
            for mu in range(3):
                for nu in range(3):
                    p_i = orbital_idx(i, 1 + mu)
                    p_j = orbital_idx(j, 1 + nu)
                    val = float(t_pp[mu, nu].item())
                    H_target[p_i, p_j] = val
                    H_target[p_j, p_i] = val
                    mask[p_i, p_j] = True
                    mask[p_j, p_i] = True

        model = E3Hamiltonian(
            eigensolver='torch_eigh',
            spin='paired',
            rank=16,
            orbital_shells={6: [0, 1]},
            equivariant_nn_kwargs={
                'layers': 2,
                'hidden_features': 2,
                'l_max': 1,
                'r_max': 2.5,
                'r_points': 8,
            },
        )

        opt = torch.optim.Adam(model.parameters(), lr=3e-2)
        model.train()

        with torch.no_grad():
            H_init = model.hamiltonian_matrix(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                k_pts=torch.zeros((1, 3), dtype=positions.dtype, device=positions.device),
            )[0]
            init_mse = (H_init - H_target)[mask].abs().square().mean().item()

        for _step in range(600):
            opt.zero_grad(set_to_none=True)
            H_pred = model.hamiltonian_matrix(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                k_pts=torch.zeros((1, 3), dtype=positions.dtype, device=positions.device),
            )[0]

            err = H_pred - H_target
            loss_main = err[mask].abs().square().mean()
            loss_bg = H_pred[~mask].abs().square().mean()
            loss = loss_main + 0.01 * loss_bg

            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            H_pred = model.hamiltonian_matrix(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                k_pts=torch.zeros((1, 3), dtype=positions.dtype, device=positions.device),
            )[0]
            err = H_pred - H_target
            final_mse = err[mask].abs().square().mean().item()

        self.assertLess(final_mse, init_mse * 0.2)
        self.assertLess(final_mse, 6e-2)
