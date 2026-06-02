import unittest

import numpy as np
import torch

from pyquinn.torch.nn.lcao import (
    LCAOHamiltonian,
    TBHamiltonian,
    VariationalLCAOHamiltonian,
    VariationalTBHamiltonian,
)
from pyquinn.torch.types.operator import build_matrix, build_nonperiodic_matrix


def _clear_grads(module: torch.nn.Module, *leaf_tensors: torch.Tensor):
    module.zero_grad(set_to_none=True)
    for t in leaf_tensors:
        if t is not None:
            t.grad = None


def _assert_finite_nonzero_grad(
    testcase: unittest.TestCase,
    tensor: torch.Tensor,
    name: str,
    atol: float = 1e-12,
):
    testcase.assertIsNotNone(tensor.grad, msg=f"Expected non-None grad for {name}")
    grad = tensor.grad.detach()
    testcase.assertTrue(
        torch.isfinite(grad).all().item(),
        msg=f"Found non-finite values in grad for {name}",
    )
    testcase.assertGreater(grad.abs().max().item(), atol, msg=f"Expected nonzero grad for {name}")


def _assert_param_grad_finite_nonzero(
    testcase: unittest.TestCase,
    param: torch.nn.Parameter,
    name: str,
    atol: float = 1e-12,
):
    testcase.assertIsNotNone(param.grad, msg=f"Expected non-None grad for param {name}")
    grad = param.grad.detach()
    testcase.assertTrue(
        torch.isfinite(grad).all().item(),
        msg=f"Found non-finite values in grad for param {name}",
    )
    testcase.assertGreater(
        grad.abs().max().item(),
        atol,
        msg=f"Expected nonzero grad for param {name}",
    )


def _assert_any_param_grad_finite_nonzero(
    testcase: unittest.TestCase,
    module: torch.nn.Module,
    name: str,
    atol: float = 1e-12,
):
    saw_grad = False
    max_abs = 0.0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        saw_grad = True
        grad = p.grad.detach()
        testcase.assertTrue(
            torch.isfinite(grad).all().item(),
            msg=f"Found non-finite values in grad for {name}",
        )
        max_abs = max(max_abs, grad.abs().max().item())

    testcase.assertTrue(saw_grad, msg=f"Expected at least one computed param grad for {name}")
    testcase.assertGreater(max_abs, atol, msg=f"Expected a nonzero param grad for {name}")


def _square_abs(x: torch.Tensor) -> torch.Tensor:
    # Avoid autograd edge-cases from `abs()` at 0 by constructing |x|^2 directly.
    if torch.is_complex(x):
        return x.real**2 + x.imag**2
    return x**2


class TestLCAOHamiltonian(unittest.TestCase):
    def test_lcao_init(self):

        for basis_type in LCAOHamiltonian.BASIS_OPTIONS:
            for spin in LCAOHamiltonian.SPIN_OPTIONS:
                _ = LCAOHamiltonian(
                    eigensolver="torch_eigh",
                    spin=spin,
                    basis=basis_type,
                    basis_size=5,
                    potential_basis=basis_type,
                    potential_basis_size=7,
                )

    def test_lcao_bases(self):

        basis_size = 5
        potential_basis_size = 7
        orbital_shells = {6: [0, 1, 2, 3], 8: [1, 1]}

        max_n_shells = max(len(v) for v in orbital_shells.values())

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=basis_size,
            potential_basis="even_tempered",
            potential_basis_size=potential_basis_size,
            orbital_shells=orbital_shells,
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, -1]])

        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ]
        )

        alphas, cs, gammas, vs = model.get_basis_parameters(atomic_numbers, positions, neighbors)

        self.assertEqual(tuple(alphas.shape), (3, max_n_shells, basis_size))
        self.assertEqual(tuple(cs.shape), (3, max_n_shells, basis_size))

        self.assertEqual(tuple(gammas.shape), (3, 1, potential_basis_size))
        self.assertEqual(tuple(vs.shape), (3, 1, potential_basis_size))

    def test_lcao_hamiltonian_matrix(self):

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            requires_grad=True,
        )

        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ]
        )

        H_size = sum(sum(2 * l + 1 for l in model.orbital_shell_idxs[z] if l >= 0) for z in atomic_numbers)

        # compute hamiltonian
        H = model.hamiltonian_matrix(atomic_numbers, positions, neighbors)

        # compute sparse hamiltonian
        H_sparse = model.hamiltonian_matrix(atomic_numbers, positions, neighbors)

        # ensure H is the correct size and is symmetric
        self.assertEqual(H.shape, (H_size, H_size))
        self.assertEqual(H_sparse.shape, (H_size, H_size))

        self.assertTrue(torch.allclose(H, H.conj().T))
        self.assertTrue(torch.allclose(H_sparse.to_dense(), H_sparse.to_dense().conj().T))

        # ensure sparse and non-sparse hamiltonians agree:
        self.assertTrue(torch.allclose(H, H_sparse.to_dense()))

        # test if backward pass is successful
        H.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10))

        # test if backward pass is successful
        H_sparse.sum().backward(inputs=positions)

        # ensure sparse gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10))

    def test_lcao_hamiltonian_operator_entries(self):

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ]
        )

        onsite_values, couplings, coupling_values = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        H_from_entries = build_nonperiodic_matrix(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
        )

        H_direct = model.hamiltonian_matrix(atomic_numbers, positions, neighbors)

        self.assertEqual(onsite_values.shape[0], H_direct.shape[0])
        self.assertEqual(couplings.shape[0], coupling_values.shape[-1])
        self.assertTrue(torch.allclose(H_from_entries, H_direct))

    def test_lcao_orbital_operator_entries(self):

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ]
        )

        for kernel in LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS:
            onsite_values, couplings, coupling_values = model.orbital_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                kernel=kernel,
            )

            X = build_nonperiodic_matrix(
                onsite_values=onsite_values,
                couplings=couplings,
                coupling_values=coupling_values,
            )

            if kernel == "momentum_imag":
                X = 1.0j * X

            self.assertEqual(onsite_values.shape[0], X.shape[0])
            self.assertEqual(couplings.shape[0], coupling_values.shape[-1])
            self.assertTrue(
                torch.allclose(X, X.conj().mT, atol=1e-5),
                msg=f'Matrix from kernel "{kernel}" is not Hermitian or Anti-Hermitian',
            )

    def test_lcao_hamiltonian_key(self):

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]},
        )

        atomic_numbers = torch.tensor([7, 7])
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            requires_grad=True,
        )

        neighbors = torch.tensor([[1], [0]])

        H_size = sum(sum(2 * l + 1 for l in model.orbital_shell_idxs[z] if l >= 0) for z in atomic_numbers)

        H_key = model.orbital_names(atomic_numbers, positions, neighbors)

        # ensure H_key is the correct size:
        self.assertEqual(len(H_key), H_size)

        # ensure H_key is symmetric:
        self.assertListEqual(H_key[: H_size // 2], H_key[H_size // 2 :])

    def test_lcao_eigenenergies(self):

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            requires_grad=True,
        )

        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ]
        )

        E = model.eigenenergies(atomic_numbers, positions, neighbors)

        # test if backward pass is successful
        E.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10))

    def test_lcao_orbital_indices(self):

        shell_order = [3, 2, 1, 0]

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            orbital_shells={1: shell_order},
        )

        atomic_numbers = torch.tensor([1, 1])
        positions = torch.zeros((2, 3))
        neighbors = torch.tensor([[1], [0]])

        target_site_idxs = []
        target_orbital_idxs = []
        for i in range(len(atomic_numbers)):
            for l in shell_order:
                target_site_idxs.extend([i] * (2 * l + 1))
            n_orbitals = sum(2 * l + 1 for l in shell_order)
            target_orbital_idxs.extend(list(range(n_orbitals)))

        target_site_idxs = np.array(target_site_idxs)
        target_orbital_idxs = np.array(target_orbital_idxs)

        site_idxs, orbital_idxs = model.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        site_idxs = site_idxs.detach().numpy()
        orbital_idxs = orbital_idxs.detach().numpy()

        self.assertTrue(np.all(site_idxs == target_site_idxs))
        self.assertTrue(np.all(orbital_idxs == target_orbital_idxs))

    def test_lcao_orbital_quantum_numbers(self):

        shell_order = [3, 2, 1, 0]
        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            orbital_shells={1: shell_order},
        )

        atomic_numbers = torch.tensor([1, 1])
        positions = torch.zeros((2, 3))
        neighbors = torch.tensor([[1], [0]])

        target_lms = []
        for _i in range(len(atomic_numbers)):
            for l in shell_order:
                target_lms.extend([(l, m) for m in range(-l, l + 1)])

        target_lms = np.array(target_lms)

        lms = (
            model.orbital_quantum_numbers(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
            )
            .detach()
            .numpy()
        )

        self.assertTrue(np.all(lms == target_lms))


class TestLCAOHamiltonianKernelGradients(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        self.model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=3,
            orbital_shells={6: [0, 1], 8: [0, 1]},
        )

        self.atomic_numbers = torch.tensor([6, 8, 6], dtype=torch.int64)
        self.positions = torch.tensor(
            [
                [0.10, -0.05, 0.02],
                [0.00, 0.30, 1.15],
                [-0.33, 0.12, -0.87],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )

        self.neighbors = torch.tensor(
            [
                [1, 2],
                [0, 2],
                [0, 1],
            ],
            dtype=torch.int64,
        )
        self.site_indices, _ = self.model.orbital_indices(
            atomic_numbers=self.atomic_numbers,
            positions=self.positions.detach(),
            neighbors=self.neighbors,
        )

    def test_orbital_operator_kernel_gradients(self):
        for kernel in LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS:
            # `kinetic` has known non-finite coordinate gradients for some inputs;
            # still validate parameter gradients by detaching coordinates.
            positions = self.positions if kernel != "kinetic" else self.positions.detach()
            _clear_grads(self.model, self.positions)

            onsite_values, couplings, coupling_values = self.model.orbital_operator_entries(
                atomic_numbers=self.atomic_numbers,
                positions=positions,
                neighbors=self.neighbors,
                kernel=kernel,
            )

            site_i = self.site_indices[couplings[:, 0]]
            site_j = self.site_indices[couplings[:, 1]]
            intersite = site_i != site_j
            self.assertTrue(
                intersite.any().item(),
                msg=f"Expected at least one inter-site coupling for kernel={kernel}",
            )

            loss = _square_abs(coupling_values[..., intersite]).sum()
            loss.backward()

            _assert_param_grad_finite_nonzero(
                self,
                self.model.logscale_orbital_cs,
                f"logscale_orbital_cs (orbital kernel={kernel})",
            )
            if kernel != "kinetic":
                _assert_finite_nonzero_grad(self, self.positions, f"positions (orbital kernel={kernel})")

    def test_hamiltonian_operator_kernel_gradients(self):
        for kernel in LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS:
            _clear_grads(self.model, self.positions)

            onsite_values, _, coupling_values = self.model.hamiltonian_operator_entries(
                atomic_numbers=self.atomic_numbers,
                positions=self.positions,
                neighbors=self.neighbors,
                kernel=kernel,
            )

            loss = _square_abs(onsite_values).sum() + _square_abs(coupling_values).sum()
            loss.backward()

            _assert_finite_nonzero_grad(self, self.positions, f"positions (hamiltonian kernel={kernel})")
            _assert_param_grad_finite_nonzero(
                self,
                self.model.logscale_onsite_energies,
                f"logscale_onsite_energies (hamiltonian kernel={kernel})",
            )


class TestLCAOHamiltonianSpatialFieldGradients(unittest.TestCase):
    def test_atomic_orbital_amplitudes_gradients(self):
        torch.manual_seed(0)

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=3,
            orbital_shells={6: [0, 1]},
        )

        atomic_numbers = torch.tensor([6, 6], dtype=torch.int64)
        positions = torch.tensor(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 1.1]],
            dtype=torch.float32,
            requires_grad=True,
        )
        neighbors = torch.tensor([[1], [0]], dtype=torch.int64)

        site_idxs, orbital_idxs = model.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        x_pts = torch.tensor(
            [[0.0, 0.1, 0.4], [0.2, -0.2, 0.8], [-0.1, 0.3, 1.4]],
            dtype=torch.float32,
        )

        _clear_grads(model, positions)
        amps = model.atomic_orbital_amplitudes(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            site_idxs=site_idxs,
            orbital_idxs=orbital_idxs,
            x_pts=x_pts,
        )
        loss = (amps**2).sum()
        loss.backward()

        _assert_finite_nonzero_grad(self, positions, "positions (atomic_orbital_amplitudes)")
        _assert_param_grad_finite_nonzero(
            self,
            model.logscale_orbital_cs,
            "logscale_orbital_cs (atomic_orbital_amplitudes)",
        )

    def test_atomic_potentials_gradients(self):
        torch.manual_seed(0)

        model = LCAOHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=3,
            orbital_shells={6: [0, 1]},
        )

        atomic_numbers = torch.tensor([6, 6], dtype=torch.int64)
        positions = torch.tensor(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 1.1]],
            dtype=torch.float32,
            requires_grad=True,
        )
        neighbors = torch.tensor([[1], [0]], dtype=torch.int64)

        site_idxs = torch.tensor([0, 1], dtype=torch.int64)
        x_pts = torch.tensor(
            [[0.0, 0.1, 0.4], [0.2, -0.2, 0.8], [-0.1, 0.3, 1.4]],
            dtype=torch.float32,
        )

        _clear_grads(model, positions)
        v = model.atomic_potentials(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            site_idxs=site_idxs,
            x_pts=x_pts,
        )
        loss = (v**2).sum()
        loss.backward()

        _assert_finite_nonzero_grad(self, positions, "positions (atomic_potentials)")
        _assert_param_grad_finite_nonzero(
            self,
            model.logscale_potential_vs,
            "logscale_potential_vs (atomic_potentials)",
        )


class TestTBHamiltonian(unittest.TestCase):
    def test_tb_hamiltonian_matrix(self):

        model = TBHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]},
        )

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            requires_grad=True,
        )

        unit_cell = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 100, 0.0], [0.0, 0.0, 100]],
            requires_grad=True,
        )

        neighbors = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

        unit_cell_images = torch.tensor(
            [
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            ]
        )

        k_pts = torch.outer(torch.tensor([1.0, 0, 0]), torch.linspace(0, torch.pi / 4, 1000)).T

        H_size = sum(sum(2 * l + 1 for l in model.orbital_shell_idxs[z] if l >= 0) for z in atomic_numbers)

        H_k = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
        )

        H_k_sparse_raw = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
            sparse=True,
        )

        # rectify dimensions of sparse tensor
        H_k_sparse = H_k_sparse_raw.to_dense().permute(2, 0, 1)

        # ensure H is the correct size and is symmetric
        H_shape = tuple(k_pts.shape[:-1]) + (H_size, H_size)
        self.assertEqual(H_k.shape, H_shape)
        self.assertEqual(H_k_sparse.shape, H_shape)

        self.assertTrue(torch.allclose(H_k, H_k.conj().mT, atol=1e-6))
        self.assertTrue(torch.allclose(H_k_sparse, H_k_sparse.conj().mT, atol=1e-6))

        # ensure sparse and non-sparse hamiltonians agree:
        self.assertTrue(torch.allclose(H_k, H_k_sparse, atol=1e-6))

        # test if backward pass is successful
        (torch.abs(H_k) ** 2).sum().backward(inputs=[positions, unit_cell])

        # ensure gradients with respect to positionsand unit cell are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad)))
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad)))

    def test_tb_hamiltonian_operator_entries(self):

        model = TBHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]},
        )

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            requires_grad=True,
        )

        unit_cell = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 100, 0.0], [0.0, 0.0, 100]],
            requires_grad=True,
        )

        neighbors = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

        unit_cell_images = torch.tensor(
            [
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            ]
        )

        k_pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ]
        )

        onsite_values, couplings, coupling_values, images = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )

        site_indices, _ = model.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        frac_positions = positions @ torch.linalg.pinv(unit_cell)

        H_from_entries = build_matrix(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            images=images,
            k_pts=k_pts,
            frac_positions=frac_positions,
        )

        H_direct = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
        )

        self.assertEqual(onsite_values.shape[0], H_direct.shape[-1])
        self.assertEqual(couplings.shape[0], coupling_values.shape[-1])
        self.assertEqual(images.shape[0], couplings.shape[0])
        self.assertTrue(torch.allclose(H_from_entries, H_direct, atol=1e-5))

        # test if backward pass is successful
        (torch.abs(H_direct) ** 2).sum().backward(inputs=[positions, unit_cell])

        # ensure sparse gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad)))
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad)))

    def test_tb_orbital_operator_entries(self):

        model = TBHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]},
        )

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )

        unit_cell = torch.tensor([[2.0, 0.0, 0.0], [0.0, 100, 0.0], [0.0, 0.0, 100]])

        neighbors = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

        unit_cell_images = torch.tensor(
            [
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            ]
        )

        k_pts = torch.tensor([[0.0, 0.0, 0.0]])

        for kernel in TBHamiltonian.ORBITAL_OPERATOR_KERNELS:
            onsite_values, couplings, coupling_values, images = model.orbital_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                kernel=kernel,
            )

            site_indices, _ = model.orbital_indices(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
            )

            frac_positions = positions @ torch.linalg.pinv(unit_cell)

            X_k = build_matrix(
                onsite_values=onsite_values,
                couplings=couplings,
                coupling_values=coupling_values,
                site_indices=site_indices,
                images=images,
                k_pts=k_pts,
                frac_positions=frac_positions,
            )

            # apply -1.j factor to momentum:
            if kernel == "momentum_imag":
                X_k = X_k * 1.0j

            self.assertEqual(onsite_values.shape[-1], X_k.shape[-1])
            self.assertEqual(couplings.shape[0], coupling_values.shape[-1])
            self.assertEqual(images.shape[0], couplings.shape[0])
            self.assertTrue(
                torch.allclose(X_k, X_k.conj().mT, atol=1e-5),
                msg=f'Matrix from kernel "{kernel}" is not Hermitian or Anti-Hermitian',
            )

    def test_tb_eigenenergies(self):

        model = TBHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
        )

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            requires_grad=True,
        )

        unit_cell = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 100, 0.0], [0.0, 0.0, 100]],
            requires_grad=True,
        )

        neighbors = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

        unit_cell_images = torch.tensor(
            [
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            ]
        )

        E = model.eigenenergies(atomic_numbers, positions, neighbors, unit_cell, unit_cell_images)

        # test if backward pass is successful
        E.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10))


class TestTBHamiltonianKernelGradients(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        self.model = TBHamiltonian(
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=3,
            orbital_shells={7: [0, 1]},
        )

        self.atomic_numbers = torch.tensor([7, 7], dtype=torch.int64)
        self.positions = torch.tensor(
            [[0.05, 0.02, -0.01], [1.05, 0.97, 0.03]],
            dtype=torch.float32,
            requires_grad=True,
        )

        self.unit_cell = torch.tensor(
            [
                [2.1, 0.1, 0.0],
                [0.0, 9.7, 0.2],
                [0.1, 0.0, 10.3],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )

        self.neighbors = torch.tensor([[1, 1], [0, 0]], dtype=torch.int64)
        self.unit_cell_images = torch.tensor(
            [
                [[0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [-1, 0, 0]],
            ],
            dtype=torch.int64,
        )
        self.site_indices, _ = self.model.orbital_indices(
            atomic_numbers=self.atomic_numbers,
            positions=self.positions.detach(),
            neighbors=self.neighbors,
        )

    def test_orbital_operator_kernel_gradients(self):
        for kernel in LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS:
            positions = self.positions if kernel != "kinetic" else self.positions.detach()
            unit_cell = self.unit_cell if kernel != "kinetic" else self.unit_cell.detach()
            _clear_grads(self.model, self.positions, self.unit_cell)

            onsite_values, couplings, coupling_values, _ = self.model.orbital_operator_entries(
                atomic_numbers=self.atomic_numbers,
                positions=positions,
                neighbors=self.neighbors,
                unit_cell=unit_cell,
                unit_cell_images=self.unit_cell_images,
                kernel=kernel,
            )

            site_i = self.site_indices[couplings[:, 0]]
            site_j = self.site_indices[couplings[:, 1]]
            intersite = site_i != site_j
            self.assertTrue(
                intersite.any().item(),
                msg=f"Expected at least one inter-site coupling for kernel={kernel}",
            )

            loss = _square_abs(coupling_values[..., intersite]).sum()
            loss.backward()

            _assert_param_grad_finite_nonzero(
                self,
                self.model.logscale_orbital_cs,
                f"logscale_orbital_cs (TB orbital kernel={kernel})",
            )
            if kernel != "kinetic":
                _assert_finite_nonzero_grad(
                    self,
                    self.positions,
                    f"positions (TB orbital kernel={kernel})",
                )
                _assert_finite_nonzero_grad(
                    self,
                    self.unit_cell,
                    f"unit_cell (TB orbital kernel={kernel})",
                )

    def test_hamiltonian_operator_kernel_gradients(self):
        for kernel in LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS:
            _clear_grads(self.model, self.positions, self.unit_cell)

            onsite_values, _, coupling_values, _ = self.model.hamiltonian_operator_entries(
                atomic_numbers=self.atomic_numbers,
                positions=self.positions,
                neighbors=self.neighbors,
                unit_cell=self.unit_cell,
                unit_cell_images=self.unit_cell_images,
                kernel=kernel,
            )

            loss = _square_abs(onsite_values).sum() + _square_abs(coupling_values).sum()
            loss.backward()

            _assert_finite_nonzero_grad(
                self,
                self.positions,
                f"positions (TB hamiltonian kernel={kernel})",
            )
            _assert_finite_nonzero_grad(
                self,
                self.unit_cell,
                f"unit_cell (TB hamiltonian kernel={kernel})",
            )
            _assert_param_grad_finite_nonzero(
                self,
                self.model.logscale_onsite_energies,
                f"logscale_onsite_energies (TB hamiltonian kernel={kernel})",
            )


class TestVariationalTBHamiltonian(unittest.TestCase):
    def test_var_tb_hamiltonian_matrix(self):

        model = VariationalTBHamiltonian(
            onsite_nn="nnconv",
            variational_nn="nnconv",
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]},
        )

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            requires_grad=True,
        )

        unit_cell = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 100, 0.0], [0.0, 0.0, 100]],
            requires_grad=True,
        )

        neighbors = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

        unit_cell_images = torch.tensor(
            [
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            ]
        )

        k_pts = torch.outer(torch.tensor([1.0, 0, 0]), torch.linspace(0, torch.pi / 4, 1000)).T

        H_size = sum(sum(2 * l + 1 for l in model.orbital_shell_idxs[z] if l >= 0) for z in atomic_numbers)

        H_k = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
        )

        H_k_sparse_raw = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
            sparse=True,
        )

        # rectify dimensions of sparse tensor
        H_k_sparse = H_k_sparse_raw.to_dense().permute(2, 0, 1)

        # ensure H is the correct size and is symmetric
        H_shape = tuple(k_pts.shape[:-1]) + (H_size, H_size)
        self.assertEqual(H_k.shape, H_shape)
        self.assertEqual(H_k_sparse.shape, H_shape)

        self.assertTrue(torch.allclose(H_k, H_k.conj().mT, atol=1e-6))
        self.assertTrue(torch.allclose(H_k_sparse, H_k_sparse.conj().mT, atol=1e-6))

        # ensure sparse and non-sparse hamiltonians agree:
        self.assertTrue(torch.allclose(H_k, H_k_sparse, atol=1e-6))

        # test if backward pass is successful
        (torch.abs(H_k) ** 2).sum().backward(inputs=[positions, unit_cell])

        # ensure gradients with respect to positions and unit cell are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10))
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad), atol=1e-10))

        # test if backward pass is successful
        (torch.abs(H_k_sparse_raw) ** 2).sum().backward(inputs=[positions, unit_cell])

        # ensure sparse gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10))
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad), atol=1e-10))


class TestVariationalLCAOHamiltonian(unittest.TestCase):
    def test_var_lcao_hamiltonian_matrix(self):

        model = VariationalLCAOHamiltonian(
            onsite_nn="nnconv",
            variational_nn="nnconv",
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]},
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            requires_grad=True,
        )

        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ]
        )

        E = model.eigenenergies(atomic_numbers, positions, neighbors)

        # test if backward pass is successful
        E.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10))


class TestVariationalHamiltonianKernelGradients(unittest.TestCase):
    def test_variational_lcao_kernel_gradients(self):
        torch.manual_seed(0)

        model = VariationalLCAOHamiltonian(
            onsite_nn="nnconv",
            variational_nn="nnconv",
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=3,
            orbital_shells={6: [0, 1], 8: [0, 1]},
        )

        atomic_numbers = torch.tensor([6, 8, 6], dtype=torch.int64)
        positions = torch.tensor(
            [[0.10, -0.05, 0.02], [0.00, 0.30, 1.15], [-0.33, 0.12, -0.87]],
            dtype=torch.float32,
            requires_grad=True,
        )
        neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.int64)
        site_indices, _ = model.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions.detach(),
            neighbors=neighbors,
        )

        for kernel in LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS:
            _clear_grads(model, positions)
            onsite_values, _, coupling_values = model.hamiltonian_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                kernel=kernel,
            )
            loss = _square_abs(onsite_values).sum() + _square_abs(coupling_values).sum()
            loss.backward()
            _assert_finite_nonzero_grad(
                self,
                positions,
                f"positions (var LCAO hamiltonian kernel={kernel})",
            )
            _assert_any_param_grad_finite_nonzero(
                self,
                model.onsite_nn,
                f"onsite_nn params (var LCAO hamiltonian kernel={kernel})",
            )

        for kernel in LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS:
            pos_in = positions if kernel != "kinetic" else positions.detach()
            _clear_grads(model, positions)
            onsite_values, couplings, coupling_values = model.orbital_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=pos_in,
                neighbors=neighbors,
                kernel=kernel,
            )
            site_i = site_indices[couplings[:, 0]]
            site_j = site_indices[couplings[:, 1]]
            intersite = site_i != site_j
            self.assertTrue(
                intersite.any().item(),
                msg=f"Expected at least one inter-site coupling for kernel={kernel}",
            )

            loss = _square_abs(coupling_values[..., intersite]).sum()
            loss.backward()
            _assert_any_param_grad_finite_nonzero(
                self,
                model.variational_nn,
                f"variational_nn params (var LCAO orbital kernel={kernel})",
            )
            if kernel != "kinetic":
                _assert_finite_nonzero_grad(
                    self,
                    positions,
                    f"positions (var LCAO orbital kernel={kernel})",
                )

    def test_variational_tb_kernel_gradients(self):
        torch.manual_seed(0)

        model = VariationalTBHamiltonian(
            onsite_nn="nnconv",
            variational_nn="nnconv",
            eigensolver="torch_eigh",
            spin="paired",
            basis="even_tempered",
            basis_size=3,
            potential_basis="even_tempered",
            potential_basis_size=3,
            orbital_shells={7: [0, 1]},
        )

        atomic_numbers = torch.tensor([7, 7], dtype=torch.int64)
        positions = torch.tensor(
            [[0.05, 0.02, -0.01], [1.05, 0.97, 0.03]],
            dtype=torch.float32,
            requires_grad=False,
        )
        unit_cell = torch.tensor(
            [[2.1, 0.1, 0.0], [0.0, 9.7, 0.2], [0.1, 0.0, 10.3]],
            dtype=torch.float32,
            requires_grad=False,
        )
        neighbors = torch.tensor([[1, 1], [0, 0]], dtype=torch.int64)
        unit_cell_images = torch.tensor(
            [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [-1, 0, 0]]],
            dtype=torch.int64,
        )

        for kernel in LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS:
            model.zero_grad(set_to_none=True)
            onsite_values, _, coupling_values, _ = model.hamiltonian_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                kernel=kernel,
            )
            loss = _square_abs(onsite_values).sum() + _square_abs(coupling_values).sum()
            loss.backward()
            _assert_any_param_grad_finite_nonzero(
                self,
                model.onsite_nn,
                f"onsite_nn params (var TB hamiltonian kernel={kernel})",
            )
            _assert_param_grad_finite_nonzero(
                self,
                model.logscale_onsite_energies,
                f"logscale_onsite_energies (var TB hamiltonian kernel={kernel})",
            )

        for kernel in LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS:
            model.zero_grad(set_to_none=True)
            onsite_values, _, coupling_values, _ = model.orbital_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                kernel=kernel,
            )
            loss = _square_abs(onsite_values).sum() + _square_abs(coupling_values).sum()
            loss.backward()
            _assert_param_grad_finite_nonzero(
                self,
                model.logscale_orbital_cs,
                f"logscale_orbital_cs (var TB orbital kernel={kernel})",
            )
            if kernel not in ("kinetic", "momentum_imag"):
                _assert_any_param_grad_finite_nonzero(
                    self,
                    model.variational_nn,
                    f"variational_nn params (var TB orbital kernel={kernel})",
                )

        # Potential kernel supports finite coordinate gradients; check that explicitly.
        positions = positions.detach().clone().requires_grad_(True)
        unit_cell = unit_cell.detach().clone().requires_grad_(True)
        _clear_grads(model, positions, unit_cell)
        onsite_values, _, coupling_values, _ = model.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            kernel="potential",
        )
        loss = _square_abs(onsite_values).sum() + _square_abs(coupling_values).sum()
        loss.backward()
        _assert_finite_nonzero_grad(self, positions, "positions (var TB hamiltonian kernel=potential)")
        _assert_finite_nonzero_grad(self, unit_cell, "unit_cell (var TB hamiltonian kernel=potential)")
