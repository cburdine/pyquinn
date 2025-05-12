import unittest
import torch

import numpy as np

from pyquinn.torch.lcao.lcao_hamiltonian import (
    LCAOModel,
    TBModel,
    VariationalLCAOModel,
    VariationalTBModel,
)


class TestLCAOModel(unittest.TestCase):

    def test_lcao_model_init(self):

        for basis_type in LCAOModel.BASIS_OPTIONS:
            for spin in LCAOModel.SPIN_OPTIONS:
                _ = LCAOModel(
                    eigensolver='torch_eigh',
                    spin=spin,
                    basis=basis_type,
                    basis_size=5,
                    potential_basis=basis_type,
                    potential_basis_size=7)

    def test_lcao_model_bases(self):

        basis_size = 5
        potential_basis_size = 7
        orbital_shells = {
            6: [0, 1, 2, 3],
            8: [1, 1]
        }

        max_n_shells = max(len(v) for v in orbital_shells.values())

        model = LCAOModel(
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=basis_size,
            potential_basis='even_tempered',
            potential_basis_size=potential_basis_size,
            orbital_shells=orbital_shells
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 0, -1]
        ])

        neighbors = torch.tensor([
            [1, 2],
            [0, -1],
            [0, -1],
        ])

        alphas, cs, gammas, vs = \
            model.get_basis_parameters(atomic_numbers, positions, neighbors)

        self.assertEqual(tuple(alphas.shape), (3, max_n_shells, basis_size))
        self.assertEqual(tuple(cs.shape), (3, max_n_shells, basis_size))

        self.assertEqual(tuple(gammas.shape), (3, 1, potential_basis_size))
        self.assertEqual(tuple(vs.shape), (3, 1, potential_basis_size))

    def test_lcao_model_hamiltonian_matrix(self):

        model = LCAOModel(
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=3,
            potential_basis='even_tempered',
            potential_basis_size=5
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor([
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, -1.0]
        ], requires_grad=True)

        neighbors = torch.tensor([
            [1, 2],
            [0, -1],
            [0, -1],
        ])

        H_size = sum(
            sum(2*l+1 for l in model.orbital_shell_idxs[z] if l >= 0)
            for z in atomic_numbers
        )

        # compute hamiltonian
        H = model.hamiltonian_matrix(
            atomic_numbers, 
            positions, 
            neighbors, sparse=False)

        # compute sparse hamiltonian
        H_sparse = model.hamiltonian_matrix(
            atomic_numbers, 
            positions, 
            neighbors, sparse=True)

        # ensure H is the correct size and is symmetric
        self.assertEqual(H.shape, (H_size, H_size))
        self.assertEqual(H_sparse.shape, (H_size, H_size))

        self.assertTrue(torch.allclose(H, H.conj().T))
        self.assertTrue(torch.allclose(
            H_sparse.to_dense(), 
            H_sparse.to_dense().conj().T))

        # ensure sparse and non-sparse hamiltonians agree:
        self.assertTrue(torch.allclose(H, H_sparse.to_dense()))

        # test if backward pass is successful
        H.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad),
                        atol=1e-10)
        )

        # test if backward pass is successful
        H_sparse.sum().backward(inputs=positions)

        # ensure sparse gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad),
                        atol=1e-10)
        )

    def test_lcao_model_hamiltonian_key(self):

        model = LCAOModel(
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=3,
            potential_basis='even_tempered',
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]})

        atomic_numbers = torch.tensor([7, 7])
        positions = torch.tensor([
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
        ], requires_grad=True)

        neighbors = torch.tensor([
            [1], [0]
        ])

        H_size = sum(
            sum(2*l+1 for l in model.orbital_shell_idxs[z] if l >= 0)
            for z in atomic_numbers
        )

        H_key = model.orbital_names(atomic_numbers, positions, neighbors)

        # ensure H_key is the correct size:
        self.assertEqual(len(H_key), H_size)

        # ensure H_key is symmetric:
        self.assertListEqual(H_key[:H_size//2], H_key[H_size//2:])

    def test_lcao_model_eigenenergies(self):

        model = LCAOModel(
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=3,
            potential_basis='even_tempered',
            potential_basis_size=5
        )

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor([
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, -1.0]
        ], requires_grad=True)

        neighbors = torch.tensor([
            [1, 2],
            [0, -1],
            [0, -1],
        ])

        E = model.eigenenergies(
            atomic_numbers, positions, neighbors)

        # test if backward pass is successful
        E.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(
                positions_grad, np.zeros_like(positions_grad), atol=1e-10)
        )

    def test_lcao_model_orbital_indices(self):

        shell_order = [3, 2, 1, 0]

        model = LCAOModel(
            eigensolver='torch_eigh',
            spin='paired',
            orbital_shells={1: shell_order}
        )

        atomic_numbers = torch.tensor([1, 1])
        positions = torch.zeros((2, 3))
        neighbors = torch.tensor([[1], [0]])

        target_site_idxs = []
        target_orbital_idxs = []
        for i in range(len(atomic_numbers)):
            for l in shell_order:
                target_site_idxs.extend([i]*(2*l+1))
            n_orbitals = sum(2*l+1 for l in shell_order)
            target_orbital_idxs.extend(list(range(n_orbitals)))

        target_site_idxs = np.array(target_site_idxs)
        target_orbital_idxs = np.array(target_orbital_idxs)

        site_idxs, orbital_idxs = model.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors
        )

        site_idxs = site_idxs.detach().numpy()
        orbital_idxs = orbital_idxs.detach().numpy()

        self.assertTrue(np.all(site_idxs == target_site_idxs))
        self.assertTrue(np.all(orbital_idxs == target_orbital_idxs))

    def test_lcao_model_orbital_quantum_numbers(self):

        shell_order = [3, 2, 1, 0]
        model = LCAOModel(
            eigensolver='torch_eigh',
            spin='paired',
            orbital_shells={1: shell_order}
        )

        atomic_numbers = torch.tensor([1, 1])
        positions = torch.zeros((2, 3))
        neighbors = torch.tensor([[1], [0]])

        target_lms = []
        for i in range(len(atomic_numbers)):
            for l in shell_order:
                target_lms.extend([(l, m) for m in range(-l, l+1)])

        target_lms = np.array(target_lms)

        lms = model.orbital_quantum_numbers(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors
        ).detach().numpy()

        self.assertTrue(np.all(lms == target_lms))



class TestTBModel(unittest.TestCase):

    def test_tb_model_hamiltonian_matrix(self):

        model = TBModel(
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=3,
            potential_basis='even_tempered',
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]})

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor([
            [0.0, 0.0, 0.0], 
            [1.0, 1.0, 0.0],
        ], requires_grad=True)

        unit_cell = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 100, 0.0],
            [0.0, 0.0, 100]
        ], requires_grad=True)

        neighbors = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        unit_cell_images = torch.tensor([
            [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]]
        ])

        k_pts = torch.outer(
            torch.tensor([1.0, 0, 0]), 
            torch.linspace(0, torch.pi/4, 1000)).T

        H_size = sum(
            sum(2*l+1 for l in model.orbital_shell_idxs[z] if l >= 0)
            for z in atomic_numbers
        )

        H_k = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
            sparse=False
        )

        H_k_sparse_raw = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
            sparse=True
        )

        # rectify dimensions of sparse tensor
        H_k_sparse = H_k_sparse_raw.to_dense().permute(2, 0, 1)

        # ensure H is the correct size and is symmetric
        H_shape = tuple(k_pts.shape[:-1]) + (H_size, H_size)
        self.assertEqual(H_k.shape, H_shape)
        self.assertEqual(H_k_sparse.shape, H_shape)

        self.assertTrue(torch.allclose(H_k, H_k.conj().mT, atol=1e-6))
        self.assertTrue(torch.allclose(
            H_k_sparse, H_k_sparse.conj().mT, atol=1e-6))

        # ensure sparse and non-sparse hamiltonians agree:
        self.assertTrue(torch.allclose(H_k, H_k_sparse.to_dense(), atol=1e-6))

        # test if backward pass is successful
        (torch.abs(H_k)**2).sum().backward(inputs=[positions, unit_cell])

        # ensure gradients with respect to positionsand unit cell are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad))
        )
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(
            np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad))
        )

        # test if backward pass is successful
        (torch.abs(H_k_sparse_raw)**2).sum().backward(
            inputs=[positions, unit_cell])

        # ensure sparse gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad))
        )
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(
            np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad))
        )

    def test_tb_model_eigenenergies(self):

        model = TBModel(
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=3,
            potential_basis='even_tempered',
            potential_basis_size=5
        )

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor([
            [0.0, 0.0, 0.0], 
            [1.0, 1.0, 0.0],
        ], requires_grad=True)

        unit_cell = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 100, 0.0],
            [0.0, 0.0, 100]
        ], requires_grad=True)

        neighbors = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        unit_cell_images = torch.tensor([
            [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]]
        ])

        E = model.eigenenergies(
            atomic_numbers, 
            positions, 
            neighbors,
            unit_cell,
            unit_cell_images,
            sparse=False)

        # test if backward pass is successful
        E.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad), atol=1e-10)
        )


class TestVariationalTBModel(unittest.TestCase):

    def test_var_tb_model_hamiltonian_matrix(self):
        
        model = VariationalTBModel(
            onsite_nn='nnconv',
            variational_nn='nnconv',
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=3,
            potential_basis='even_tempered',
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]})

        atomic_numbers = torch.tensor([7, 7])

        positions = torch.tensor([
            [0.0, 0.0, 0.0], 
            [1.0, 1.0, 0.0],
        ], requires_grad=True)

        unit_cell = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 100, 0.0],
            [0.0, 0.0, 100]
        ], requires_grad=True)

        neighbors = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        unit_cell_images = torch.tensor([
            [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]]
        ])

        k_pts = torch.outer(
            torch.tensor([1.0, 0, 0]), 
            torch.linspace(0, torch.pi/4, 1000)).T

        H_size = sum(
            sum(2*l+1 for l in model.orbital_shell_idxs[z] if l >= 0)
            for z in atomic_numbers
        )

        H_k = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
            sparse=False
        )

        H_k_sparse_raw = model.hamiltonian_matrix(
            atomic_numbers,
            positions,
            neighbors,
            unit_cell,
            unit_cell_images,
            k_pts,
            sparse=True
        )

        # rectify dimensions of sparse tensor
        H_k_sparse = H_k_sparse_raw.to_dense().permute(2, 0, 1)

        # ensure H is the correct size and is symmetric
        H_shape = tuple(k_pts.shape[:-1]) + (H_size, H_size)
        self.assertEqual(H_k.shape, H_shape)
        self.assertEqual(H_k_sparse.shape, H_shape)

        self.assertTrue(torch.allclose(H_k, H_k.conj().mT, atol=1e-6))
        self.assertTrue(torch.allclose(
            H_k_sparse, H_k_sparse.conj().mT, atol=1e-6))

        # ensure sparse and non-sparse hamiltonians agree:
        self.assertTrue(torch.allclose(H_k, H_k_sparse.to_dense(), atol=1e-6))

        # test if backward pass is successful
        (torch.abs(H_k)**2).sum().backward(inputs=[positions, unit_cell])

        # ensure gradients with respect to positionsand unit cell are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad),
                        atol=1e-10)
        )
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(
            np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad),
                        atol=1e-10)
        )

        # test if backward pass is successful
        (torch.abs(H_k_sparse_raw)**2).sum().backward(
            inputs=[positions, unit_cell])

        # ensure sparse gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad),
                        atol=1e-10)
        )
        unit_cell_grad = unit_cell.grad.detach().numpy()
        self.assertFalse(
            np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad),
                        atol=1e-10)
        )


class TestVariationalLCAOModel(unittest.TestCase):

    def test_var_lcao_model_hamiltonian_matrix(self):
        
        model = VariationalLCAOModel(
            onsite_nn='nnconv',
            variational_nn='nnconv',
            eigensolver='torch_eigh',
            spin='paired',
            basis='even_tempered',
            basis_size=3,
            potential_basis='even_tempered',
            potential_basis_size=5,
            orbital_shells={7: [0, 1, 2, 3]})

        atomic_numbers = torch.tensor([6, 8, 8])
        positions = torch.tensor([
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, -1.0]
        ], requires_grad=True)

        neighbors = torch.tensor([
            [1, 2],
            [0, -1],
            [0, -1],
        ])

        E = model.eigenenergies(
            atomic_numbers, positions, neighbors)

        # test if backward pass is successful
        E.sum().backward(inputs=positions)

        # ensure gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad),
                        atol=1e-10)
        )
