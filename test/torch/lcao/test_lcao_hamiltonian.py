import unittest
import torch

import numpy as np

from pyquinn.torch.lcao.lcao_hamiltonian import LCAOModel

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
            np.allclose(positions_grad, np.zeros_like(positions_grad))
        )

        # test if backward pass is successful
        H_sparse.sum().backward(inputs=positions)

        # ensure sparse gradients with respect to positions are nonzero
        positions_grad = positions.grad.detach().numpy()
        self.assertFalse(
            np.allclose(positions_grad, np.zeros_like(positions_grad))
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

        H_key = model.hamiltonian_key(atomic_numbers, positions, neighbors)

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
            np.allclose(positions_grad, np.zeros_like(positions_grad))
        )