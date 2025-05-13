import unittest
import numpy as np

from pyquinn.torch.nn.lcao import VariationalE3TBModel
from pyquinn.torch.nn.lcao.lcao_hamiltonian_e3 import _make_shell_irrep_map


class TestVariationalTBModel(unittest.TestCase):

    def test_make_shell_irrep_map(self):

        shell_idxs = \
            VariationalE3TBModel._make_shell_idxs({
                1: [0, 0, 1, 2, 2, 2],
                2: [2, 1, 0],
                3: [0, 1, 2, 3],
            })

        irrep_shells, shell_map, irrep_map = _make_shell_irrep_map(shell_idxs)

        irrep_shells = irrep_shells.detach().numpy()
        shell_map = shell_map.detach().numpy()
        irrep_map = irrep_map.detach().numpy()

        target_irrep_shells = np.array([0, 0, 1, 2, 2, 2, 3])
        blocks = {
            0: [0], 
            1: [1], 
            2: [2, 3, 4],
            3: [5, 6, 7, 8, 9],
            4: [10, 11, 12, 13, 14],
            5: [15, 16, 17, 18, 19],
            6: [20, 21, 22, 23, 24, 25, 26]
        }

        target_irrep_map = np.array([
            [-1]*20,
            blocks[0]+blocks[1]+blocks[2]+blocks[3]+blocks[4]+blocks[5], # noqa
            blocks[3]+blocks[2]+blocks[0]+[-1]*11,
            blocks[0]+blocks[2]+blocks[3]+blocks[6]+[-1]*4
        ])

        target_shell_map = np.array([
            [-1]*20,
            [0]+[1]+[2]*3+[3]*5+[4]*5+[5]*5,
            [0]*5+[1]*3+[2]+[-1]*11,
            [0]+[1]*3+[2]*5+[3]*7 + [-1]*4
        ])

        self.assertTrue(np.all(irrep_shells == target_irrep_shells))
        self.assertTrue(np.all(irrep_map == target_irrep_map))
        self.assertTrue(np.all(shell_map == target_shell_map))
        
    # def test_var_tb_model_hamiltonian_matrix(self):
        
    #     model = VariationalE3TBModel(
    #         onsite_nn='nnconv',
    #         variational_nn='nnconv',
    #         eigensolver='torch_eigh',
    #         spin='paired',
    #         basis='even_tempered',
    #         basis_size=3,
    #         potential_basis='even_tempered',
    #         potential_basis_size=5,
    #         orbital_shells={7: [0, 1, 2, 3]})

    #     atomic_numbers = torch.tensor([7, 7])

    #     positions = torch.tensor([
    #         [0.0, 0.0, 0.0], 
    #         [1.0, 1.0, 0.0],
    #     ], requires_grad=True)

    #     unit_cell = torch.tensor([
    #         [2.0, 0.0, 0.0],
    #         [0.0, 100, 0.0],
    #         [0.0, 0.0, 100]
    #     ], requires_grad=True)

    #     neighbors = torch.tensor([
    #         [1, 1, 0, 0],
    #         [0, 0, 1, 1]
    #     ])

    #     unit_cell_images = torch.tensor([
    #         [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
    #         [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]]
    #     ])

    #     k_pts = torch.outer(
    #         torch.tensor([1.0, 0, 0]), 
    #         torch.linspace(0, torch.pi/4, 1000)).T

    #     H_size = sum(
    #         sum(2*l+1 for l in model.orbital_shell_idxs[z] if l >= 0)
    #         for z in atomic_numbers
    #     )

    #     H_k = model.hamiltonian_matrix(
    #         atomic_numbers,
    #         positions,
    #         neighbors,
    #         unit_cell,
    #         unit_cell_images,
    #         k_pts,
    #         sparse=False
    #     )

    #     H_k_sparse_raw = model.hamiltonian_matrix(
    #         atomic_numbers,
    #         positions,
    #         neighbors,
    #         unit_cell,
    #         unit_cell_images,
    #         k_pts,
    #         sparse=True
    #     )

    #     # rectify dimensions of sparse tensor
    #     H_k_sparse = H_k_sparse_raw.to_dense().permute(2, 0, 1)

    #     # ensure H is the correct size and is symmetric
    #     H_shape = tuple(k_pts.shape[:-1]) + (H_size, H_size)
    #     self.assertEqual(H_k.shape, H_shape)
    #     self.assertEqual(H_k_sparse.shape, H_shape)

    #     self.assertTrue(torch.allclose(H_k, H_k.conj().mT, atol=1e-6))
    #     self.assertTrue(torch.allclose(
    #         H_k_sparse, H_k_sparse.conj().mT, atol=1e-6))

    #     # ensure sparse and non-sparse hamiltonians agree:
    #     self.assertTrue(torch.allclose(H_k, H_k_sparse.to_dense(), atol=1e-6))

    #     # test if backward pass is successful
    #     (torch.abs(H_k)**2).sum().backward(inputs=[positions, unit_cell])

    #     # ensure gradients with respect to positionsand unit cell are nonzero
    #     positions_grad = positions.grad.detach().numpy()
    #     self.assertFalse(
    #         np.allclose(positions_grad, np.zeros_like(positions_grad))
    #     )
    #     unit_cell_grad = unit_cell.grad.detach().numpy()
    #     self.assertFalse(
    #         np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad))
    #     )

    #     # test if backward pass is successful
    #     (torch.abs(H_k_sparse_raw)**2).sum().backward(
    #         inputs=[positions, unit_cell])

    #     # ensure sparse gradients with respect to positions are nonzero
    #     positions_grad = positions.grad.detach().numpy()
    #     self.assertFalse(
    #         np.allclose(positions_grad, np.zeros_like(positions_grad))
    #     )
    #     unit_cell_grad = unit_cell.grad.detach().numpy()
    #     self.assertFalse(
    #         np.allclose(unit_cell_grad, np.zeros_like(unit_cell_grad))
    #     )