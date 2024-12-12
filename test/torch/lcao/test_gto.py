import unittest

import torch
import numpy as np
from pyquinn.torch.lcao.gto import (
    _make_gto_cho_tensor, 
    gaussian_3d, l1_gaussian_3d, l2_gaussian_3d, 
    gto, sto_ng, potential_gto, potential_ng, sto_ng_norm_coeff,
    sto_ng_overlap_matrix,
    sto_ng_block_primitive_overlap_matrix,
    sto_ng_block_overlap_matrix, 
    sto_ng_potential_overlap_matrix,
    sto_ng_block_potential_overlap_matrix,
    cho_ng, cho_ng_norm_coeff,
    cho_ng_overlap_matrix,
    cho_ng_potential_overlap_matrix,
    cho_ng_block_overlap_matrix,
    cho_ng_block_potential_overlap_matrix
)


class GTOTest(unittest.TestCase):

    def test__make_gto_cho_tensor(self):

        test_ijks = {
            0: [(0, 0, 0)],
            1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            2: [(2, 0, 0), (1, 1, 0), (1, 0, 1),
                (0, 2, 0), (0, 1, 1), (0, 0, 2)],

            3: [(3, 0, 0), (2, 1, 0), (2, 0, 1),
                (1, 2, 0), (1, 1, 1), (1, 0, 2),
                (0, 3, 0), (0, 2, 1), (0, 1, 2), 
                (0, 0, 3)],
            4: [(4, 0, 0), (3, 1, 0), (3, 0, 1), 
                (2, 2, 0), (2, 1, 1), (2, 0, 2), 
                (1, 3, 0), (1, 2, 1), (1, 1, 2), 
                (1, 0, 3), (0, 4, 0), (0, 3, 1), 
                (0, 2, 2), (0, 1, 3), (0, 0, 4)],

            5: [(5, 0, 0), (4, 1, 0), (4, 0, 1),
                (3, 2, 0), (3, 1, 1), (3, 0, 2),
                (2, 3, 0), (2, 2, 1), (2, 1, 2), 
                (2, 0, 3), (1, 4, 0), (1, 3, 1), 
                (1, 2, 2), (1, 1, 3), (1, 0, 4), 
                (0, 5, 0), (0, 4, 1), (0, 3, 2), 
                (0, 2, 3), (0, 1, 4), (0, 0, 5)],
        }

        test_projection_matrices = {
            0: np.array([
                [1.0],
            ]),
            1: np.array([
                [0, 0, 1.0],
                [1.0, 0, 0],
                [0, 1.0, 0],
            ]),
            2: np.array([
                [-0.5, 0, 0, -0.5, 0, 1.0],
                [0, 0, 1.0, 0, 0, 0],
                [0, 0, 0, 0, 1.0, 0],
                [0.86602540378443864676, 0, 0, -0.86602540378443864676, 0, 0],
                [0, 1.0, 0, 0, 0, 0],
            ]),
            3: np.array([
                [0, 0, -0.67082039324993690892, 0, 0, 0, 0, -0.67082039324993690892, 0, 1.0], # noqa
                [-0.61237243569579452455, 0, 0, -0.27386127875258305673, 0, 1.0954451150103322269, 0, 0, 0, 0], # noqa
                [0, -0.27386127875258305673, 0, 0, 0, 0, -0.61237243569579452455, 0, 1.0954451150103322269, 0], # noqa
                [0, 0, 0.86602540378443864676, 0, 0, 0, 0, -0.86602540378443864676, 0, 0], # noqa
                [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                [0.790569415042094833, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0], # noqa
                [0, 1.0606601717798212866, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0], # noqa
            ]),
            4: np.array([
                [0.375, 0, 0, 0.21957751641341996535, 0, -0.87831006565367986142, 0, 0, 0, 0, 0.375, 0, -0.87831006565367986142, 0, 1.0], # noqa
                [0, 0, -0.89642145700079522998, 0, 0, 0, 0, -0.40089186286863657703, 0, 1.19522860933439364, 0, 0, 0, 0, 0], # noqa
                [0, 0, 0, 0, -0.40089186286863657703, 0, 0, 0, 0, 0, 0, -0.89642145700079522998, 0, 1.19522860933439364, 0], # noqa
                [-0.5590169943749474241, 0, 0, 0, 0, 0.9819805060619657157, 0, 0, 0, 0, 0.5590169943749474241, 0, -0.9819805060619657157, 0, 0], # noqa
                [0, -0.42257712736425828875, 0, 0, 0, 0, -0.42257712736425828875, 0, 1.1338934190276816816, 0, 0, 0, 0, 0, 0], # noqa
                [0, 0, 0.790569415042094833, 0, 0, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0, 0], # noqa
                [0, 0, 0, 0, 1.0606601717798212866, 0, 0, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0], # noqa
                [0.73950997288745200532, 0, 0, -1.2990381056766579701, 0, 0, 0, 0, 0, 0, 0.73950997288745200532, 0, 0, 0, 0], # noqa
                [0, 1.1180339887498948482, 0, 0, 0, 0, -1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0, 0], # noqa
            ]),
            5: np.array([
                [0, 0, 0.625, 0, 0, 0, 0, 0.36596252735569994226, 0, -1.0910894511799619063, 0, 0, 0, 0, 0, 0, 0.625, 0, -1.0910894511799619063, 0, 1.0], # noqa
                [0.48412291827592711065, 0, 0, 0.21128856368212914438, 0, -1.2677313820927748663, 0, 0, 0, 0, 0.16137430609197570355, 0, -0.56694670951384084082, 0, 1.2909944487358056284, 0, 0, 0, 0, 0, 0], # noqa
                [0, 0.16137430609197570355, 0, 0, 0, 0, 0.21128856368212914438, 0, -0.56694670951384084082, 0, 0, 0, 0, 0, 0, 0.48412291827592711065, 0, -1.2677313820927748663, 0, 1.2909944487358056284, 0], # noqa
                [0, 0, -0.85391256382996653193, 0, 0, 0, 0, 0, 0, 1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0.85391256382996653193, 0, -1.1180339887498948482, 0, 0], # noqa
                [0, 0, 0, 0, -0.6454972243679028142, 0, 0, 0, 0, 0, 0, -0.6454972243679028142, 0, 1.2909944487358056284, 0, 0, 0, 0, 0, 0, 0], # noqa
                [-0.52291251658379721749, 0, 0, 0.22821773229381921394, 0, 0.91287092917527685576, 0, 0, 0, 0, 0.52291251658379721749, 0, -1.2247448713915890491, 0, 0, 0, 0, 0, 0, 0, 0], # noqa
                [0, -0.52291251658379721749, 0, 0, 0, 0, -0.22821773229381921394, 0, 1.2247448713915890491, 0, 0, 0, 0, 0, 0, 0.52291251658379721749, 0, -0.91287092917527685576, 0, 0, 0], # noqa
                [0, 0, 0.73950997288745200532, 0, 0, 0, 0, -1.2990381056766579701, 0, 0, 0, 0, 0, 0, 0, 0, 0.73950997288745200532, 0, 0, 0, 0], # noqa
                [0, 0, 0, 0, 1.1180339887498948482, 0, 0, 0, 0, 0, 0, -1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa
                [0.7015607600201140098, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 1.169267933366856683, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa
                [0, 1.169267933366856683, 0, 0, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 0, 0, 0.7015607600201140098, 0, 0, 0, 0, 0], # noqa
            ])
        }

        def reorder(T, l):
            idxs = np.array([2*i + 2 for i in range(l-1, -1, -1)] + [0] +
                            [2*i + 1 for i in range(l)])
            return T[idxs]

        for l, X_l_test in test_projection_matrices.items():

            X_l = _make_gto_cho_tensor(l, ijks=test_ijks[l]).detach().numpy()
            X_l_test_reordered = reorder(X_l_test, l)

            for X_lm, X_lm_test in zip(X_l, X_l_test_reordered):
                self.assertTrue(np.allclose(X_lm, X_lm_test))

    def test_gaussian_3d(self):

        dx = torch.linspace(-3, 3, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T
        psi = gaussian_3d(xyz, mu=0.0, beta=1.0).reshape(x.shape)
        A = torch.trapz(torch.trapz(torch.trapz(psi, dx), dx), dx)

        self.assertAlmostEqual(A.detach().numpy(),
                               np.pi**(3/2),
                               places=3)

    def test_l1_gaussian_3d(self):

        dx = torch.linspace(-3, 3, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T
        psi = l1_gaussian_3d(xyz, mu=0.0, gamma=0.8).reshape(x.shape)
        A = torch.trapz(torch.trapz(torch.trapz(psi, dx), dx), dx)

        self.assertAlmostEqual(A.detach().numpy(),
                               1.0,
                               places=3)

    def test_l2_gaussian_3d(self):

        dx = torch.linspace(-3, 3, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T
        psi = l2_gaussian_3d(xyz, mu=0.0, alpha=1.0).reshape(x.shape)
        N = torch.trapz(torch.trapz(torch.trapz(psi**2, dx), dx), dx)

        self.assertAlmostEqual(N.detach().numpy(),
                               1.0,
                               places=3)

    def test_gto(self):

        dx = torch.linspace(-3, 3, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        # test for (i,j,k) = (2, 1, 0)
        ijk_1 = torch.tensor([2, 1, 0])
        psi_1 = gto(xyz, mu=0.0, ijk=ijk_1, alpha=2.0).reshape(x.shape)
        N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)
        self.assertAlmostEqual(N_1.detach().numpy(),
                               1.0,
                               places=3)

        # test for (i,j,k) = (3, 2, 4)
        ijk_2 = torch.tensor([3, 2, 4])
        psi_2 = gto(xyz, mu=0.0, ijk=ijk_2, alpha=2.0).reshape(x.shape)
        N_2 = torch.trapz(torch.trapz(torch.trapz(psi_2**2, dx), dx), dx)
        self.assertAlmostEqual(N_2.detach().numpy(),
                               1.0,
                               places=3)

    def test_potential_gto(self):

        dx = torch.linspace(-8, 8, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        # test for (i,j,k) = (0, 0, 0)
        ijk_1 = torch.tensor([0, 0, 0])
        F_1 = potential_gto(xyz, mu=0.0, ijk=ijk_1, gamma=0.2)\
            .reshape(x.shape)
        N_1 = torch.trapz(torch.trapz(torch.trapz(torch.abs(F_1), dx), dx), dx)
        self.assertAlmostEqual(N_1.detach().numpy(),
                               1.0,
                               places=3)

        # test for (i,j,k) = (3, 2, 4)
        ijk_2 = torch.tensor([3, 2, 4])
        F_2 = potential_gto(xyz, mu=0.0, ijk=ijk_2, gamma=0.6)\
            .reshape(x.shape)
        N_2 = torch.trapz(torch.trapz(torch.trapz(torch.abs(F_2), dx), dx), dx)
        self.assertAlmostEqual(N_2.detach().numpy(),
                               1.0,
                               places=3)

    def test_sto_ng_norm(self):

        alphas = torch.linspace(0.3, 1.2, 5).view(1, -1)
        cs = torch.linspace(0.1, 2, 5).view(1, -1)

        # test for (i,j,k) = (2, 1, 0)
        ijk_1 = torch.tensor([[2, 1, 0]])
        cs_norm_1 = sto_ng_norm_coeff(
            ijk_1,
            alphas,
            torch.zeros((1), dtype=torch.int),
            cs).squeeze()

        norm_1 = sto_ng_norm_coeff(
            ijk_1,
            alphas,
            torch.zeros((1), dtype=torch.int),
            cs / cs_norm_1)

        self.assertEqual(norm_1.squeeze().detach().numpy(), 1.0)

        # test for (i,j,k) = (3,2,4)
        ijk_2 = torch.tensor([[3, 2, 4]])
        cs_norm_2 = sto_ng_norm_coeff(
            ijk_2,
            alphas,
            torch.zeros((1), dtype=torch.int),
            cs).squeeze()

        norm_2 = sto_ng_norm_coeff(
            ijk_2,
            alphas,
            torch.zeros((1), dtype=torch.int),
            cs / cs_norm_2)

        self.assertEqual(norm_2.squeeze().detach().numpy(), 1.0)

    def test_sto_ng(self):

        dx = torch.linspace(-5, 5, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.3, 1.2, 5).view(1, -1)
        mu = torch.zeros(3)
        cs = torch.linspace(0.1, 2, 5).view(1, -1)

        # test for (i,j,k) = (2, 1, 0)
        ijk_1 = torch.tensor([[2, 1, 0]])
        norm_1 = sto_ng_norm_coeff(
            ijk_1,
            alphas,
            torch.zeros((1), dtype=torch.int),
            cs).squeeze()

        psi_1 = sto_ng(
            xyz, 
            mu=mu, 
            ijk=ijk_1.squeeze(), 
            alphas=alphas.squeeze(), 
            cs=cs.squeeze()).reshape(x.shape)

        N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)

        self.assertAlmostEqual(N_1.detach().numpy(),
                               norm_1.detach().numpy()**2,
                               places=4)

        # test for (i,j,k) = (3,2,4)
        ijk_2 = torch.tensor([[3, 2, 4]])
        norm_2 = sto_ng_norm_coeff(
            ijk_2,
            alphas,
            torch.zeros((1), dtype=torch.int),
            cs).squeeze()

        psi_2 = sto_ng(
            xyz, 
            mu=mu, 
            ijk=ijk_2.squeeze(), 
            alphas=alphas.squeeze(), 
            cs=cs.squeeze()).reshape(x.shape)

        N_2 = torch.trapz(torch.trapz(torch.trapz(psi_2**2, dx), dx), dx)

        self.assertAlmostEqual(N_2.detach().numpy(),
                               norm_2.detach().numpy()**2,
                               places=4)

    def test_sto_ng_overlap_matrix(self):

        dx = torch.linspace(-6, 6, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.4, 1.5, 5).view(1, 1, -1)
        alpha_idxs = torch.zeros((1, 1), dtype=torch.int)
        cs = torch.linspace(0.9, 0.1, 5).view(1, 1, -1)
        mu = torch.tensor([[0.0, 0.0, 0.0]])

        # test for (i,j,k) = (2, 1, 0), unnormalized inner product:
        ijk_1 = torch.tensor([[[2, 1, 0]]])
        M_1 = sto_ng_overlap_matrix(
            mu, ijk_1, alphas, 
            alpha_idxs, cs,
            mu, ijk_1, alphas, 
            alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(xyz, mu=mu, ijk=ijk_1, alphas=alphas, cs=cs)\
            .reshape(x.shape)
        N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)

        self.assertAlmostEqual(N_1.detach().numpy(),
                               M_1.detach().numpy(),
                               places=5)

        # product of orbital (i,j,k) = (2,1,0) with (i,j,k) = (3,2,4), 
        # but shifted
        mu_shifted = torch.tensor([0.1, 0.2, 0.5])
        ijk_2 = torch.tensor([[3, 2, 4]])
        M_12 = sto_ng_overlap_matrix(
            mu, ijk_1, alphas, 
            alpha_idxs, cs, 
            mu_shifted, ijk_2, alphas, 
            alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(
            xyz, 
            mu=mu, ijk=ijk_1, 
            alphas=alphas, cs=cs).reshape(x.shape)

        psi_2_shifted = sto_ng(
            xyz, mu=mu_shifted, ijk=ijk_2, 
            alphas=alphas, cs=cs).reshape(x.shape)

        rho = psi_1 * psi_2_shifted
        N_12 = torch.trapz(torch.trapz(torch.trapz(rho, dx), dx), dx)

        self.assertAlmostEqual(N_12.detach().numpy(),
                               M_12.detach().numpy(),
                               places=5)

    def test_sto_ng_overlap_matrix_full(self):

        def orthogonality_matrix(ijks):
            ijks_np = ijks.detach().numpy()
            ijks_1 = ijks_np.reshape(1, -1, 3)
            ijks_2 = ijks_np.reshape(-1, 1, 3)
            ortho_parity_mask = ((ijks_1 + ijks_2 + 1) % 2).prod(-1)

            return 1-ortho_parity_mask

        test_ijks = {
            0: [(0, 0, 0)],
            1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            2: [(2, 0, 0), (1, 1, 0), (1, 0, 1),
                (0, 2, 0), (0, 1, 1), (0, 0, 2)],

            3: [(3, 0, 0), (2, 1, 0), (2, 0, 1),
                (1, 2, 0), (1, 1, 1), (1, 0, 2),
                (0, 3, 0), (0, 2, 1), (0, 1, 2), 
                (0, 0, 3)],
            4: [(4, 0, 0), (3, 1, 0), (3, 0, 1), 
                (2, 2, 0), (2, 1, 1), (2, 0, 2), 
                (1, 3, 0), (1, 2, 1), (1, 1, 2), 
                (1, 0, 3), (0, 4, 0), (0, 3, 1), 
                (0, 2, 2), (0, 1, 3), (0, 0, 4)],
        }

        full_ijks = torch.vstack([
            torch.tensor(l_ijks)
            for l_ijks in test_ijks.values()
        ])

        exact_ijk_orthos = orthogonality_matrix(full_ijks)

        alphas = torch.linspace(0.4, 1.5, 2).view(1, -1)
        alpha_idxs = torch.zeros((full_ijks.shape[0]), dtype=torch.int)
        cs = torch.linspace(0.9, 0.1, 2).view(1, -1)\
            .expand(full_ijks.shape[0], -1)
        cs_normed = cs / sto_ng_norm_coeff(full_ijks, alphas, 
                                           alpha_idxs, cs)\
            .unsqueeze(-1)
        mu = torch.tensor([0.0, 0.0, 0.0])

        M = sto_ng_overlap_matrix(
            mu, full_ijks, alphas, 
            alpha_idxs, cs_normed,
            mu, full_ijks, alphas, 
            alpha_idxs, cs_normed).squeeze()

        # ensure diagonal contains 1's:
        self.assertTrue(np.allclose(np.diag(M), 1.0))

        # check orthogonality is correct:
        M_orthos = np.isclose(M.detach().numpy(), 0).astype(np.int64)

        self.assertTrue(np.allclose(exact_ijk_orthos, M_orthos))

    def test_sto_ng_block_overlap_matrix(self):

        dx = torch.linspace(-6, 6, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.4, 1.5, 5).view(1, 1, -1)
        alpha_idxs = torch.zeros((1, 1), dtype=torch.int)
        cs = torch.linspace(0.9, 0.1, 5).view(1, 1, -1)
        mu = torch.tensor([[0.0, 0.0, 0.0]])

        # test for (i,j,k) = (2, 1, 0), unnormalized inner product:
        ijk_1 = torch.tensor([[[2, 1, 0]]])
        M_1 = sto_ng_block_overlap_matrix(
            mu, ijk_1, alphas, 
            alpha_idxs, cs,
            mu, ijk_1, alphas, 
            alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(xyz, mu=mu, ijk=ijk_1, alphas=alphas, cs=cs)\
            .reshape(x.shape)
        N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)

        self.assertAlmostEqual(N_1.detach().numpy(),
                               M_1.detach().numpy(),
                               places=5)

        # product of orbital (i,j,k) = (2,1,0) with (i,j,k) = (3,2,4), 
        # but shifted
        mu_shifted = torch.tensor([0.1, 0.2, 0.5])
        ijk_2 = torch.tensor([[3, 2, 4]])
        M_12 = sto_ng_block_overlap_matrix(
            mu, ijk_1, alphas, 
            alpha_idxs, cs, 
            mu_shifted, ijk_2, alphas, 
            alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(
            xyz, 
            mu=mu, ijk=ijk_1, 
            alphas=alphas, cs=cs).reshape(x.shape)

        psi_2_shifted = sto_ng(
            xyz, mu=mu_shifted, ijk=ijk_2, 
            alphas=alphas, cs=cs).reshape(x.shape)

        rho = psi_1 * psi_2_shifted
        N_12 = torch.trapz(torch.trapz(torch.trapz(rho, dx), dx), dx)

        self.assertAlmostEqual(N_12.detach().numpy(),
                               M_12.detach().numpy(),
                               places=5)

    def test_sto_ng_block_overlap_matrix_shape(self):

        n1, n2 = 4, 5
        b1, b2 = 2, 3
        i1, i2 = 6, 7
        mu_n = 8

        alphas_1 = torch.linspace(0.4, 1.5, n1).view(1, 1, -1)\
            .expand(mu_n, b1, n1)
        alphas_2 = torch.linspace(0.4, 1.5, n2).view(1, 1, -1)\
            .expand(mu_n, b2, n2)
        cs_1 = torch.linspace(0.9, 0.1, n1).view(1, 1, -1).expand(mu_n, i1, n1)
        cs_2 = torch.linspace(0.9, 0.1, n2).view(1, 1, -1).expand(mu_n, i2, n2)

        mu_1 = torch.tensor([[0.0, 0.0, 0.0]]).expand(mu_n, 3)
        mu_2 = torch.tensor([[0.1, 0.1, 0.1]]).expand(mu_n, 3)

        ijk_1 = torch.tensor([[[2, 1, 0]]]).expand(mu_n, i1, 3)
        ijk_2 = torch.tensor([[[2, 1, 0]]]).expand(mu_n, i2, 3)

        alpha_idxs_1 = torch.tensor([[0, 1, 0, 1, 0, 1]]).expand(mu_n, i1)
        alpha_idxs_2 = torch.tensor([[0, 1, 2, 0, 1, 2, 0]]).expand(mu_n, i2)

        # ensure primitive overlap matrix has the correct shape:
        T = sto_ng_block_primitive_overlap_matrix(
            mu_1, alphas_1, mu_2, alphas_2)

        self.assertEqual(tuple(T.shape[:-2]), (mu_n, b1, b2, n1, n2, 3))
        self.assertEqual(T.shape[-2], T.shape[-1])

        # ensure batched overlap matrices have the correct shape:
        M = sto_ng_block_overlap_matrix(
            mu_1, ijk_1, alphas_1, alpha_idxs_1, cs_1,
            mu_2, ijk_2, alphas_2, alpha_idxs_2, cs_2)

        self.assertEqual(tuple(M.shape), (mu_n, i1, i2))

        # ensure norm are computed with the correct shapes:
        cs_1_norm = sto_ng_norm_coeff(ijk_1, alphas_1, alpha_idxs_1, cs_1)
        self.assertEqual(tuple(cs_1_norm.shape), tuple(cs_1.shape[:-1]))

        cs_2_norm = sto_ng_norm_coeff(ijk_2, alphas_2, alpha_idxs_2, cs_2)
        self.assertEqual(tuple(cs_2_norm.shape), tuple(cs_2.shape[:-1]))

    def test_sto_ng_potential_overlap_matrix(self):

        dx = torch.linspace(-6, 6, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.4, 1.5, 5).view(1, 1, -1)
        gammas = torch.linspace(0.4, 1.5, 4).view(1, -1)

        alpha_idxs = torch.zeros((1, 1), dtype=torch.int)
        cs = torch.linspace(0.9, 0.1, 5).view(1, 1, -1)
        vs = torch.linspace(-10, -5, 4).view(1, -1)

        mu = torch.tensor([[0.0, 0.0, 0.0]])

        # test for (i,j,k) = (2, 1, 0), unnormalized inner product:
        ijk_1 = torch.tensor([[[2, 1, 0]]])
        M_1 = sto_ng_potential_overlap_matrix(
            mu, ijk_1, alphas, alpha_idxs, cs,
            gammas, vs,
            mu, ijk_1, alphas, alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(xyz, mu=mu, ijk=ijk_1, alphas=alphas, cs=cs)\
            .reshape(x.shape)

        V_1 = potential_ng(
            xyz, mu, torch.tensor([0, 0, 0]), 
            gammas.squeeze(), 
            vs.squeeze()).reshape(x.shape)

        # ensure potential is evaluated correctly:
        N_1 = torch.trapz(torch.trapz(torch.trapz(
            psi_1*V_1*psi_1, dx), dx), dx)
        self.assertAlmostEqual(N_1.detach().numpy(),
                               M_1.detach().numpy(),
                               places=5)

        # test the product of orbital (i,j,k) = (2,1,0) 
        # with (i,j,k) = (3,2,4), but shifted
        mu_shifted = torch.tensor([0.1, 0.2, 0.5])
        ijk_2 = torch.tensor([[3, 2, 4]])
        M_12 = sto_ng_potential_overlap_matrix(
            mu, 
            ijk_1, alphas, 
            alpha_idxs, cs,
            gammas, vs,
            mu_shifted, ijk_2, alphas, 
            alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(
            xyz, mu=mu, ijk=ijk_1, 
            alphas=alphas, cs=cs).reshape(x.shape)

        psi_2_shifted = sto_ng(
            xyz, mu=mu_shifted, ijk=ijk_2, 
            alphas=alphas, cs=cs).reshape(x.shape)

        N_12 = torch.trapz(torch.trapz(torch.trapz(
            psi_1 * V_1 * psi_2_shifted, dx), dx), dx)

        self.assertAlmostEqual(N_12.detach().numpy(),
                               M_12.detach().numpy(),
                               places=5)

    def test_sto_ng_block_potential_overlap_matrix(self):

        dx = torch.linspace(-6, 6, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.4, 1.5, 5).view(1, 1, -1)
        gammas = torch.linspace(0.4, 1.5, 4).view(1, -1)

        alpha_idxs = torch.zeros((1, 1), dtype=torch.int)
        cs = torch.linspace(0.9, 0.1, 5).view(1, 1, -1)
        vs = torch.linspace(-10, -5, 4).view(1, -1)

        mu = torch.tensor([[0.0, 0.0, 0.0]])

        # test for (i,j,k) = (2, 1, 0), unnormalized inner product:
        ijk_1 = torch.tensor([[[2, 1, 0]]])
        M_1 = sto_ng_block_potential_overlap_matrix(
            mu, ijk_1, alphas, alpha_idxs, cs,
            gammas, vs,
            mu, ijk_1, alphas, alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(xyz, mu=mu, ijk=ijk_1, alphas=alphas, cs=cs)\
            .reshape(x.shape)

        V_1 = potential_ng(
            xyz, mu, torch.tensor([0, 0, 0]), 
            gammas.squeeze(), 
            vs.squeeze()).reshape(x.shape)

        # ensure potential is evaluated correctly:
        N_1 = torch.trapz(torch.trapz(torch.trapz(
            psi_1*V_1*psi_1, dx), dx), dx)
        self.assertAlmostEqual(N_1.detach().numpy(),
                               M_1.detach().numpy(),
                               places=5)

        # test the product of orbital (i,j,k) = (2,1,0) 
        # with (i,j,k) = (3,2,4), but shifted
        mu_shifted = torch.tensor([0.1, 0.2, 0.5])
        ijk_2 = torch.tensor([[3, 2, 4]])
        M_12 = sto_ng_block_potential_overlap_matrix(
            mu, 
            ijk_1, alphas, 
            alpha_idxs, cs,
            gammas, vs,
            mu_shifted, ijk_2, alphas, 
            alpha_idxs, cs).squeeze()

        psi_1 = sto_ng(
            xyz, mu=mu, ijk=ijk_1, 
            alphas=alphas, cs=cs).reshape(x.shape)

        psi_2_shifted = sto_ng(
            xyz, mu=mu_shifted, ijk=ijk_2, 
            alphas=alphas, cs=cs).reshape(x.shape)

        N_12 = torch.trapz(torch.trapz(torch.trapz(
            psi_1 * V_1 * psi_2_shifted, dx), dx), dx)

        self.assertAlmostEqual(N_12.detach().numpy(),
                               M_12.detach().numpy(),
                               places=5)

    def test_cho_ng(self):

        dx = torch.linspace(-5, 5, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(1.2, 1.2, 5).view(1, -1)
        mu = torch.zeros(3)
        cs = torch.linspace(1.0, 1.0, 5).view(1, -1)

        # test for (l, m) = (4, 1)
        l_1 = torch.tensor([2], dtype=torch.int64)

        for m_1_val in range(-l_1, l_1+1):
            m_1 = torch.tensor([m_1_val], dtype=torch.int64)

            norm_1 = cho_ng_norm_coeff(
                l_1, alphas, 
                torch.zeros((1), dtype=torch.int), 
                cs).squeeze()

            cs_1 = cs / norm_1.unsqueeze(-1)

            # ensure normalized orbitals have a consistent norm of 1:
            alpha_idxs_1 = torch.zeros((1), dtype=torch.int64)
            normed_norm = cho_ng_norm_coeff(
                l_1, alphas, 
                alpha_idxs_1, 
                cs_1).squeeze()

            self.assertAlmostEqual(normed_norm.detach().numpy(), 1.0, places=6)

            # evaluate integral of cubic harmonic orbital:
            psi_1 = cho_ng(
                xyz, mu=mu, l=l_1, m=m_1, 
                alphas=alphas, cs=cs_1).reshape(x.shape)

            N_1 = torch.trapz(torch.trapz(
                torch.trapz(psi_1*psi_1, dx), dx), dx)

            self.assertAlmostEqual(N_1.detach().numpy(), 1.0, places=4)

    def test_cho_ng_overlap_matrix(self):

        n_l = 5
        n_gauss = 3
        alphas = torch.linspace(0.1, 0.4, n_gauss)\
            .view(1, -1).expand(n_l, -1)
        alpha_idxs = torch.arange(n_l)
        mu = torch.zeros((n_l, 3))

        cs = torch.linspace(0.1, 0.7, n_gauss)\
            .view(1, -1).expand(n_l, -1)

        l = torch.arange(0, n_l)
        cs_normed = cs / cho_ng_norm_coeff(l, alphas, alpha_idxs, cs)\
            .unsqueeze(-1)

        M_X = cho_ng_overlap_matrix(
            mu, l, alphas, cs_normed,
            mu, l, alphas, cs_normed)

        M_X_np = M_X.detach().numpy()

        self.assertTrue(
            np.allclose(M_X_np, np.eye(M_X_np.shape[0]), atol=1e-6))

    def test_cho_ng_potential_overlap_matrix(self):

        dx = torch.linspace(-5, 5, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        n_gauss = 3
        V_n_gauss = 5
        alphas = torch.linspace(0.5, 0.9, n_gauss)\
            .view(1, -1).expand(1, -1)
        cs = torch.linspace(0.1, 0.7, n_gauss)\
            .view(1, -1).expand(1, -1)

        mu_1 = torch.tensor([0.0, 0.0, 0.0])
        mu_2 = torch.tensor([0.1, 0.2, 0.3])
        l_1 = torch.tensor([2])
        l_2 = torch.tensor([3])

        gammas_1 = torch.linspace(0.2, 0.5, V_n_gauss)
        vs_1 = torch.linspace(-100, -20, V_n_gauss)

        V = potential_ng(
            xyz, mu_1, 
            ijk=torch.tensor([0, 0, 0]), 
            gammas=gammas_1, 
            vs=vs_1).reshape(x.shape)

        # compute block overlap matrix:
        M_X = cho_ng_potential_overlap_matrix(
            mu_1, l_1, alphas, cs,
            gammas_1, vs_1,
            mu_2, l_2, alphas, cs)

        for m_1 in range(-l_1, l_1+1):
            for m_2 in range(-l_2, l_2+1):

                # compare with numerical results:
                psi_1 = cho_ng(
                    xyz, mu=mu_1, 
                    l=l_1, m=m_1, 
                    alphas=alphas.squeeze(),
                    cs=cs.squeeze()).reshape(x.shape)

                psi_2 = cho_ng(
                    xyz, mu=mu_2, 
                    l=l_2, m=m_2, 
                    alphas=alphas.squeeze(),
                    cs=cs.squeeze()).reshape(x.shape)

                I = torch.trapz(torch.trapz(
                    torch.trapz(psi_1*psi_2*V, dx), dx), dx)

                self.assertAlmostEqual(
                    I.item(), 
                    M_X[l_1+m_1, l_2+m_2].item(), 
                    places=5)

    def test_cho_ng_potential_overlap_matrix_shape(self):

        n_gauss = 3
        V_n_gauss = 5

        _mu_1 = torch.tensor([0.0, 0.0, 0.0])
        _mu_2 = torch.tensor([0.1, 0.2, 0.3])

        alphas = torch.linspace(0.5, 0.9, n_gauss)
        cs = torch.linspace(0.1, 0.7, n_gauss)

        l_1 = torch.tensor([0, 1, 2])
        l_2 = torch.tensor([3])

        mu_1 = _mu_1.view(1, 3)\
            .expand(len(l_1), -1)

        mu_2 = _mu_2.view(1, 3)\
            .expand(len(l_2), -1)

        alphas_1 = alphas.view(1, -1)\
            .expand(len(l_1), -1)
        cs_1 = cs.view(1, -1)\
            .expand(len(l_1), -1)

        alphas_2 = alphas.view(1, -1)\
            .expand(len(l_2), -1)
        cs_2 = cs.view(1, -1)\
            .expand(len(l_2), -1)

        gammas_1 = torch.linspace(0.2, 0.5, V_n_gauss)
        vs_1 = torch.linspace(-100, -20, V_n_gauss)

        M_X = cho_ng_potential_overlap_matrix(
            mu_1, l_1, alphas_1, cs_1,
            gammas_1, vs_1,
            mu_2, l_2, alphas_2, cs_2
        )

        # ensure that the shape returned is correct
        expected_shape = (
            (2*l_1+1).sum().item(),
            (2*l_2+1).sum().item()
        )
        self.assertEqual(tuple(M_X.shape), expected_shape)

    def test_cho_ng_block_overlap_matrix(self):

        n_l = 5
        n_gauss = 3
        alphas = torch.linspace(0.2, 0.5, n_gauss)\
            .view(1, -1).expand(n_l, -1)
        alpha_idxs = torch.arange(n_l)
        mu = torch.zeros((3))

        cs = torch.linspace(0.1, 0.7, n_gauss)\
            .view(1, -1).expand(n_l, -1)

        l = torch.arange(0, n_l)
        cs_normed = cs / cho_ng_norm_coeff(l, alphas, alpha_idxs, cs)\
            .unsqueeze(-1)

        M_X = cho_ng_block_overlap_matrix(
            mu, l, alphas, cs_normed, 
            mu, l, alphas, cs_normed)

        M_X_np = M_X.detach().numpy()

        self.assertTrue(
            np.allclose(M_X_np, np.eye(M_X_np.shape[0]), atol=1e-6))

    def test_cho_ng_block_potential_overlap_matrix(self):

        dx = torch.linspace(-5, 5, 32)
        x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        n_gauss = 3
        V_n_gauss = 5
        alphas = torch.linspace(0.5, 0.9, n_gauss)\
            .view(1, -1).expand(1, -1)
        cs = torch.linspace(0.1, 0.7, n_gauss)\
            .view(1, -1).expand(1, -1)

        mu_1 = torch.tensor([0.0, 0.0, 0.0])
        mu_2 = torch.tensor([0.1, 0.2, 0.3])
        l_1 = torch.tensor([2])
        l_2 = torch.tensor([3])

        gammas_1 = torch.linspace(0.2, 0.5, V_n_gauss)
        vs_1 = torch.linspace(-100, -20, V_n_gauss)

        V = potential_ng(
            xyz, mu_1, 
            ijk=torch.tensor([0, 0, 0]), 
            gammas=gammas_1, 
            vs=vs_1).reshape(x.shape)

        # compute block overlap matrix:
        M_X = cho_ng_block_potential_overlap_matrix(
            mu_1, l_1, alphas, cs,
            gammas_1, vs_1,
            mu_2, l_2, alphas, cs)

        for m_1 in range(-l_1, l_1+1):
            for m_2 in range(-l_2, l_2+1):

                # compare with numerical results:
                psi_1 = cho_ng(
                    xyz, mu=mu_1, 
                    l=l_1, m=m_1, 
                    alphas=alphas.squeeze(),
                    cs=cs.squeeze()).reshape(x.shape)

                psi_2 = cho_ng(
                    xyz, mu=mu_2, 
                    l=l_2, m=m_2, 
                    alphas=alphas.squeeze(),
                    cs=cs.squeeze()).reshape(x.shape)

                I = torch.trapz(torch.trapz(
                    torch.trapz(psi_1*psi_2*V, dx), dx), dx)

                self.assertAlmostEqual(
                    I.item(), 
                    M_X[l_1+m_1, l_2+m_2].item(), 
                    places=5)

    def test_cho_ng_block_overlap_matrix_shape(self):

        n_gauss = 3
        batch_shape = (2, 1, 3)

        l_1 = torch.tensor([1, 2])
        l_2 = torch.tensor([2])

        def pad_batch(a, batch_pad):
            a_shape = a.shape
            return a.view(*([1]*len(batch_pad)), *a_shape)\
                    .expand(*batch_pad, *a.shape)

        alphas_1 = pad_batch(torch.vstack([
            torch.linspace(0.5, 0.9, n_gauss),
            torch.linspace(0.4, 0.8, n_gauss)
        ]), batch_shape)

        cs_1 = pad_batch(torch.vstack([
            torch.linspace(0.1, 0.7, n_gauss),
            torch.linspace(0.2, 0.8, n_gauss)
        ]), batch_shape)

        alphas_2 = pad_batch(torch.vstack([
            torch.linspace(0.5, 0.6, n_gauss),
        ]), batch_shape)

        cs_2 = pad_batch(torch.vstack([
            torch.linspace(0.2, 0.7, n_gauss),
        ]), batch_shape)

        mu_1 = pad_batch(torch.tensor([0.0, 0.0, 0.0]), batch_shape)
        mu_2 = pad_batch(torch.tensor([0.1, 0.2, 0.3]), batch_shape)

        M_shape = (torch.sum(2*l_1+1).item(),
                   torch.sum(2*l_2+1).item())

        M_X = cho_ng_block_overlap_matrix(
            mu_1, l_1, alphas_1, cs_1, 
            mu_2, l_2, alphas_2, cs_2
        )

        # ensure batch size is correct
        self.assertEqual(M_X.shape[:-2], batch_shape)

        # ensure matrix block size is correct
        self.assertEqual(M_X.shape[-2:], M_shape)

    def test_cho_ng_block_potential_overlap_matrix_shape(self):

        n_gauss = 3
        V_n_gauss = 5
        batch_shape = (3, 2, 1)

        l_1 = torch.tensor([1, 2])
        l_2 = torch.tensor([3])

        def pad_batch(a, batch_pad):
            a_shape = a.shape
            return a.view(*([1]*len(batch_pad)), *a_shape)\
                    .expand(*batch_pad, *a.shape)

        alphas_1 = pad_batch(torch.vstack([
            torch.linspace(0.5, 0.9, n_gauss),
            torch.linspace(0.4, 0.8, n_gauss)
        ]), batch_shape)

        cs_1 = pad_batch(torch.vstack([
            torch.linspace(0.1, 0.7, n_gauss),
            torch.linspace(0.2, 0.8, n_gauss)
        ]), batch_shape)

        alphas_2 = pad_batch(torch.vstack([
            torch.linspace(0.5, 0.6, n_gauss),
        ]), batch_shape)

        cs_2 = pad_batch(torch.vstack([
            torch.linspace(0.2, 0.7, n_gauss),
        ]), batch_shape)

        gammas_1 = pad_batch(
            torch.linspace(0.2, 0.5, V_n_gauss),
            batch_shape)
        vs_1 = pad_batch(
            torch.linspace(-100, -20, V_n_gauss),
            batch_shape)

        mu_1 = pad_batch(torch.tensor([0.0, 0.0, 0.0]), batch_shape)
        mu_2 = pad_batch(torch.tensor([0.1, 0.2, 0.3]), batch_shape)

        M_shape = (torch.sum(2*l_1+1).item(),
                   torch.sum(2*l_2+1).item())

        M_X = cho_ng_block_potential_overlap_matrix(
            mu_1, l_1, alphas_1, cs_1, 
            gammas_1, vs_1,
            mu_2, l_2, alphas_2, cs_2
        )

        # ensure batch size is correct
        self.assertEqual(M_X.shape[:-2], batch_shape)

        # ensure matrix block size is correct
        self.assertEqual(M_X.shape[-2:], M_shape)
