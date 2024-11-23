import unittest

import torch
import numpy as np
from math import sqrt
from pyquinn.torch.lcao.gto import (
        _make_gto_cho_tensor, _make_gto_ijk_values, _make_gto_alpha_tensor, _make_gto_hermite_tensor,
        sto_ng_prmitive_overlap_matrix,
        gaussian_3d, l1_gaussian_3d, l2_gaussian_3d, 
        gto_3d, sto_ng_3d, sto_ng_3d_norm, sto_ng_3d_overlap_matrix,
        cho_ng_3d, cho_ng_3d_overlap_matrix
)


class GTOTest(unittest.TestCase):

    def test__make_gto_cho_tensor(self):

        # This conforms to VASP ijk ordering (but we could use any ordering)
        test_ijks = {
                0 : [(0,0,0)],
                1 : [(0,1,0), (0,0,1), (1,0,0)],
                2 : [(1,1,0), (0,1,1), (0,0,2), (1,0,1), (2,0,0), (0,2,0)],
                3 : [(2,1,0), (0,3,0), (1,1,1), (0,1,2), (0,0,3),
                    (1,0,2), (2,0,1), (0,2,1), (3,0,0), (1,2,0) ]
            }

        test_projection_matrices = {
            0 : np.eye(1),
            1 : np.eye(3),
            2 : np.array([
                    [ 1,  0, 0,              0,          0,              0          ],   # dxy -> dxy
                    [ 0,  1, 0,              0,          0,              0,         ],   # dyz -> dyz
                    [ 0,  0, 1,              0,         -1/2,           -1/2        ],   # (dz^2, dx^2, dy^2) -> dz^2
                    [ 0,  0, 0,              1,          0,              0          ],   #  dxz -> dxz
                    [ 0,  0, 0,              0,          sqrt(3)/2,     -sqrt(3)/2  ]]), # (dx^2, dz^2) -> d(x^2 - y^2)

            3 : np.array([ #  x^2y       y^3        xyz   yz^2         z^3   xz^2         x^2z        y^2z        x^3        xy^2
                    np.array([ sqrt(18), -sqrt(10),  0,    0,           0,    0,           0,          0,          0,         0           ])/4,
                    np.array([ 0,         0,         1,    0,           0,    0,           0,          0,          0,         0           ]),
                    np.array([-sqrt(30), -5*sqrt(6), 0,    4*sqrt(30),  0,    0,           0,          0,          0,         0           ])/20,
                    np.array([ 0,         0,         0,    0,           10,   0,          -3*sqrt(5), -3*sqrt(5),  0,         0           ])/10,
                    np.array([ 0,         0,         0,    0,           0,    4*sqrt(30),  0,          0,         -5*sqrt(6), -sqrt(30)   ])/20,
                    np.array([ 0,         0,         0,    0,           0,    0,           sqrt(3),   -sqrt(3),    0,         0           ])/2,
                    np.array([ 0,         0,         0,    0,           0,    0,           0,          0,          sqrt(10),  -sqrt(18)   ])/4,
                ])
        }

        for l, ijks in test_ijks.items():
            X_l = _make_gto_cho_tensor(l, ijks).detach().numpy()
            X_l_test = test_projection_matrices[l]
            for X_lm, X_lm_test in zip(X_l, X_l_test):
                self.assertTrue(np.allclose(X_lm, X_lm_test))
            
    def test__make_gto_cho_tensor_v2(self):

        
        test_ijks = {
            0 : [(0, 0, 0)],
            1 : [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            2 : [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
            3 : [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)],
            4 : [(4, 0, 0), (3, 1, 0), (3, 0, 1), (2, 2, 0), (2, 1, 1), (2, 0, 2), (1, 3, 0), (1, 2, 1), (1, 1, 2), (1, 0, 3), (0, 4, 0), (0, 3, 1), (0, 2, 2), (0, 1, 3), (0, 0, 4)],
            5 : [(5, 0, 0), (4, 1, 0), (4, 0, 1), (3, 2, 0), (3, 1, 1), (3, 0, 2), (2, 3, 0), (2, 2, 1), (2, 1, 2), (2, 0, 3), (1, 4, 0), (1, 3, 1), (1, 2, 2), (1, 1, 3), (1, 0, 4), 
                 (0, 5, 0), (0, 4, 1), (0, 3, 2), (0, 2, 3), (0, 1, 4), (0, 0, 5)],
        }
        
        test_projection_matrices = {
            0 : np.array([
                [1.0],
            ]),
            1 : np.array([
                [0, 0, 1.0],
                [1.0, 0, 0],
                [0, 1.0, 0],
            ]),
            2 : np.array([
                [-0.5, 0, 0, -0.5, 0, 1.0],
                [0, 0, 1.0, 0, 0, 0],
                [0, 0, 0, 0, 1.0, 0],
                [0.86602540378443864676, 0, 0, -0.86602540378443864676, 0, 0],
                [0, 1.0, 0, 0, 0, 0],
            ]),
            3 : np.array([
                [0, 0, -0.67082039324993690892, 0, 0, 0, 0, -0.67082039324993690892, 0, 1.0],
                [-0.61237243569579452455, 0, 0, -0.27386127875258305673, 0, 1.0954451150103322269, 0, 0, 0, 0],
                [0, -0.27386127875258305673, 0, 0, 0, 0, -0.61237243569579452455, 0, 1.0954451150103322269, 0],
                [0, 0, 0.86602540378443864676, 0, 0, 0, 0, -0.86602540378443864676, 0, 0],
                [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                [0.790569415042094833, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0],
                [0, 1.0606601717798212866, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0],
            ]),
            4 : np.array([
                [0.375, 0, 0, 0.21957751641341996535, 0, -0.87831006565367986142, 0, 0, 0, 0, 0.375, 0, -0.87831006565367986142, 0, 1.0],
                [0, 0, -0.89642145700079522998, 0, 0, 0, 0, -0.40089186286863657703, 0, 1.19522860933439364, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -0.40089186286863657703, 0, 0, 0, 0, 0, 0, -0.89642145700079522998, 0, 1.19522860933439364, 0],
                [-0.5590169943749474241, 0, 0, 0, 0, 0.9819805060619657157, 0, 0, 0, 0, 0.5590169943749474241, 0, -0.9819805060619657157, 0, 0],
                [0, -0.42257712736425828875, 0, 0, 0, 0, -0.42257712736425828875, 0, 1.1338934190276816816, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.790569415042094833, 0, 0, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1.0606601717798212866, 0, 0, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0],
                [0.73950997288745200532, 0, 0, -1.2990381056766579701, 0, 0, 0, 0, 0, 0, 0.73950997288745200532, 0, 0, 0, 0],
                [0, 1.1180339887498948482, 0, 0, 0, 0, -1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            5 : np.array([
                [0, 0, 0.625, 0, 0, 0, 0, 0.36596252735569994226, 0, -1.0910894511799619063, 0, 0, 0, 0, 0, 0, 0.625, 0, -1.0910894511799619063, 0, 1.0],
                [0.48412291827592711065, 0, 0, 0.21128856368212914438, 0, -1.2677313820927748663, 0, 0, 0, 0, 0.16137430609197570355, 0, -0.56694670951384084082, 0, 1.2909944487358056284, 0, 0, 0, 0, 0, 0],
                [0, 0.16137430609197570355, 0, 0, 0, 0, 0.21128856368212914438, 0, -0.56694670951384084082, 0, 0, 0, 0, 0, 0, 0.48412291827592711065, 0, -1.2677313820927748663, 0, 1.2909944487358056284, 0],
                [0, 0, -0.85391256382996653193, 0, 0, 0, 0, 0, 0, 1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0.85391256382996653193, 0, -1.1180339887498948482, 0, 0],
                [0, 0, 0, 0, -0.6454972243679028142, 0, 0, 0, 0, 0, 0, -0.6454972243679028142, 0, 1.2909944487358056284, 0, 0, 0, 0, 0, 0, 0],
                [-0.52291251658379721749, 0, 0, 0.22821773229381921394, 0, 0.91287092917527685576, 0, 0, 0, 0, 0.52291251658379721749, 0, -1.2247448713915890491, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -0.52291251658379721749, 0, 0, 0, 0, -0.22821773229381921394, 0, 1.2247448713915890491, 0, 0, 0, 0, 0, 0, 0.52291251658379721749, 0, -0.91287092917527685576, 0, 0, 0],
                [0, 0, 0.73950997288745200532, 0, 0, 0, 0, -1.2990381056766579701, 0, 0, 0, 0, 0, 0, 0, 0, 0.73950997288745200532, 0, 0, 0, 0],
                [0, 0, 0, 0, 1.1180339887498948482, 0, 0, 0, 0, 0, 0, -1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.7015607600201140098, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 1.169267933366856683, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1.169267933366856683, 0, 0, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 0, 0, 0.7015607600201140098, 0, 0, 0, 0, 0],
            ])
        }

        def reorder(T, l):
            idxs = np.array([2*i + 2 for i in range(l-1,-1,-1)] + [0] +
                            [2*i + 1 for i in range(l) ])
            return T[idxs]

        for l, X_l_test in test_projection_matrices.items():

            X_l = _make_gto_cho_tensor(l, ijks=test_ijks[l]).detach().numpy()
            X_l_test_reordered = reorder(X_l_test, l)

            for X_lm, X_lm_test in zip(X_l, X_l_test_reordered):
                self.assertTrue(np.allclose(X_lm, X_lm_test))



    def test_gaussian_3d(self):
        
        dx = torch.linspace(-3, 3, 32)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T
        psi = gaussian_3d(xyz, mu=0.0, beta=1.0).reshape(x.shape)
        A = torch.trapz(torch.trapz(torch.trapz(psi, dx), dx), dx)
        
        self.assertAlmostEqual(A.detach().numpy(),
                               np.pi**(3/2),
                               places=3)

    def test_l1_gaussian_3d(self):
        
        dx = torch.linspace(-3, 3, 32)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T
        psi = l1_gaussian_3d(xyz, mu=0.0, gamma=0.8).reshape(x.shape)
        A = torch.trapz(torch.trapz(torch.trapz(psi, dx), dx), dx)
        
        self.assertAlmostEqual(A.detach().numpy(),
                               1.0,
                               places=3)

    def test_l2_gaussian_3d(self):
        
        dx = torch.linspace(-3, 3, 32)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T
        psi = l2_gaussian_3d(xyz, mu=0.0, alpha=1.0).reshape(x.shape)
        N = torch.trapz(torch.trapz(torch.trapz(psi**2, dx), dx), dx)
        
        self.assertAlmostEqual(N.detach().numpy(),
                               1.0,
                               places=3)

    def test_gto_3d(self):
         
        dx = torch.linspace(-3, 3, 64)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        # test for (i,j,k) = (2, 1, 0)
        ijk_1 = torch.tensor([2, 1, 0])
        psi_1 = gto_3d(xyz, mu=0.0, ijk=ijk_1, alpha=2.0).reshape(x.shape)
        N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)
        self.assertAlmostEqual(N_1.detach().numpy(),
                               1.0,
                               places=3)

        # test for (i,j,k) = (3,2,4)
        ijk_2 = torch.tensor([3, 2, 4])
        psi_2 = gto_3d(xyz, mu=0.0, ijk=ijk_2, alpha=2.0).reshape(x.shape)
        N_2 = torch.trapz(torch.trapz(torch.trapz(psi_2**2, dx), dx), dx)
        self.assertAlmostEqual(N_2.detach().numpy(),
                               1.0,
                               places=3)

    def test_sto_ng_3d(self):

        dx = torch.linspace(-5, 5, 64)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.3, 1.2,5)
        mu = torch.zeros((3))
        cs = torch.linspace(0.1, 2, 5)

        # test for (i,j,k) = (2, 1, 0)
        ijk_1 = torch.tensor([[2, 1, 0]])
        cs_1 = cs / sto_ng_3d_norm(alphas, cs, ijk_1)

        self.assertAlmostEqual(sto_ng_3d_norm(alphas, cs_1, ijk_1).detach().numpy(), 1.0, places=5)

        psi_1 = sto_ng_3d(xyz, mu=mu, ijk=ijk_1, alphas=alphas, cs=cs_1).reshape(x.shape)
        N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)
        
        self.assertAlmostEqual(N_1.detach().numpy(),
                               1.0,
                               places=3)

        # test for (i,j,k) = (3,2,4)
        ijk_2 = torch.tensor([[3, 2, 4]])
        cs_2 = cs / sto_ng_3d_norm(alphas, cs, ijk_2)
        
        self.assertAlmostEqual(sto_ng_3d_norm(alphas, cs_2, ijk_2).detach().numpy(), 1.0, places=5)

        psi_2 = sto_ng_3d(xyz, mu=mu, ijk=ijk_2, alphas=alphas, cs=cs_2).reshape(x.shape)
        N_2 = torch.trapz(torch.trapz(torch.trapz(psi_2**2, dx), dx), dx)
        
        self.assertAlmostEqual(N_2.detach().numpy(),
                               1.0,
                               places=3)

    def test_sto_ng_3d_overlap_matrix(self):

        dx = torch.linspace(-6, 6, 128)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.4, 1.5,5)
        cs = torch.linspace(0.9, 0.1, 5)
        mu = torch.tensor([[0.0, 0.0, 0.0]])

        # test for (i,j,k) = (2, 1, 0), unnormalized inner product:
        ijk_1 = torch.tensor([[2, 1, 0]])
        M_1 = sto_ng_3d_overlap_matrix(mu, ijk_1, alphas, cs, mu, ijk_1, alphas, cs).squeeze()

        psi_1 = sto_ng_3d(xyz, mu=mu, ijk=ijk_1, alphas=alphas, cs=cs).reshape(x.shape)
        N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)

        self.assertAlmostEqual(N_1.detach().numpy(),
                               M_1.detach().numpy(),
                               places=5)

        # product of orbital (i,j,k) = (2,1,0) with (i,j,k) = (3,2,4), but shifted
        mu_shifted = torch.tensor([[0.1, 0.2, 0.5]])
        ijk_2 = torch.tensor([[3,2,4]])
        M_12 = sto_ng_3d_overlap_matrix(mu, ijk_1, alphas, cs, mu_shifted, ijk_2, alphas, cs).squeeze()

        psi_1 = sto_ng_3d(xyz, mu=mu, ijk=ijk_1, alphas=alphas, cs=cs).reshape(x.shape)
        psi_2_shifted = sto_ng_3d(xyz, mu=mu_shifted, ijk=ijk_2, alphas=alphas, cs=cs).reshape(x.shape)
        N_12 = torch.trapz(torch.trapz(torch.trapz(psi_1 * psi_2_shifted, dx), dx), dx)

        self.assertAlmostEqual(N_12.detach().numpy(),
                               M_12.detach().numpy(),
                               places=5)

    def test_sto_ng_3d_overlap_matrix_batched_alphas(self):
        dx = torch.linspace(-6, 6, 128)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.vstack([ 
            torch.linspace(0.4, 1.5,5),
            torch.linspace(0.4, 1.5,5)
        ])
        cs = torch.linspace(0.9, 0.1, 5)
        mu = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ijk = torch.tensor([[2, 1, 0]])

        T = sto_ng_prmitive_overlap_matrix(mu, alphas, mu, alphas)

        print(T.shape)


    def test_cho_ng_3d(self):

        dx = torch.linspace(-5, 5, 64)
        x, y, z = torch.meshgrid((dx,dx,dx), indexing='ij')
        xyz = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

        alphas = torch.linspace(0.3, 1.2,5)
        mu = torch.zeros((3))
        cs = torch.linspace(0.1, 2, 5)


        # test for (l,m) = (3, 1)
        # cs_1 = cs / sto_ng_3d_norm(alphas, cs, ijk_1)

        # self.assertAlmostEqual(sto_ng_3d_norm(alphas, cs_1, ijk_1).detach().numpy(), 1.0, places=5)

        # psi_1 = cho_ng_3d(xyz, mu=mu, l=, alphas=alphas, cs=cs_1).reshape(x.shape)
        # N_1 = torch.trapz(torch.trapz(torch.trapz(psi_1**2, dx), dx), dx)
        
        # self.assertAlmostEqual(N_1.detach().numpy(),
        #                        1.0,
        #                        places=3)



    def test_cho_ng_3d_overlap_matrix(self):
        
        alphas = torch.linspace(0.5, 0.5,1)
        mu = torch.zeros((3))
        cs = torch.linspace(0.1,0.7, 1)
        
        for l in range(5):

            ijks = _make_gto_ijk_values(l)
            cs /= sto_ng_3d_norm(alphas, cs, ijks[:1])

            M_X = cho_ng_3d_overlap_matrix(mu, l, alphas, cs, mu, l, alphas, cs)
            self.assertTrue(np.allclose(M_X.detach().numpy(), np.eye(M_X.shape[0]), atol=1e-6))
        


