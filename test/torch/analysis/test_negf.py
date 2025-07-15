import unittest

import numpy as np
import torch

from pyquinn.torch.analysis.negf import (
    sancho_rubio,
    greens_function
)


class TorchNEGFTest(unittest.TestCase):

    def test_sancho_rubio(self):
        
        def h_chain_g_surface(E, eta, e0, t0):
            z = E+1.j*eta-e0
            g = z/(2*t0**2) - 1.j * torch.sqrt(4*t0**2 - z**2)/(2*t0**2)
            
            return g

        e_onsite = 1.0
        t_hop = -1.0

        etas = [10**n for n in range(6)]
        energies = torch.linspace(-2, 2, 10)

        h = torch.tensor([[e_onsite]], dtype=torch.complex64)
        v = torch.tensor([[t_hop]], dtype=torch.complex64)

        for eta in etas:

            g_numerical = sancho_rubio(
                energies=energies,
                hamiltonian_matrix=h,
                coupling_matrix=v,
                eta=eta,
                tol=1e-8).squeeze().detach().cpu().numpy()

            g_analytical = h_chain_g_surface(
                energies, eta=eta, e0=e_onsite, t0=t_hop).detach().cpu().numpy()

            self.assertTrue(np.allclose(g_numerical, g_analytical, atol=1e-4))

    def test_greens_function(self):

        e_onsite = 1.0
        t_hop = -1.0
        a_lat = 1.0
        k_pts = torch.arange(0, 1/a_lat, 100)
        
        H = (e_onsite * torch.eye(2)).view(1, 2, 2) + \
            (2*t_hop*torch.cos(2*torch.pi*k_pts*a_lat)).view(-1, 1, 1) * \
            torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]).view(1, 2, 2)

        print(H.shape)

        energies = torch.linspace(-2, 2, 10)
        gf = greens_function(energies, H, eta=1e-5)
                
        print(gf.shape)
    
        self.assertTupleEqual(tuple(gf.shape), (100, 10, 2, 2))
