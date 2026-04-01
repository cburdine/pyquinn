import unittest

import numpy as np
import torch

from pyquinn.torch.analysis.negf import (
    EinsteinPhononScatterer,
    PhotonDipoleScatterer,
    sancho_rubio,
    greens_function,
)


class TorchNEGFTest(unittest.TestCase):

    def test_photon_dipole_scatterer_vacuum_limit(self):
        # If G(E) is constant over energy, the ±ħω shifts should not matter
        # (except for edge-handling, avoided here by choosing small ħω).
        #
        # Then Sigma^<(E) ~ (∫ dω J(ω) (2N(ω)+1)) * P G^<(E) P^†
        # where the "+1" is the vacuum fluctuation / spontaneous contribution.

        dtype = torch.complex64
        device = torch.device("cpu")

        E = 64
        N = 2
        energies = torch.linspace(-1.0, 1.0, E, device=device)

        # use a small frequency window
        frequencies = torch.tensor([0.05, 0.10, 0.15], device=device)
        W = torch.zeros_like(frequencies, dtype=torch.float32)

        g_lesser = (0.25j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)
        g_greater = (-0.75j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)

        p_xyz = torch.zeros((3, N, N), dtype=dtype, device=device)
        p_xyz[0] = torch.eye(N, dtype=dtype, device=device)

        # pure x polarization (rho = |x><x|).
        rho = torch.zeros((3, 3), dtype=dtype, device=device)
        rho[0, 0] = 1.0 + 0j

        # Use constant S_vac and set e/m=1 for a clean check.
        S_vac = torch.ones_like(frequencies, dtype=torch.float32)

        scatterer = PhotonDipoleScatterer(
            frequencies=frequencies,
            W=W,
            p_xyz=p_xyz,
            rho_pol=rho,
            S_vac=S_vac,
            hbar=1.0,
            e_charge=1.0,
            m0=1.0,
            eps0=1.0,
            c=1.0,
        )

        sigmaL, sigmaG = scatterer.correlation_self_energy(
            g_lesser=g_lesser,
            g_greater=g_greater,
            energies=energies,
        )

        domega = (frequencies[-1] - frequencies[0]).item()

        mid = E // 2
        self.assertTrue(torch.allclose(sigmaL[0, mid], g_lesser[0, mid] * domega, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(sigmaG[0, mid], g_greater[0, mid] * domega, atol=1e-5, rtol=1e-5))

    def test_einstein_phonon_scatterer_scaling(self):
        # check for constant g scaling:

        # For a constant G(E), the E+hw, E-hw shifts should not change the value at
        # interior points, and \Sigma^{</>} should reduce to a scalar factor:
        #   factor = g2 * (2N0 + 1)
        dtype = torch.complex64
        device = torch.device("cpu")

        E = 256
        N = 3
        energies = torch.linspace(-10.0, 10.0, E, device=device)

        omega0 = 0.2
        hbar = 1.0
        kT = 0.5
        g2 = 0.7

        gL = (0.25j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)
        gG = (-0.75j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)

        scatterer = EinsteinPhononScatterer(
            omega0=omega0,
            g2=g2,
            kT=kT,
            hbar=hbar,
        )

        sigmaL, sigmaG = scatterer.correlation_self_energy(
            g_lesser=gL,
            g_greater=gG,
            energies=energies,
        )

        N0 = 1.0 / (torch.exp(torch.tensor(hbar * omega0 / kT)) - 1.0)
        factor = g2 * (2 * N0 + 1.0)

        mid = E // 2
        self.assertTrue(torch.allclose(sigmaL[0, mid], factor * gL[0, mid], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(sigmaG[0, mid], factor * gG[0, mid], atol=1e-5, rtol=1e-5))

    def test_einstein_phonon_scatterer_sandwich(self):
        dtype = torch.complex64
        device = torch.device("cpu")

        E = 128
        N = 2
        energies = torch.linspace(-5.0, 5.0, E, device=device)

        omega0 = 0.1
        kT = 1.0
        g2 = 1.0

        gL = (0.5j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)
        gG = (-1.5j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)

        # M = 2I --> M G M^t = 4G
        M = (2.0 + 0j) * torch.eye(N, dtype=dtype, device=device)

        scatterer = EinsteinPhononScatterer(
            omega0=omega0,
            g2=g2,
            kT=kT,
            M=M,
            hbar=1.0,
        )

        sigmaL, sigmaG = scatterer.correlation_self_energy(
            g_lesser=gL,
            g_greater=gG,
            energies=energies,
        )

        N0 = 1.0 / (torch.exp(torch.tensor(omega0 / kT)) - 1.0)
        factor = (2 * N0 + 1.0) * 4.0

        mid = E // 2
        self.assertTrue(torch.allclose(sigmaL[0, mid], factor * gL[0, mid], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(sigmaG[0, mid], factor * gG[0, mid], atol=1e-5, rtol=1e-5))

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
        k_pts = torch.linspace(0, 1/a_lat, 100)
        
        H = (e_onsite * torch.eye(2)).view(1, 2, 2) + \
            (2*t_hop*torch.cos(2*torch.pi*k_pts*a_lat)).view(-1, 1, 1) * \
            torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]).view(1, 2, 2)

        energies = torch.linspace(-2, 2, 10)
        gf = greens_function(energies, H, eta=1e-5)
    
        self.assertTupleEqual(tuple(gf.shape), (100, 10, 2, 2))
