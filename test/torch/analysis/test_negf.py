import unittest

import numpy as np
import torch

from pyquinn.torch.analysis.negf import (
    EinsteinPhononScatterer,
    PhotonDipoleScatterer,
    surface_greens_function,
    SanchoRubioSolver,
    greens_function,
    TorchNEGFResult,
)


class TorchNEGFTest(unittest.TestCase):

    def test_current_matches_coherent_current_without_scatterers(self):
        # In the absence of scattering, Meir-Wingreen current should reduce
        # to the coherent (Landauer) current between two leads.

        dtype = torch.complex64
        device = torch.device("cpu")

        E = 2001
        energies = torch.linspace(-5.0, 5.0, E, device=device)

        # Single-level device with wide-band leads.
        e0 = 0.3
        gamma_L = 0.8
        gamma_R = 0.4

        # Lead self-energies: Sigma = -i * gamma / 2.
        se_L = (-0.5j * gamma_L) * torch.ones((E, 1, 1), dtype=dtype, device=device)
        se_R = (-0.5j * gamma_R) * torch.ones((E, 1, 1), dtype=dtype, device=device)

        lead_self_energies = torch.stack([se_L, se_R], dim=0).unsqueeze(1)  # [L=2, K=1, E, N=1, N=1]

        # Retarded GF: G^r = [E - e0 - sum Sigma]^-1
        sigma_tot = (se_L + se_R).view(1, E, 1, 1)  # [K=1, E, 1, 1]
        g_ret = 1.0 / ((energies - e0).to(dtype).view(1, E, 1, 1) - sigma_tot)

        # Fermi occupations for each lead.
        mu_L = 0.2
        mu_R = -0.2
        kT = 0.05
        f_L = 1.0 / (1.0 + torch.exp((energies - mu_L) / kT))
        f_R = 1.0 / (1.0 + torch.exp((energies - mu_R) / kT))
        lead_occ = torch.stack([f_L, f_R], dim=0)  # [L, E]

        # Correlation self-energies from leads:
        #   Sigma^< = i * sum_l [Gamma_l f_l]
        #   Sigma^> = -i * sum_l [Gamma_l (1 - f_l)]
        Gamma_L = gamma_L
        Gamma_R = gamma_R
        sigma_lesser = (1.0j * (Gamma_L * f_L + Gamma_R * f_R)).to(dtype).view(1, E, 1, 1)
        sigma_greater = (-1.0j * (Gamma_L * (1 - f_L) + Gamma_R * (1 - f_R))).to(dtype).view(1, E, 1, 1)

        g_lesser = g_ret @ sigma_lesser @ g_ret.mH
        g_greater = g_ret @ sigma_greater @ g_ret.mH

        result = TorchNEGFResult(
            energies=energies,
            hamiltonian=torch.tensor([[e0]], dtype=dtype, device=device),
            g_ret=g_ret,
            lead_self_energies=lead_self_energies,
            lead_occupation_probs=lead_occ,
            g_lesser=g_lesser,
            g_greater=g_greater,
        )

        I_coh = result.coherent_current(in_idx=0, out_idx=1, k_resolved=False)
        I_mw = result.current(in_idx=0, out_idx=1, k_resolved=False)

        self.assertTrue(torch.allclose(I_mw, I_coh, atol=1e-4, rtol=1e-4))

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

            g_numerical = surface_greens_function(
                energies=energies,
                hamiltonian_matrix=h,
                coupling_matrix=v,
                eta=eta,
                solver=SanchoRubioSolver(tol=1e-8),
            ).squeeze().detach().cpu().numpy()

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

    def test_occupied_ldos_and_dos(self):
        energies = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        K, E, N = 2, 3, 2

        g_ret = torch.zeros((K, E, N, N), dtype=torch.complex64)
        hamiltonian = torch.zeros((K, N, N), dtype=torch.complex64)

        g_lesser = torch.zeros((K, E, N, N), dtype=torch.complex64)
        # diagonal imaginary occupancy pattern
        # k=0
        g_lesser[0, :, 0, 0] = 1j * torch.tensor([1.0, 2.0, 3.0])
        g_lesser[0, :, 1, 1] = 1j * torch.tensor([2.0, 3.0, 4.0])
        # k=1
        g_lesser[1, :, 0, 0] = 1j * torch.tensor([2.0, 3.0, 4.0])
        g_lesser[1, :, 1, 1] = 1j * torch.tensor([3.0, 4.0, 5.0])

        result = TorchNEGFResult(
            energies=energies,
            hamiltonian=hamiltonian,
            g_ret=g_ret,
            g_lesser=g_lesser,
        )

        band_mult = 2.0
        pref = band_mult / (2 * torch.pi)

        diag_imag = torch.diagonal(g_lesser.imag, dim1=-2, dim2=-1)  # [K, E, N]
        expected_occ_ldos = pref * diag_imag.sum(dim=0) / K
        expected_occ_ldos_k = pref * diag_imag

        expected_occ_dos = pref * diag_imag.sum(dim=(-1, 0)) / K  # [E]
        expected_occ_dos_k = pref * diag_imag.sum(dim=-1)  # [K, E]

        occ_ldos = result.occupied_ldos(band_mult=band_mult, k_resolved=False)
        occ_ldos_k = result.occupied_ldos(band_mult=band_mult, k_resolved=True)
        occ_dos = result.occupied_dos(band_mult=band_mult, k_resolved=False)
        occ_dos_k = result.occupied_dos(band_mult=band_mult, k_resolved=True)

        self.assertTrue(torch.allclose(occ_ldos, expected_occ_ldos, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(occ_ldos_k, expected_occ_ldos_k, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(occ_dos, expected_occ_dos, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(occ_dos_k, expected_occ_dos_k, atol=1e-6, rtol=1e-6))

    def test_current(self):
        energies = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        L, K, E, N = 2, 2, 4, 1

        hamiltonian = torch.zeros((K, N, N), dtype=torch.complex64)
        g_ret = torch.zeros((K, E, N, N), dtype=torch.complex64)

        gamma0 = 1.0
        gamma1 = 2.0
        lead_self_energies = torch.zeros((L, K, E, N, N), dtype=torch.complex64)
        lead_self_energies[0, ...] = -0.5j * gamma0
        lead_self_energies[1, ...] = -0.5j * gamma1

        f0 = torch.tensor([0.9, 0.7, 0.3, 0.1], dtype=torch.float32)
        f1 = torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=torch.float32)
        lead_occ = torch.stack([f0, f1], dim=0)  # [L, E]

        a = torch.tensor(
            [[0.5, 0.7, 0.9, 1.1],
             [0.6, 0.8, 1.0, 1.2]],
            dtype=torch.float32,
        )  # [K, E]
        b = torch.tensor(
            [[1.1, 0.9, 0.7, 0.5],
             [1.2, 1.0, 0.8, 0.6]],
            dtype=torch.float32,
        )  # [K, E]

        g_lesser = torch.zeros((K, E, N, N), dtype=torch.complex64)
        g_greater = torch.zeros((K, E, N, N), dtype=torch.complex64)
        g_lesser[..., 0, 0] = 1j * a
        g_greater[..., 0, 0] = -1j * b

        result = TorchNEGFResult(
            energies=energies,
            hamiltonian=hamiltonian,
            g_ret=g_ret,
            lead_self_energies=lead_self_energies,
            lead_occupation_probs=lead_occ,
            g_lesser=g_lesser,
            g_greater=g_greater,
        )

        band_mult = 2.0
        pref = band_mult / (2 * torch.pi)

        # Mirror the current() convention for the in=0, out=1 pair:
        # jE = 0.5 * [(Gamma0*(f0*b - (1-f0)*a)) - (Gamma1*(f1*b - (1-f1)*a))]
        j_in = gamma0 * (f0.view(1, E) * b - (1 - f0).view(1, E) * a)
        j_out = gamma1 * (f1.view(1, E) * b - (1 - f1).view(1, E) * a)
        jE = 0.5 * (j_in - j_out)

        expected_k = torch.trapz(jE, energies, dim=-1) * pref
        expected = expected_k.mean()

        I = result.current(in_idx=0, out_idx=1, band_mult=band_mult, k_resolved=False)
        I_k = result.current(in_idx=0, out_idx=1, band_mult=band_mult, k_resolved=True)

        self.assertTrue(torch.allclose(I, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(I_k.squeeze(0), expected_k, atol=1e-6, rtol=1e-6))
