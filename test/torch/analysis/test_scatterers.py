import unittest

import torch

from pyquinn.torch.analysis.negf import (
    ElasticDephasingScatterer,
    EinsteinPhononScatterer,
    ImpactIonizationScatterer,
    PhotonDipoleScatterer,
    correlation_self_energies_from_scatterers,
    einstein_phonon_self_energy,
    optical_dipole_self_energy,
)


class ScattererSyntaxTest(unittest.TestCase):

    def test_scatterer_sum_matches_manual(self):
        dtype = torch.complex64
        device = torch.device("cpu")

        E = 128
        N = 2
        energies = torch.linspace(-5.0, 5.0, E, device=device)

        gL = (0.3j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)
        gG = (-0.9j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)

        D = 0.5
        omega0 = 0.2
        kT = 0.8
        g2 = 0.4

        sc1 = ElasticDephasingScatterer(D)
        sc2 = EinsteinPhononScatterer(omega0=omega0, g2=g2, kT=kT, M=None, hbar=1.0)

        sigL, sigG = correlation_self_energies_from_scatterers(
            scatterers=[sc1, sc2],
            g_lesser=gL,
            g_greater=gG,
            energies=energies,
            kt=kT,
            dE=torch.diff(energies).mean(),
        )

        sigL2, sigG2 = einstein_phonon_self_energy(
            g_lesser=gL,
            g_greater=gG,
            energies=energies,
            omega0=omega0,
            g2=g2,
            kT=kT,
            M=None,
            hbar=1.0,
        )

        self.assertTrue(torch.allclose(sigL, D * gL + sigL2, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(sigG, D * gG + sigG2, atol=1e-5, rtol=1e-5))

    def test_impact_ionization_scatterer_runs(self):
        dtype = torch.complex64
        device = torch.device("cpu")

        E = 32
        N = 2
        energies = torch.linspace(-1.0, 1.0, E, device=device)

        # Simple diagonal correlation functions.
        gL = (0.1j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)
        gG = (-0.9j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)

        sc = ImpactIonizationScatterer(D_ionization=0.05, valence_cutoff=0.0)
        sigL, sigG = sc.correlation_self_energy(gL, gG, energies, dE=torch.diff(energies).mean())

        self.assertEqual(sigL.shape, gL.shape)
        self.assertEqual(sigG.shape, gG.shape)
        self.assertTrue(torch.isfinite(sigL.real).all())
        self.assertTrue(torch.isfinite(sigG.real).all())

    def test_impact_ionization_band_masks_orbital_cutoff(self):
        device = torch.device("cpu")
        energies = torch.tensor([-1.0, 0.0, 1.0], device=device)

        sc = ImpactIonizationScatterer(
            D_ionization=0.05,
            valence_cutoff=torch.tensor([-0.5, 0.5], device=device),
        )

        theta_c, theta_v = sc._band_masks(energies=energies, n_orbitals=2, lead_ndim=0)

        # At E=0: orbital 0 is conduction, orbital 1 is valence.
        # Inter-band matrix elements should be removed by cc/vv projections.
        expected_theta_c_e1 = torch.tensor(
            [[1.0, 0.0], [0.0, 0.0]],
            device=device,
        )
        expected_theta_v_e1 = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0]],
            device=device,
        )
        self.assertTrue(torch.equal(theta_c[1], expected_theta_c_e1))
        self.assertTrue(torch.equal(theta_v[1], expected_theta_v_e1))

    def test_impact_ionization_band_masks_invalid_cutoff_shape_raises(self):
        device = torch.device("cpu")
        energies = torch.tensor([-1.0, 0.0, 1.0], device=device)
        sc = ImpactIonizationScatterer(
            D_ionization=0.05,
            valence_cutoff=torch.zeros((2, 2), device=device),
        )

        with self.assertRaises(ValueError):
            sc._band_masks(energies=energies, n_orbitals=2, lead_ndim=0)

    def test_photon_dipole_scatterer_matches_function(self):
        dtype = torch.complex64
        device = torch.device("cpu")

        E = 64
        N = 2
        energies = torch.linspace(-1.0, 1.0, E, device=device)
        frequencies = torch.tensor([0.05, 0.10, 0.15], device=device)
        W = torch.zeros_like(frequencies, dtype=torch.float32)

        gL = (0.25j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)
        gG = (-0.75j) * torch.eye(N, dtype=dtype, device=device).view(1, 1, N, N).expand(1, E, N, N)

        p_xyz = torch.zeros((3, N, N), dtype=dtype, device=device)
        p_xyz[0] = torch.eye(N, dtype=dtype, device=device)

        rho = torch.zeros((3, 3), dtype=dtype, device=device)
        rho[0, 0] = 1.0 + 0j

        S_vac = torch.ones_like(frequencies, dtype=torch.float32)

        sc = PhotonDipoleScatterer(
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

        sigL1, sigG1 = sc.correlation_self_energy(gL, gG, energies)
        sigL2, sigG2 = optical_dipole_self_energy(
            g_lesser=gL,
            g_greater=gG,
            energies=energies,
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

        self.assertTrue(torch.allclose(sigL1, sigL2, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(sigG1, sigG2, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
