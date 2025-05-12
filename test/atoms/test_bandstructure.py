import unittest
import numpy as np
from scipy.integrate import cumulative_trapezoid

from pyquinn.atoms.bandstructure import BandStructure


class BandStructureTest(unittest.TestCase):

    def test_band_structure(self):

        dk = np.linspace(-1, 1, 5)
        Kx, Ky, Kz = np.meshgrid(dk, dk, dk)
        k_pts = np.stack(
            tuple(K.flatten() for K in (Kx, Ky, Kz)),
            axis=1)

        n_bands = 5
        n_sites = 2
        n_orbitals = 3
        n_spin = 2
        mu = 12.34

        energies = np.outer((k_pts**2).sum(-1), np.arange(n_bands))

        band_projections = np.linspace(0, 1, n_bands).reshape(1, n_bands)

        projections = np.broadcast_to(
            band_projections.reshape(1, n_bands, 1, 1, 1), 
            (k_pts.shape[0], n_bands, n_sites, n_orbitals, n_spin),
        )

        bs = BandStructure(
            energies=energies,
            k_pts=k_pts,
            projections=projections,
            chemical_potential=mu
        )

        self.assertTrue(np.allclose(bs.k_pts, k_pts))
        self.assertTrue(np.allclose(bs.energies, energies))
        self.assertTrue(np.allclose(bs.projections, projections))
        self.assertAlmostEqual(bs.chemical_potential, mu)

        k_pts_2 = np.zeros_like(k_pts)
        energies_2 = np.zeros_like(energies)
        projections_2 = np.ones_like(projections)
        mu_2 = 0.0
        
        bs.k_pts = k_pts_2
        bs.energies = energies_2
        bs.projections = projections_2
        bs.chemical_potential = mu_2

        self.assertTrue(np.allclose(bs.k_pts, k_pts_2))
        self.assertTrue(np.allclose(bs.energies, energies_2))
        self.assertTrue(np.allclose(bs.projections, projections_2))
        self.assertAlmostEqual(bs.chemical_potential, mu_2)

    def test_gaussian_dos(self):

        k = np.zeros((2, 3))
        energies = np.linspace(10, 100, 10).reshape(2, 5)
        sigma = 1.0

        bs = BandStructure(energies=energies, k_pts=k)
        
        weights = np.broadcast_to(
            np.array([1, -1]).reshape(1, 1, 2),
            (2, 5, 2)
        )

        E_dos = np.linspace(5, 100, 20)

        target_dos = np.ones(20)/(sigma*np.sqrt(np.pi)*k.shape[0])
        target_dos[::2] = 0

        gaussian_dos = bs.dos(
            E_dos, 
            smearing='gaussian', 
            sigma=sigma, 
            spin_mult=1.0)

        self.assertTrue(np.allclose(gaussian_dos, target_dos))

        proj_target_dos = np.outer(gaussian_dos, np.array([1, -1]))

        proj_gaussian_dos = bs.dos(
            E_dos, 
            smearing='gaussian', 
            sigma=sigma, 
            spin_mult=1.0,
            projections=weights)

        self.assertTrue(np.allclose(proj_gaussian_dos, proj_target_dos))

    def test_fermi_dirac_dos(self):

        k = np.zeros((2, 3))
        energies = np.linspace(10, 100, 10).reshape(2, 5)
        sigma = 0.1

        bs = BandStructure(energies=energies, k_pts=k)
        
        weights = np.broadcast_to(
            np.array([1, -1]).reshape(1, 1, 2),
            (2, 5, 2)
        )

        E_dos = np.linspace(5, 100, 20)

        target_dos = np.ones(20)/(4*sigma*k.shape[0])
        target_dos[::2] = 0

        fermi_dos = bs.dos(
            E_dos, 
            smearing='fermi-dirac', 
            sigma=sigma, 
            spin_mult=1.0)

        self.assertTrue(np.allclose(fermi_dos, target_dos))

        proj_target_dos = np.outer(fermi_dos, np.array([1, -1]))

        proj_fermi_dos = bs.dos(
            E_dos, 
            smearing='fermi-dirac', 
            sigma=sigma,
            spin_mult=1.0,
            projections=weights)

        self.assertTrue(np.allclose(proj_fermi_dos, proj_target_dos))

    def test_idos_extrema(self):
        
        energies = np.linspace(10, 100, 10).reshape(2, 5)
        k = np.zeros((2, 3))

        bs = BandStructure(energies=energies, k_pts=k)
        
        for smearing in bs._dos_smearing:
            self.assertAlmostEqual(bs.idos(-1e4, smearing=smearing), 0)
            self.assertAlmostEqual(bs.idos(1e4, smearing=smearing), 5)

    def test_idos(self):

        k = np.zeros((2, 3))
        energies = np.linspace(10, 100, 10).reshape(2, 5)

        bs = BandStructure(energies=energies, k_pts=k)

        E_dos = np.linspace(-30, 140, 2000)
        
        for smearing in bs._dos_smearing:
            dos = bs.dos(
                E_dos, 
                smearing=smearing,
                sigma=5.0,
                spin_mult=2.0)

            idos = bs.idos(
                E_dos, 
                smearing=smearing,
                sigma=5.0,
                spin_mult=2.0)

            trapz_idos = cumulative_trapezoid(dos, E_dos, initial=0)
            self.assertTrue(np.allclose(idos, trapz_idos, atol=1e-3))