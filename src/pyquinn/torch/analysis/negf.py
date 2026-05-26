from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from pyquinn.torch.types import TorchAtomGraph
from pyquinn.torch.models import TorchBandStructureModel

from pyquinn.io.stdout import print_overwrite

from math import ceil
import os
from glob import glob

class DysonSolver(ABC):
    """Abstract solver that computes the Dyson equation retarded Green's function."""

    @abstractmethod
    def solve(
            self,
            energies: torch.Tensor,
            hamiltonian_matrix: torch.Tensor,
            self_energies: torch.Tensor | None = None,
            overlap_matrix: torch.Tensor | None = None,
            eta: float = 1e-5) -> torch.Tensor:
        raise NotImplementedError

class DenseDysonSolver(DysonSolver):
    """Dense Dyson solver that inverts the perturbed Hamiltonian directly."""

    def build_matrix(
            self,
            energies: torch.Tensor,
            hamiltonian_matrix: torch.Tensor,
            self_energies: torch.Tensor | None = None,
            overlap_matrix: torch.Tensor | None = None,
            eta: float = 1e-5) -> torch.Tensor:
        """Builds the Dyson matrix ``A(E) = (E S + iηI) - (H + Σ)``.

        This is the matrix whose inverse is the retarded Green's function.
        Exposed so higher-level solvers (e.g. SCBA) can avoid forming
        ``G^{-1}`` by explicitly inverting an already-inverted matrix.
        """

        E = energies.view(-1, 1, 1)
        H_site = hamiltonian_matrix.unsqueeze(-3)
        dtype = hamiltonian_matrix.dtype
        device = hamiltonian_matrix.device
        I = torch.eye(hamiltonian_matrix.shape[-1], dtype=dtype, device=device)
        S = I if overlap_matrix is None else overlap_matrix

        H_se = 0 if self_energies is None else self_energies.sum(0)

        return (E * S + 1.j * eta * I) - (H_site + H_se)

    def solve(
            self,
            energies: torch.Tensor,
            hamiltonian_matrix: torch.Tensor,
            self_energies: torch.Tensor | None = None,
            overlap_matrix: torch.Tensor | None = None,
            eta: float = 1e-5) -> torch.Tensor:
        A_ret = self.build_matrix(
            energies=energies,
            hamiltonian_matrix=hamiltonian_matrix,
            self_energies=self_energies,
            overlap_matrix=overlap_matrix,
            eta=eta,
        )
        return torch.linalg.inv(A_ret)


class Scatterer(ABC):
    """Interface for SCBA-style scattering models that contribute NEGF self-energies.

    A scatterer maps the current electronic correlation Green's functions
    :math:`G^{<}(E)` and :math:`G^{>}(E)` onto correlation self-energies
    :math:`\\Sigma^{<}(E)` and :math:`\\Sigma^{>}(E)`.

    The steady-state solver can then iterate these self-energies to
    self-consistency.
    """

    @abstractmethod
    def correlation_self_energy(
            self,
            g_lesser: torch.Tensor,
            g_greater: torch.Tensor,
            energies: torch.Tensor,
            **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ElasticDephasingScatterer(Scatterer):
    """Elastic dephasing model: :math:`\\Sigma^{</>}(E) = D\\,G^{</>}(E)`."""

    def __init__(self, D: float | torch.Tensor):
        self.D = D

    def correlation_self_energy(
            self,
            g_lesser: torch.Tensor,
            g_greater: torch.Tensor,
            energies: torch.Tensor,
            **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.D * g_lesser, self.D * g_greater


class ImpactIonizationScatterer(Scatterer):
    """Impact ionization correlation self-energies built from electron/hole GFs."""

    def __init__(
            self,
            D_ionization: float | torch.Tensor,
            valence_cutoff: float | torch.Tensor = 0.0):
        self.D_ionization = D_ionization
        self.valence_cutoff = valence_cutoff

    def _band_masks(
            self,
            energies: torch.Tensor,
            n_orbitals: int,
            lead_ndim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Build conduction/valence masks in orbital basis.

        `valence_cutoff` may be a scalar or a 1D tensor with one cutoff per
        orbital. Masks are built as outer products so intra-band couplings are
        retained (`cc` and `vv`) while inter-band couplings (`cv`/`vc`) are
        removed.
        """
        cutoff = torch.as_tensor(
            self.valence_cutoff,
            device=energies.device,
            dtype=energies.dtype,
        )

        if cutoff.ndim == 0:
            cutoff = cutoff.expand(n_orbitals)
        elif cutoff.ndim == 1:
            if cutoff.numel() != n_orbitals:
                raise ValueError(
                    'Expected 1D `valence_cutoff` length to match number of orbitals '
                    f'({n_orbitals}); got {cutoff.numel()}.'
                )
        else:
            raise ValueError(
                'Expected `valence_cutoff` to be a scalar or 1D tensor with '
                f'length {n_orbitals}; got shape {tuple(cutoff.shape)}.'
            )

        # Orbital-resolved conduction mask c_i(E), then project into matrix
        # space as c_i(E) c_j(E) (same for valence).
        c_orb = (energies.view(-1, 1) > cutoff.view(1, -1)).to(energies.dtype)
        v_orb = 1 - c_orb

        theta_c = c_orb.unsqueeze(-1) * c_orb.unsqueeze(-2)
        theta_v = v_orb.unsqueeze(-1) * v_orb.unsqueeze(-2)

        if lead_ndim > 0:
            theta_c = theta_c.view((1,) * lead_ndim + theta_c.shape)
            theta_v = theta_v.view((1,) * lead_ndim + theta_v.shape)

        return theta_c, theta_v

    def correlation_self_energy(
            self,
            g_lesser: torch.Tensor,
            g_greater: torch.Tensor,
            energies: torch.Tensor,
            **kwargs) -> tuple[torch.Tensor, torch.Tensor]:

        dE = kwargs.get('dE', None)
        if dE is None:
            if energies.numel() < 2:
                raise ValueError('Expected `energies` to have length >= 2.')
            dE = torch.diff(energies).mean()

        lead_ndim = g_lesser.ndim - 3
        n_orbitals = g_lesser.shape[-1]
        theta_c, theta_v = self._band_masks(
            energies=energies,
            n_orbitals=n_orbitals,
            lead_ndim=lead_ndim,
        )

        g_maj = theta_c * g_lesser
        g_min = theta_v * g_greater

        sigma_lesser = impact_ionization_self_energy(g_min, g_maj, self.D_ionization, dE)
        sigma_greater = impact_ionization_self_energy(g_maj, g_min, self.D_ionization, dE)

        return sigma_lesser, sigma_greater


class EinsteinPhononScatterer(Scatterer):
    """Single Einstein (optical) phonon bath scattering model."""

    def __init__(
            self,
            omega0: float | torch.Tensor,
            g2: float | torch.Tensor,
            kT: float | torch.Tensor | None = None,
            M: torch.Tensor | None = None,
            hbar: float | torch.Tensor = 1.0):
        
        self.omega0 = omega0
        self.g2 = g2
        self.kT = kT
        self.M = M
        self.hbar = hbar

    def correlation_self_energy(
            self,
            g_lesser: torch.Tensor,
            g_greater: torch.Tensor,
            energies: torch.Tensor,
            **kwargs) -> tuple[torch.Tensor, torch.Tensor]:

        kT_eff = self.kT
        if kT_eff is None:
            kT_eff = kwargs.get('kt', None)
        if kT_eff is None:
            raise ValueError('EinsteinPhononScatterer requires `kT` (or pass `kt=...` in solver kwargs).')

        return einstein_phonon_self_energy(
            g_lesser=g_lesser,
            g_greater=g_greater,
            energies=energies,
            omega0=self.omega0,
            g2=self.g2,
            kT=kT_eff,
            M=self.M,
            hbar=self.hbar,
        )


class PhotonDipoleScatterer(Scatterer):
    """Photon bath scattering via the velocity-gauge (momentum) coupling."""

    def __init__(
            self,
            frequencies: torch.Tensor,
            W: torch.Tensor,
            p_xyz: torch.Tensor,
            rho_pol: torch.Tensor | None = None,
            S_vac: torch.Tensor | None = None,
            k_hat: torch.Tensor | None = None,
            hbar: float | torch.Tensor = 1.0,
            e_charge: float | torch.Tensor = 1.0,
            m0: float | torch.Tensor = 1.0,
            eps0: float | torch.Tensor = 1.0,
            c: float | torch.Tensor = 1.0):
        
        self.frequencies = frequencies
        self.W = W
        self.p_xyz = p_xyz
        self.rho_pol = rho_pol
        self.S_vac = S_vac
        self.k_hat = k_hat
        self.hbar = hbar
        self.e_charge = e_charge
        self.m0 = m0
        self.eps0 = eps0
        self.c = c

    def correlation_self_energy(
            self,
            g_lesser: torch.Tensor,
            g_greater: torch.Tensor,
            energies: torch.Tensor,
            **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return optical_dipole_self_energy(
            g_lesser=g_lesser,
            g_greater=g_greater,
            energies=energies,
            frequencies=self.frequencies,
            W=self.W,
            p_xyz=self.p_xyz,
            rho_pol=self.rho_pol,
            S_vac=self.S_vac,
            k_hat=self.k_hat,
            hbar=self.hbar,
            e_charge=self.e_charge,
            m0=self.m0,
            eps0=self.eps0,
            c=self.c,
        )

def correlation_self_energies_from_scatterers(
        scatterers: list[Scatterer],
        g_lesser: torch.Tensor,
        g_greater: torch.Tensor,
        energies: torch.Tensor,
        **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Sums :math:`\\Sigma^{<,>}` contributions from a list of scatterers."""
    sigL = torch.zeros_like(g_lesser)
    sigG = torch.zeros_like(g_greater)
    for sc in scatterers:
        sL, sG = sc.correlation_self_energy(
            g_lesser=g_lesser,
            g_greater=g_greater,
            energies=energies,
            **kwargs,
        )
        sigL.add_(sL)
        sigG.add_(sG)
    return sigL, sigG

def greens_function(
        energies: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        self_energies: torch.Tensor | None = None,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5,
        dyson_solver: DysonSolver | None = None) -> torch.Tensor:
    """Computes the retarded Green's function of a Hamiltonian represented in a 
    (possibly non-orthogonal) basis with (optional) self-energy contributions.

    Args:
        energies (torch.Tensor): The energies at which the Green's function 
            should be evaluated (shape: [E]).
        hamiltonian_matrix (torch.Tensor): The Hamiltonian matrix of 
            the system (shape: [..., N, N]).
        self_energies (torch.Tensor): A stack of `M` self-energy perturbations
            to the Hamiltonian (shape: [M, N, N]).
        overlap_matrix (torch.Tensor | None, optional): The overlap matrix
            of the Hamiltonian basis if the basis is non-orthogonal 
            (shape: [..., N, N]). Defaults to None.
        eta (float, optional): Broadening parameter of the Green's function. 
            Increasing this parameter makes the Green's function 
            well-conditioned at the cost of broadening the energy 
            spectrum. Defaults to 1e-5.
        dyson_solver (DysonSolver | None, optional):
            Custom Dyson solver used to build the Green's function. Defaults
            to a dense matrix inverse implementation.

    Returns:
        torch.Tensor: The perturbed Green's function of the Hamiltonian
            (shape: [..., E, N, N]).
    """

    solver = DenseDysonSolver() if dyson_solver is None else dyson_solver
    return solver.solve(
        energies=energies,
        hamiltonian_matrix=hamiltonian_matrix,
        self_energies=self_energies,
        overlap_matrix=overlap_matrix,
        eta=eta,
    )


def impact_ionization_self_energy(
        g_min: torch.Tensor, 
        g_maj: torch.Tensor,
        D: float | torch.Tensor,
        dE: float | torch.Tensor) -> torch.Tensor:
    """
    Computes the self-energy correlation function due to impact ionization for
    a given majority (`g_maj`) and minority (`g_min`) carrier. These
    carrier functions correspond to the lesser (electron) or greater (hole)
    Green's functions restricted to carrier conduction states 
    (e.g., valence bands for holes and conduction bands for electrons).
    
    Args:
        g_min (torch.Tensor): Minority charge carrier correlation function (shape: [..., E, N, N]).
        g_maj (torch.Tensor): Majority charge carrier correlation function (shape: [..., E, N, N]).
        D (float | torch.Tensor): On-site impact ionization coefficient (shape: [..., E, N, N]).
        dE (float | torch.Tensor): Spacing of uniform energy evaluation points.

    Returns:
        torch.Tensor: The self-energy correlation function due 
            to majority carrier ionization.
    """

    # Here the convolution is performed along the energy axis (dim=-3).
    n_E = g_min.shape[-3]
    g_min_pad = torch.nn.functional.pad(g_min, (0, 0, 0, 0, 0, 2*n_E))
    g_maj_pad = torch.nn.functional.pad(g_maj, (0, 0, 0, 0, 0, 2*n_E))

    f_min = torch.fft.fft(g_min_pad, dim=-3)
    f_maj = torch.fft.fft(g_maj_pad, dim=-3)

    f_conv = f_min.conj() @ f_maj @ f_maj

    # self_energy = d0 * torch.fft.ifft(f_conv, dim=-3)[:N, :, :] * (dE**2)
    ifft_conv = torch.fft.ifft(f_conv, dim=-3)
    self_energy = D * ifft_conv[..., :n_E, :, :] * (dE**2)

    return self_energy


def einstein_phonon_self_energy(
        g_lesser: torch.Tensor,
        g_greater: torch.Tensor,
        energies: torch.Tensor,
        omega0: float | torch.Tensor,
        g2: float | torch.Tensor,
        kT: float | torch.Tensor,
        M: torch.Tensor | None = None,
        hbar: float | torch.Tensor = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """SCBA self-energy for a single (Einstein) optical phonon mode.

    This is a lightweight inelastic relaxation model that avoids computing a
    full phonon spectrum/dispersion. It is often sufficient to introduce
    energy relaxation/cooling in steady-state NEGF iterations.

    Conventions:
    
    - `energies` is a 1D, (approximately) uniformly-spaced grid (shape: [E]).
    - `g_lesser/g_greater` are electronic :math:`G^{</>}` on the same grid
      (shape: ``[..., E, N, N]``).
    - `omega0` is the phonon angular frequency (>0). The phonon energy is
      :math:`\\hbar\\omega_0`.
    - `kT` is the lattice thermal energy (same units as `energies`).
    - `M` is an optional electron-phonon coupling operator in the same basis
      (shape: [..., N, N] or [N, N]). If omitted, a scalar local coupling is
      assumed and the operator sandwich reduces to
      :math:`G^{</>}(E\\pm\\hbar\\omega_0)`.

    Bose factor:

    .. math::

        N_0 = \\frac{1}{\\exp\\left(\\frac{\\hbar\\omega_0}{kT}\\right) - 1}

    Positive-frequency SCBA correlation self-energies:

    .. math::

        \\Sigma^{<}_{\\mathrm{ph}}(E) = g^2\\Big[
            (N_0 + 1)\\, M\\, G^{<}(E+\\hbar\\omega_0)\\, M^{\\dagger}
            + N_0\\, M\\, G^{<}(E-\\hbar\\omega_0)\\, M^{\\dagger}
        \\Big]

    .. math::

        \\Sigma^{>}_{\\mathrm{ph}}(E) = g^2\\Big[
            (N_0 + 1)\\, M\\, G^{>}(E-\\hbar\\omega_0)\\, M^{\\dagger}
            + N_0\\, M\\, G^{>}(E+\\hbar\\omega_0)\\, M^{\\dagger}
        \\Big]

    Returns:
        (sigma_lesser, sigma_greater): phonon-induced correlation self-energies
        (shape: [..., E, N, N] each).
    """

    def _shift_energy_linear_uniform(
            g: torch.Tensor,
            energies_1d: torch.Tensor,
            shift: torch.Tensor) -> torch.Tensor:
        # Returns g(E + shift) sampled onto the original `energies` grid.
        # Out-of-range values are set to zero.
        if energies_1d.ndim != 1:
            raise ValueError('Expected `energies` to be 1D.')
        if energies_1d.numel() < 2:
            raise ValueError('Expected `energies` to have length >= 2.')

        dE = energies_1d[1] - energies_1d[0]
        if torch.diff(energies_1d).std() > 1e-6 * (energies_1d.max() - energies_1d.min()):
            raise ValueError('Expected `energies` to be (approximately) uniformly spaced.')

        E0 = energies_1d[0]
        E_min = energies_1d[0]
        E_max = energies_1d[-1]

        E_shift = energies_1d + shift
        valid = (E_shift >= E_min) & (E_shift <= E_max)

        idx_f = (E_shift - E0) / dE
        idx0 = torch.floor(idx_f).to(torch.int64)
        idx0 = torch.clamp(idx0, 0, energies_1d.numel() - 2)
        idx1 = idx0 + 1
        frac = (idx_f - idx0.to(idx_f.dtype)).to(g.dtype)

        leading_shape = g.shape[:-3]
        E = g.shape[-3]
        N = g.shape[-1]
        g_flat = g.reshape((-1, E, N * N))

        g0 = g_flat[:, idx0, :]
        g1 = g_flat[:, idx1, :]
        g_out = (1 - frac).view(1, -1, 1) * g0 + frac.view(1, -1, 1) * g1
        g_out = g_out * valid.to(g_out.dtype).view(1, -1, 1)

        return g_out.reshape(leading_shape + (E, N, N))

    def _sandwich(GE: torch.Tensor) -> torch.Tensor:
        if M is None:
            return GE
        Mop = M
        if Mop.shape[-2:] != GE.shape[-2:]:
            raise ValueError('Expected `M` to have shape [..., N, N] matching `G`.')
        return Mop.unsqueeze(-3) @ GE @ Mop.mH.unsqueeze(-3)

    if energies.ndim != 1:
        raise ValueError('Expected `energies` to have shape [E].')
    if g_lesser.shape != g_greater.shape:
        raise ValueError('Expected `g_lesser` and `g_greater` to have the same shape.')
    if g_lesser.shape[-3] != energies.numel():
        raise ValueError('Expected g_* energy dimension to match len(energies).')
    if g_lesser.shape[-1] != g_lesser.shape[-2]:
        raise ValueError('Expected g_* to have shape [..., E, N, N].')

    # Bose occupation at the lattice temperature (make these tensors so we can
    # safely use torch.exp even when the inputs are plain floats).
    dev = energies.device
    dt = energies.dtype
    dE0 = torch.as_tensor(hbar, device=dev, dtype=dt) * torch.as_tensor(omega0, device=dev, dtype=dt)
    kT_t = torch.as_tensor(kT, device=dev, dtype=dt)

    # prevent divide-by-zero at kT ~ 0 by clamping.
    kT_safe = torch.clamp(kT_t, min=1e-30)

    N0 = 1.0 / (torch.exp(dE0 / kT_safe) - 1.0)
    N0 = N0.to(dtype=g_lesser.real.dtype)

    # Energy shifts.
    GL_plus = _sandwich(_shift_energy_linear_uniform(g_lesser, energies, +dE0))
    GL_minus = _sandwich(_shift_energy_linear_uniform(g_lesser, energies, -dE0))
    GG_plus = _sandwich(_shift_energy_linear_uniform(g_greater, energies, +dE0))
    GG_minus = _sandwich(_shift_energy_linear_uniform(g_greater, energies, -dE0))

    sigma_lesser = g2 * ((N0 + 1) * GL_plus + N0 * GL_minus)
    sigma_greater = g2 * ((N0 + 1) * GG_minus + N0 * GG_plus)

    return sigma_lesser, sigma_greater


def optical_dipole_self_energy(
        g_lesser: torch.Tensor, 
        g_greater: torch.Tensor,
        energies: torch.Tensor,
        frequencies: torch.Tensor,
        W: torch.Tensor,
        p_xyz: torch.Tensor,
        rho_pol: torch.Tensor | None = None,
        S_vac: torch.Tensor | None = None,
        k_hat: torch.Tensor | None = None,
        hbar: float | torch.Tensor = 1.0,
        e_charge: float | torch.Tensor = 1.0,
        m0: float | torch.Tensor = 1.0,
        eps0: float | torch.Tensor = 1.0,
        c: float | torch.Tensor = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Photon (optical) self-energy in SCBA using the velocity gauge.

    This routine is meant to be "straightforward": you provide the electronic
    lesser/greater Green's functions and a momentum operator in your working
    orbital basis; the function returns the corresponding photon-induced
    correlation self-energies :math:`\\Sigma^{<,>}_{ph}(E)`.

    The photon polarization state can be supplied as a 3x3 density matrix
    `rho_pol` that weights correlations between the Cartesian field components.
    This allows modeling tunable polarization decoherence as a mixture between
    pure states (off-diagonal coherence) and incoherent mixtures (diagonal).

    The polarization contraction used in this implementation is:

    .. math::
        \\mathcal{P}_{\\rho}[G] \\,=\\, \\sum_{i,j\\in\\{x,y,z\\}} \\rho_{ij}\\, p_i\\, G\\, p_j^{\\dagger}.

    Important conventions/assumptions:

    - `W(omega)` is interpreted as a photon occupation number :math:`N(\\omega)`
      (dimensionless). If you have an intensity/irradiance spectrum instead,
      it must be converted to an occupation spectrum before calling this.
    - `energies` must be uniformly spaced for the internal linear interpolation.
    - Free-space vacuum fluctuations are included via the
      :math:`N(\\omega)+1` terms; i.e. emission persists even when `W == 0`.

    Correlation self-energies (for :math:`\\omega>0`):

    .. math::
        \\Sigma^{<}_{ph}(E) = \\int_0^{\\infty} d\\omega\\, J(\\omega)\\Big[
            N(\\omega)\\, \\mathcal{P}_{\\rho}[G^{<}(E-\\hbar\\omega)]
            + (N(\\omega)+1)\\, \\mathcal{P}_{\\rho}[G^{<}(E+\\hbar\\omega)]
        \\Big]

    .. math::
        \\Sigma^{>}_{ph}(E) = \\int_0^{\\infty} d\\omega\\, J(\\omega)\\Big[
            (N(\\omega)+1)\\, \\mathcal{P}_{\\rho}[G^{>}(E-\\hbar\\omega)]
            + N(\\omega)\\, \\mathcal{P}_{\\rho}[G^{>}(E+\\hbar\\omega)]
        \\Big]

    Args:
        g_lesser: Electronic lesser Green's function (shape: [..., E, N, N]).
        g_greater: Electronic greater Green's function (shape: [..., E, N, N]).
        energies: Energy grid (shape: [E]), assumed uniform spacing.
        frequencies: Positive frequency grid (shape: [F]).
        W: Photon occupation N(omega) (shape: [F]) or broadcastable to [F].
        p_xyz: Momentum operator components (shape: [..., 3, N, N] or [3, N, N]).
        rho_pol: 3x3 polarization density matrix (shape: [3, 3]). If None,
            defaults to an unpolarized state (identity/3), optionally projected
            to the transverse subspace if `k_hat` is given.
        S_vac: Photon spectral density / LDOS weighting (shape: [F]). If None,
            uses a simple free-space scaling :math:`\\propto \\omega^3`.
        k_hat: Optional propagation direction unit vector (shape: [3]). If
            provided and `rho_pol` is None, an unpolarized transverse density
            matrix `(I - k_hat k_hat^T)/2` is used.
        hbar, e_charge, m0, eps0, c: Physical constants / scalings.

    Returns:
        (sigma_lesser, sigma_greater): Photon-induced self-energies
        (shape: [..., E, N, N] each).
    """

    def _shift_energy_linear_uniform(
            g: torch.Tensor,
            energies_1d: torch.Tensor,
            shift: torch.Tensor) -> torch.Tensor:
        # Returns g(E + shift) sampled onto the original `energies` grid.
        # Out-of-range values are set to zero.
        if energies_1d.ndim != 1:
            raise ValueError('Expected `energies` to be 1D.')
        if energies_1d.numel() < 2:
            raise ValueError('Expected `energies` to have length >= 2.')

        dE = energies_1d[1] - energies_1d[0]
        if torch.diff(energies_1d).std() > 1e-6 * (energies_1d.max() - energies_1d.min()):
            raise ValueError('Expected `energies` to be (approximately) uniformly spaced.')

        E0 = energies_1d[0]
        E_min = energies_1d[0]
        E_max = energies_1d[-1]

        # We want g_shifted[E_idx] = g(Energies[E_idx] + shift)
        E_shift = energies_1d + shift
        valid = (E_shift >= E_min) & (E_shift <= E_max)

        # Fractional index in the original grid.
        idx_f = (E_shift - E0) / dE
        idx0 = torch.floor(idx_f).to(torch.int64)
        idx0 = torch.clamp(idx0, 0, energies_1d.numel() - 2)
        idx1 = idx0 + 1
        frac = (idx_f - idx0.to(idx_f.dtype)).to(g.dtype)

        # Flatten everything except energy, and flatten (N,N) -> (NN) to gather easily.
        leading_shape = g.shape[:-3]
        E = g.shape[-3]
        N = g.shape[-1]
        g_flat = g.reshape((-1, E, N * N))

        g0 = g_flat[:, idx0, :]  # [B, E, NN]
        g1 = g_flat[:, idx1, :]
        g_out = (1 - frac).view(1, -1, 1) * g0 + frac.view(1, -1, 1) * g1

        # Zero out out-of-range energies.
        g_out = g_out * valid.to(g_out.dtype).view(1, -1, 1)

        return g_out.reshape(leading_shape + (E, N, N))

    def _polarization_sandwich(
            gE: torch.Tensor,
            p_xyz_op: torch.Tensor,
            rho: torch.Tensor) -> torch.Tensor:
        # Computes sum_{i,j} rho[i,j] * p_i * gE * p_j^\dagger
        out = torch.zeros_like(gE)
        for i in range(3):
            p_i = p_xyz_op[..., i, :, :].unsqueeze(-3)  # [..., 1, N, N]
            left = p_i @ gE
            for j in range(3):
                coeff = rho[i, j]
                if torch.is_tensor(coeff) and coeff.ndim != 0:
                    raise ValueError('Expected `rho_pol` entries to be scalars.')
                p_j = p_xyz_op[..., j, :, :].mH.unsqueeze(-3)
                out = out + coeff * (left @ p_j)
        return out

    # Basic validation.
    if energies.ndim != 1:
        raise ValueError('Expected `energies` to have shape [E].')
    if frequencies.ndim != 1:
        raise ValueError('Expected `frequencies` to have shape [F].')
    if g_lesser.shape != g_greater.shape:
        raise ValueError('Expected `g_lesser` and `g_greater` to have the same shape.')
    if g_lesser.shape[-3] != energies.numel():
        raise ValueError('Expected g_* energy dimension to match len(energies).')
    if g_lesser.shape[-1] != g_lesser.shape[-2]:
        raise ValueError('Expected g_* to have shape [..., E, N, N].')

    # Normalize / broadcast momentum operator shape to [..., 3, N, N]
    if p_xyz.ndim == 3:
        # [3, N, N] -> [1..., 3, N, N] broadcast over g leading dims
        p_xyz_op = p_xyz
    elif p_xyz.ndim >= 4:
        p_xyz_op = p_xyz
    else:
        raise ValueError('Expected `p_xyz` to have shape [3, N, N] or [..., 3, N, N].')
    if p_xyz_op.shape[-3] != 3:
        raise ValueError('Expected `p_xyz[..., 3, N, N]` (3 Cartesian components).')

    # Make rho_pol.
    device = energies.device
    dtype = g_lesser.dtype
    if rho_pol is None:
        if k_hat is not None:
            if k_hat.shape != (3,):
                raise ValueError('Expected `k_hat` to have shape [3].')
            k = k_hat.to(device=device, dtype=torch.float32)
            k = k / (k.norm() + 1e-30)
            I3 = torch.eye(3, device=device, dtype=torch.float32)
            rho = (I3 - torch.outer(k, k)) / 2.0
        else:
            rho = torch.eye(3, device=device, dtype=torch.float32) / 3.0
        rho = rho.to(device=device, dtype=dtype)
    else:
        if rho_pol.shape != (3, 3):
            raise ValueError('Expected `rho_pol` to have shape [3, 3].')
        rho = rho_pol.to(device=device, dtype=dtype)

    # Sort frequencies (and corresponding W, S_vac).
    freqs, sort_idx = torch.sort(frequencies)
    W = W.to(device=device, dtype=dtype).reshape(-1)
    if W.numel() == 1:
        W = W.expand(freqs.numel())
    elif W.numel() != freqs.numel():
        raise ValueError('Expected `W` to have shape [F] or be a scalar.')
    W = W[sort_idx]

    if S_vac is None:
        # Simple free-space scaling. Users can pass a more faithful LDOS if needed.
        S = hbar * freqs**3 / (6 * torch.pi * eps0 * c**3)
    else:
        S = S_vac.to(device=device, dtype=dtype).reshape(-1)
        if S.numel() == 1:
            S = S.expand(freqs.numel())
        elif S.numel() != freqs.numel():
            raise ValueError('Expected `S_vac` to have shape [F] or be a scalar.')
        S = S[sort_idx]

    # Coupling strength prefactor in the velocity gauge.
    J = (e_charge / m0) ** 2 * S

    sigma_lesser = torch.zeros_like(g_lesser)
    sigma_greater = torch.zeros_like(g_greater)

    prev_less = None
    prev_great = None
    prev_w = None

    for i in range(freqs.numel()):
        w = freqs[i]
        Nw = W[i]
        Jw = J[i]

        dE = hbar * w

        # G(E - ħω) and G(E + ħω)
        gL_minus = _shift_energy_linear_uniform(g_lesser, energies, -dE)
        gL_plus = _shift_energy_linear_uniform(g_lesser, energies, +dE)
        gG_minus = _shift_energy_linear_uniform(g_greater, energies, -dE)
        gG_plus = _shift_energy_linear_uniform(g_greater, energies, +dE)

        # SCBA correlation self-energies:
        I_less = Jw * (
            Nw * _polarization_sandwich(gL_minus, p_xyz_op, rho) +
            (Nw + 1) * _polarization_sandwich(gL_plus, p_xyz_op, rho)
        )
        I_great = Jw * (
            (Nw + 1) * _polarization_sandwich(gG_minus, p_xyz_op, rho) +
            Nw * _polarization_sandwich(gG_plus, p_xyz_op, rho)
        )

        if prev_less is not None:
            dw = (w - prev_w).to(dtype)
            sigma_lesser = sigma_lesser + 0.5 * (prev_less + I_less) * dw
            sigma_greater = sigma_greater + 0.5 * (prev_great + I_great) * dw

        prev_less = I_less
        prev_great = I_great
        prev_w = w

    return sigma_lesser, sigma_greater

def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """Computes the Hilbert transform of a Greens function operator in the energy basis

    Args:
        x (torch.Tensor): Broadening matrix operator (shape: [..., E, N, N]).

    Returns:
        torch.Tensor: Transformed operator (shape: [..., E, N, N]).

    Notes:
        This uses the standard FFT-based discrete Hilbert transform

            H[x] = IFFT( -i * sign(f) * FFT(x) )

        along the energy axis. This corresponds to a *periodic* discrete
        Hilbert transform on the sampled grid. If you need a non-periodic
        principal-value transform on a truncated energy interval, consider
        padding/windowing the input before applying this routine.
    """
    
    X = torch.fft.fft(x, dim=-3)
    n = x.shape[-3]

    freqs = torch.fft.fftfreq(n, device=X.device)
    sgn = torch.sign(freqs)

    H = torch.fft.ifft(-1j * sgn.view(-1, 1, 1) * X, dim=-3)
    return H


def surface_greens_function(
        energies: torch.Tensor,
        coupling_matrix: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5,
        solver: 
            SurfaceGreensFunctionSolver | None = None,        
        ) -> torch.Tensor:
    """Numerically solves for the surface Green's function of a semi-infinite 
    boundary using a surface Green's function solver (defaults to the
    Sancho-Rubio method.)

    .. warning::
            This function does not currently support backward passes,
            due to potential numerical instabilities. If the method
            does not converge, try increasing the `eta` parameter.

    Args:
        energies (torch.Tensor): The energies at which the Green's function 
            should be evaluated (shape: [E]).
        coupling_matrix (torch.Tensor): The coupling matrix between 
            the Hamiltonian cell and the cell's nearest neighbor along the
            semi-infinite direction (shape: [..., N, N]).
        hamiltonian_matrix (torch.Tensor): The Hamiltonian matrix of 
            a unit cell of system along the semi-infinite direction 
            (shape: [..., N, N]).
        overlap_matrix (torch.Tensor | None, optional): The overlap matrix
            of the Hamiltonian basis if the basis is non-orthogonal 
            (shape: [..., N, N]). Defaults to None.
        eta (float, optional): Broadening parameter of the Green's function. 
            Increasing this parameter makes the method converge faster
            at the cost of broadening the energy spectrum. Defaults to 1e-5.
        max_iter (int, optional): Maximum number of solver iterations. 
            Defaults to 1024.
        tol (float, optional): Error tolerance of solver. Defaults to 1e-6.

    Returns:
        torch.Tensor: The surface Green's function of the semi-infinite
            Hamiltonian (shape: [..., E, N, N])
    """
    solver = SanchoRubioSolver() if solver is None else solver

    return solver.solve(
        energies=energies,
        coupling_matrix=coupling_matrix,
        hamiltonian_matrix=hamiltonian_matrix,
        overlap_matrix=overlap_matrix,
        eta=eta,
    )
    
class SurfaceGreensFunctionSolver(ABC):

    @abstractmethod
    def solve(
        self,
        energies: torch.Tensor,
        coupling_matrix: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5):
        
        raise NotImplementedError

class SanchoRubioSolver(SurfaceGreensFunctionSolver):

    def __init__(
        self,
        max_iter: int = 10000,
        tol: float = 1e-6,
        eta_decay: float = 0.1,
        eta_start: float = 1.0,
        k_batch_size: int | None = None,
        dyson_solver: DysonSolver | None = None):

        self.max_iter = max_iter
        self.tol = tol
        self.eta_decay = eta_decay
        self.eta_start = eta_start
        self.dyson_solver = dyson_solver
        self.k_batch_size = k_batch_size


    def solve(
        self,
        energies: torch.Tensor,
        coupling_matrix: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5,
        verbosity: int = 0):

        E = energies.view(-1, 1, 1)
        H = hamiltonian_matrix.unsqueeze(-3)
        I = torch.eye(hamiltonian_matrix.shape[-1], device=hamiltonian_matrix.device)
        S = I if overlap_matrix is None else overlap_matrix
        V = coupling_matrix.unsqueeze(-3)
        Z0 = E*S

        if self.k_batch_size and H.ndim >= 4:
            H_batches = torch.split(H, self.k_batch_size, dim=-4)
            V_batches = torch.split(V, self.k_batch_size, dim=-4)
        else:
            H_batches = (H,)
            V_batches = (V,)

        with torch.no_grad():

            eta_current = max(eta, self.eta_start)
            Z = Z0 + 1.j*eta_current*I
            g_surf_batches = []

            for n, (H_batch, V_batch) in enumerate(zip(H_batches, V_batches)):
                
                eps_surf = H_batch
                eps = H_batch
                alpha = V_batch
                beta = V_batch.mH



                for i in range(self.max_iter):

                    g = torch.linalg.pinv(Z - eps)
                    alpha_new = alpha @ g @ alpha
                    beta_new = beta @ g @ beta

                    a_g_b = (alpha @ g @ beta)
                    b_g_a = (beta @ g @ alpha)
                    eps_new = eps + (a_g_b + b_g_a)
                    eps_surf_new = eps_surf + (a_g_b)
                    
                    eps_err = (eps_new - eps).abs().square().mean(
                        axis=(-2, -1))
                    eps_surf_err = (eps_surf_new - eps_surf).abs().square().mean(
                        axis=(-2, -1))
                    
                    max_err = torch.maximum(eps_err, eps_surf_err).max().sqrt()

                    if torch.isnan(max_err):
                        raise RuntimeError(f'Invalid value encountered for "err": {max_err.detach().item()})')
                    
                    if max_err < self.tol:
                        if eta_current > eta:
                            eta_current = max(eta, eta_current * self.eta_decay)
                            Z = Z0 + 1.j*eta_current*I
                            eps_surf_new = H_batch
                            eps_new = H_batch
                            alpha_new = V_batch
                            beta_new = V_batch.mH
                        else:
                            break

                    if verbosity >= 2:
                        print_overwrite(f'Sancho Rubio (batch: {n}, iter: {i}, eta: {eta_current:5e}, err: {max_err:5e}) ')

                    alpha = alpha_new
                    beta = beta_new
                    eps = eps_new
                    eps_surf = eps_surf_new

                if max_err > self.tol or eta_current > eta or torch.isnan(max_err):
                    if verbosity >= 1:
                        print('[Failed]')
                        raise RuntimeError('Sancho-Rubio failed to Converge. (batch: {n}, iter: {i}, eta: {eta_current:5e}, err: {max_err:5e}).')
                else:
                    g_surf_batch = torch.linalg.pinv(Z - eps_surf)
                    g_surf_batches.append(g_surf_batch)
        
        if verbosity >= 1:
            print('[Converged]')

        g_surf = torch.cat(g_surf_batches, dim=-4) \
            if self.k_batch_size and H.ndim >= 4 else g_surf_batches[0]

        return g_surf



class TorchNEGFDevice:

    def _validate_boundary_atomgraph(
            self,
            graph: TorchAtomGraph,
            orientation: torch.IntTensor,
            index: int | None = None):

        if torch.abs(orientation).sum().item() > 1:
            raise ValueError(f'Invalid orientation vector: {orientation}')

        if torch.any(torch.abs(orientation) > 0):

            neighbors_idx = (graph.neighbors >= 0)
            neighbor_images = graph.unit_cell_images[neighbors_idx]
            orientation_idx = torch.nonzero(torch.abs(orientation)).item()
            max_images = torch.abs(neighbor_images).max(dim=0)[0]

            max_oriented_images = \
                max_images[orientation_idx].squeeze().item()

            if max_oriented_images > 1:
                raise ValueError(
                    f'Boundary graph unit cell at index {index} exhibits '
                    'second-nearest cell neighbor coupling. (Got cell image '
                    'interaction radius of {max_oriented_images} for '
                    'orientation {orientation}. \nConsider expanding the '
                    'boundary graph to a supercell along the orientation '
                    'direction or decreasing the cutoff radius.')

    def _validate_potential(self, graph, potential, type='', idx=''):

        if potential.ndim != 1:
            raise ValueError(f'Expected {type} potential to have ndim = 1 (got ndim = {potential.ndim}')
        if graph.atomic_numbers.shape[0] != potential.shape[0]:
            raise ValueError(
                f'Expected len({type} potential) == len({type} graph), '
                f'(got {graph.atomic_numbers.shape[0]} != {potential.shape[0]}')

    def __init__(
            self,
            device: TorchAtomGraph | None,
            leads: list[TorchAtomGraph] | None,
            orientations: list[torch.IntTensor] | None = None,
            device_potential: torch.Tensor | None = None,
            lead_potentials: list[torch.Tensor] | None = None,
        ):
        """Creates a Non-Equilibrium Green's Function (NEGF) system to model
        a device or site of interest. This calculation consists of a 
        system of interest (the `device`) connected to other boundary 
        systems (the `leads`). The leads may be non-periodic, 
        or even semi-infinite with respect to a 
        unit cell orientation.

        In all NEGF calculations, it is assumed that each lead is
        coupled directly to the device and not to any other leads.

        Args:
            device (TorchAtomGraph | None): The site of the NEGF calculation.
            leads (list[TorchAtomGraph] | None): A list of leads (boundary 
                system TorchAtomGraphs), if any.
            orientations (list[torch.IntTensor] | None, optional): A list of
                orientations of each boundary. Each orientation should be of 
                the form `[i, j, k]`, where i,j,k are -1, 0, or 1. Values of
                1 or -1 indicate the direction of a semi-infinite boundary 
                TorchAtomGraph with respect to the corresponding `unit_cell` 
                vector of each boundary. Values of 0 indicate either 
                periodicity or non-periodicity depending on the values 
                of the `periodic` attribute of the TorchAtomGraph. 
                Defaults to None.
        """

        if leads is None:
            leads = []

        if orientations is None:
            orientations = [
                torch.zeros(
                    3, 
                    dtype=torch.int64,
                    device=graph.positions.device)
                for graph in leads
            ]

        self.device = device
        self.leads = leads
        self.orientations = orientations
        self.device_potential = None
        self.lead_potentials = None


        # validate boundary graphs and orientations
        for i, (boundary, orientation) in \
                enumerate(zip(self.leads, self.orientations)):
            self._validate_boundary_atomgraph(
                boundary,
                orientation,
                index=i
            )
        
        self.set_device_potential(device_potential)
        self.set_lead_potentials(lead_potentials)

    def set_device_potential(self, device_potential: torch.Tensor | None):
        if device_potential is not None:
            self._validate_potential(self.device, device_potential, type='device')
        self.device_potential = device_potential

    def set_lead_potentials(self, lead_potentials: list[torch.Tensor] | None):
        if lead_potentials is not None:
            if len(lead_potentials) != len(self.leads):
                raise ValueError(
                    f'Expected len(leads) == len(lead_potentials) '
                    f'(got: {len(self.leads)} != {len(lead_potentials)})')
            for i, (lead, potential) in enumerate(zip(self.leads, lead_potentials)):
                self._validate_potential(lead, potential, type=f'lead {i}')
        
        self.lead_potentials = lead_potentials

    def system_graph(
            self, 
            unit_cell: torch.Tensor | None = None, 
            periodic: torch.BoolTensor | None = None) -> TorchAtomGraph:
        """Builds a TorchAtomGraph that includes the device coupled to the 
        unit cell of each lead. 

        Args:
            unit_cell (torch.Tensor | None, optional): The unit cell of 
                the returned graph. If None, the system is assigned a 
                non-periodic unit cell. Defaults to None.
            periodic (torch.BoolTensor | None, optional): The periodicity of
                the returned graph. If None, the system is 
                treated as non-periodic. Defaults to None.

        Returns:
            TorchAtomGraph: The system TorchAtomGraph
        """

        graphs = [] if self.device is None else [self.device]
        graphs += self.leads

        if len(graphs) == 0:
            raise ValueError(
                'The system must consist of least one AtomGraph object.')

        atomic_numbers = torch.concatenate([
            g.atomic_numbers for g in graphs])
        positions = torch.concatenate([
            g.positions for g in graphs])
        masses = torch.concatenate([
            g.masses for g in graphs])
        features = torch.concatenate([
            g.features for g in graphs])
        outputs = torch.concatenate([
            g.features for g in graphs])
        graph_outputs = torch.concatenate([
            g.features for g in graphs])
        cutoffs = torch.concatenate([
            g.cutoffs for g in graphs
        ])

        unit_cell = self.device.unit_cell
        periodic = self.device.periodic

        system_graph = TorchAtomGraph(
            atomic_numbers=atomic_numbers,
            positions=positions,
            masses=masses,
            unit_cell=unit_cell,
            periodic=periodic,
            features=features,
            outputs=outputs,
            graph_outputs=graph_outputs,
            cutoffs=cutoffs,
            compute_neighbors=True)

        return system_graph

class NEGFSolver:
    """Steady-state NEGF solver employing the 
    self-consistent Born approximation (SCBA).

    This solver iterates scattering self-energies :math:`\\Sigma^{</>}` to
    self-consistency using a fixed-point iteration with linear mixing.

    Optionally, the solver can use an adaptive mixing factor that only
    *decreases* from the configured `mixing_alpha` if the SCBA residue
    oscillates. This helps prevent limit cycles late in the iteration while
    still allowing the user to choose an aggressive initial mixing factor for
    early iterations.
    """

    def __init__(
            self,
            negf_device: TorchNEGFDevice,
            model: TorchBandStructureModel,
            mixing_alpha: float = 0.1,
            adaptive_mixing: bool = True,
            mixing_alpha_min: float = 1e-10,
            mixing_alpha_decay: float = 0.5,
            mixing_burn_in: int = 5,
            mixing_decay_cooldown: int = 2,
            max_iter: int = 1024,
            tol: float = 1e-6,
            dyson_solver: DysonSolver | None = None,
            surface_solver: SurfaceGreensFunctionSolver | None = None,
            k_batch_size=None):
        
        self.model = model
        self.negf_device = negf_device
        self.mixing_alpha = mixing_alpha
        self.adaptive_mixing = adaptive_mixing
        self.mixing_alpha_min = mixing_alpha_min
        self.mixing_alpha_decay = mixing_alpha_decay
        self.mixing_burn_in = mixing_burn_in
        self.mixing_decay_cooldown = mixing_decay_cooldown
        self.max_iter = max_iter
        self.tol = tol
        self.k_batch_size = k_batch_size

        # resolve gf solvers:
        self.dyson_solver = DenseDysonSolver() if dyson_solver is None else dyson_solver
        self.surface_solver = SanchoRubioSolver(
            tol=self.tol,
            k_batch_size=self.k_batch_size
        ) if surface_solver is None else surface_solver

    def _resolve_k_pts(
                self,
                k_pts: torch.Tensor | None, 
                template: torch.Tensor) -> torch.Tensor:

        if k_pts is None:
            k_pts = torch.zeros(
                (1, 3),
                dtype=template.dtype,
                device=template.device,
            )

        return k_pts

    def _adaptive_mixing_alpha(self, residue_history: list[float], current_alpha: float) -> float:
        """Update mixing alpha based on residue oscillations.

        The update is conservative: `current_alpha` is only reduced (never
        increased) and is always bounded below by `mixing_alpha_min`.
        """
        if not self.adaptive_mixing:
            return current_alpha

        if current_alpha <= self.mixing_alpha_min:
            return current_alpha

        if len(residue_history) < 3:
            return current_alpha

        d1 = residue_history[-1] - residue_history[-2]
        d2 = residue_history[-2] - residue_history[-3]

        # Residue "oscillation" detection:
        #   - sign change in successive deltas (down then up, or up then down),
        #   - and at least one of those deltas is non-trivial (avoid noise).
        oscillates = (d1 * d2) < 0.0
        if not oscillates:
            return current_alpha

        if max(abs(d1), abs(d2)) <= 0.0:
            return current_alpha

        return max(self.mixing_alpha_min, current_alpha * self.mixing_alpha_decay)

    def _check_model_graph_device_alignment(self, reference: torch.Tensor):
        # Many unit tests use lightweight "fake" models that only implement the
        # minimal NEGF interface (hamiltonian_matrix/orbital_indices) and are
        # not torch.nn.Module instances. In that case there are no parameters/
        # buffers to validate.
        params_fn = getattr(self.model, 'parameters', None)
        buffers_fn = getattr(self.model, 'buffers', None)
        if not callable(params_fn) or not callable(buffers_fn):
            return

        model_tensors = list(params_fn()) + list(buffers_fn())
        if not model_tensors:
            return

        model_device = model_tensors[0].device
        reference_device = reference.device
        if model_device != reference_device:
            raise RuntimeError(
                'NEGFSolver device mismatch: model parameters/buffers are on '
                f'{model_device}, but the NEGF tensors are on {reference_device}. '
                'Move the model to the same device as the graph/energies, e.g. '
                '`model = model.to(energies.device)`.'
            )

        graphs = [g for g in [self.negf_device.device, *self.negf_device.leads] if g is not None]
        for i, graph in enumerate(graphs):
            if graph.positions.device != model_device:
                raise RuntimeError(
                    'NEGFSolver device mismatch: graph tensors are on '
                    f'{graph.positions.device}, but model parameters/buffers are on {model_device}. '
                    'Move the model and all TorchAtomGraph inputs to the same device.'
                )

    def solve_lead_greens_functions(
        self,
        energies: torch.Tensor,
        k_pts: torch.Tensor | None = None,
        eta: float = 1e-5,
        verbosity: int = 0) -> list[torch.Tensor]:
        """Evaluates the retarded Green's function of each lead.
        For the surface Green's functions of any semi-infinite leads,
        the numerical Sancho-Rubio method is used.

        Args:
            energies (torch.Tensor): The energies at which the Green's function 
                should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function. Increasing this parameter may make any surface 
                Green's function methods converge faster for semi-infinite
                leads at the cost of broadening the energy spectrum. 
                Defaults to 1e-5.

        Returns:
            list[torch.Tensor]: A list of boundary Green's functions, having
            shape [M, E, N, N], where `N` is the dimension of each 
            boundary Hamiltonian.
        """

        self._check_model_graph_device_alignment(energies)
        k_pts = self._resolve_k_pts(k_pts, energies)

        g_surfaces = []

        for i, (boundary, orientation) in \
                enumerate(zip(self.negf_device.leads, self.negf_device.orientations)):

            if verbosity >= 1:
                    print(f'Solving surface Green\'s function (lead {i}).')

            if (orientation != 0).any():

                # construct supercell along orientation for 
                # extracting coupling matrix
                sc_size = torch.ones_like(orientation) + \
                    2*torch.abs(orientation)
                sc_size[orientation < 0] *= -1
                sc = boundary.supercell(sc_size)
                N = sc_size.abs().prod().item()
                
                assert N == 3

                # extract coupling matrix elements from supercell
                boundary_site_idxs, _ = self.model.orbital_indices(boundary)
                H_size = boundary_site_idxs.shape[0]
                H = self.model.hamiltonian_matrix(sc, k_pts=k_pts)
                H_cell_idxs = torch.arange(
                    H_size, 2*H_size, device=orientation.device)
                H_outer_idxs = H_cell_idxs - H_size

                V = H[:, H_cell_idxs][:, :, H_outer_idxs]
                H0 = H[:, H_cell_idxs][:, :, H_cell_idxs]

                if self.negf_device.lead_potentials is not None:
                    lead_potential = self.negf_device.lead_potentials[i]
                    H0 += torch.diag(lead_potential[boundary_site_idxs]).unsqueeze(0)
                
                # compute surface Greens function with Sancho Rubio method
                g_surface = self.surface_solver.solve(
                    energies=energies,
                    coupling_matrix=V,
                    hamiltonian_matrix=H0,
                    eta=eta,
                    verbosity=verbosity,
                )
            
            else:

                # if surface is not semi-infinite, use regular Greens function
                
                boundary_site_idxs, _ = self.model.orbital_indices(boundary)
                H0 = self.model.hamiltonian_matrix(boundary, k_pts=k_pts)

                lead_potentials = getattr(self.negf_device, 'lead_potentials', None)
                if lead_potentials is not None:
                    lead_potential = lead_potentials[i]
                    H0 += torch.diag(lead_potential[boundary_site_idxs]).unsqueeze(0)
                
                g_surface = self.dyson_solver.solve(
                    energies=energies,
                    hamiltonian_matrix=H0,
                    eta=eta
                )
            
            g_surfaces.append(g_surface)

        return g_surfaces

    def solve_hamiltonian_self_energies(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            lead_gfs: list[torch.Tensor] | None = None,
            verbosity: int = 0,
            **solver_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Solves for the device Hamiltonian and a tensor of self-energy 
        correlations to the device Hamiltonian based on the computed 
        surface Green's functions of each lead.

        Args:
            energies (torch.Tensor): The energies at which the Green's function 
                should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function. Increasing this parameter may make any surface 
                Green's function methods converge faster for semi-infinite
                boundaries at the cost of broadening the energy spectrum. 
                Defaults to 1e-5.
            lead_gfs (list[torch.Tensor] | None, optional): Pre-computed 
                lead Green's functions returned from a call to 
                :meth:`NEGFSolver.solve_lead_greens_functions()`. Defaults to None.
            dyson_solver (DysonSolver | None, optional): Custom Dyson solver
                used when building the retarded Green's function. Defaults
                to the dense inverse solver.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: A tuple containing the 
            following:

            - site_hamitonian (torch.Tensor):
                The site Hamiltonian matrix (shape: [M, N, N])
            - self_energies (torch.Tensor):
                The self-energy lead contributions or None if no leads
                are defined (shape: [N_boundaries, E, M, N, N])

        """

        self._check_model_graph_device_alignment(energies)
        k_pts = self._resolve_k_pts(k_pts, energies)

        if self.negf_device.device is None:
            raise ValueError(
                'The site AtomGraph must be defined '
                'to compute site self-energies.')

        H_sizes = [
            self.model.orbital_indices(g)[0].shape[0]
            for g in self.negf_device.leads
        ]

        # determine indices of components of the system Hamiltonian
        device_site_idxs, _ = self.model.orbital_indices(self.negf_device.device)
        H_site_size = device_site_idxs.shape[0]
        H_boundary_idxs = []
        offset = H_site_size
        for size in H_sizes:
            H_boundary_idxs.append(torch.arange(
                offset, offset + size,
                device=energies.device))
            offset += size

        if verbosity >= 1:
            print('Computing device Hamiltonian.')

        # compute system graph and Hamiltonian matrix
        system = self.negf_device.system_graph()
        H_system = self.model.hamiltonian_matrix(system, k_pts=k_pts)
        H_site = H_system[:, :H_site_size, :H_site_size]
        
        # add device potential
        device_potential = getattr(self.negf_device, 'device_potential', None)
        if device_potential is not None:
            H_site += torch.diag(device_potential[device_site_idxs]).unsqueeze(0)
        
        # compute boundary Greens functions (if not supplied)
        if lead_gfs is None:
            lead_gfs = self.solve_lead_greens_functions(
                energies=energies,
                k_pts=k_pts,
                eta=eta,
                **solver_kwargs
            )
        
        if len(lead_gfs) != len(self.negf_device.leads):
            raise ValueError(
                f'Expected a list of {len(self.negf_device.leads)} '
                f'boundary Greens functions. (got list of '
                f'size: {len(lead_gfs)}')

        # extract device coupling and compute self-energy
        self_energies_list = []
        for idxs, gf in \
                zip(H_boundary_idxs, lead_gfs):
            V = H_system[:, :H_site_size, idxs].unsqueeze(-3)

            self_energy = V @ gf @ V.mH

            self_energies_list.append(self_energy)
        
        self_energies = torch.stack(self_energies_list) if self_energies_list else None

        return H_site, self_energies

    def solve(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            chemical_potentials: list[torch.Tensor] | list[float] | None = None,
            kt: torch.Tensor | float = 0.0,
            scatterers: list[Scatterer] | None = None,
            warm_start: 'TorchNEGFResult' | None = None,
            reuse_warm_hamiltonian: bool = False,
            lead_gfs: list[torch.Tensor] | None = None,
            verbosity: int = 0,
            cache_dir: str | None = None,
            **solver_kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if energies.numel() < 2:
            raise ValueError(
                'Energies must have size > 1 and be uniformly spaced to solve for scattering self-energy.'
            )

        if torch.diff(energies).std() > 1e-5 * (energies.max() - energies.min()):
            raise ValueError(
                'Energies must have size > 1 and be uniformly spaced to solve for scattering self-energy.'
            )

        self._check_model_graph_device_alignment(energies)
        with torch.no_grad():

            dE = torch.diff(energies).mean()

            if reuse_warm_hamiltonian:

                if warm_start is None:
                    raise ValueError(
                        'reuse_warm_hamiltonian=True requires `warm_start` to be provided.'
                    )
                if warm_start.hamiltonian is None or warm_start.lead_self_energies is None:
                    raise ValueError(
                        'Warm start must include both `hamiltonian` and `lead_self_energies` '
                        'to reuse them.'
                    )

                warm_energies = warm_start._tensor(warm_start.energies).to(
                    device=energies.device,
                    dtype=energies.dtype,
                )
                if warm_energies.shape != energies.shape or not torch.allclose(warm_energies, energies):
                    raise ValueError(
                        'Warm-start energies must match `energies` when reusing hamiltonian/lead self-energies.'
                    )

                if isinstance(warm_start.hamiltonian, list):
                    H_site = torch.cat([warm_start._tensor(hb) for hb in warm_start.hamiltonian], dim=-3)
                else:
                    H_site = warm_start._tensor(warm_start.hamiltonian)

                if isinstance(warm_start.lead_self_energies, list):
                    lead_self_energies = torch.cat(
                        [warm_start._tensor(sb) for sb in warm_start.lead_self_energies],
                        dim=-4,
                    )
                else:
                    lead_self_energies = warm_start._tensor(warm_start.lead_self_energies)

                H_site = H_site.to(device=energies.device)
                lead_self_energies = lead_self_energies.to(device=energies.device)
            else:
                H_site, lead_self_energies = self.solve_hamiltonian_self_energies(
                    energies=energies,
                    k_pts=k_pts,
                    eta=eta,
                    lead_gfs=lead_gfs,
                    verbosity=verbosity,
                    **solver_kwargs,
                )

            if verbosity >= 1:
                print('Solving device Green\'s function.')
            

            #site_gf_inv = torch.linalg.inv(site_gf)
            num_leads = len(self.negf_device.leads)

            if chemical_potentials is None:
                chemical_potentials = [0.0] * num_leads
            if len(chemical_potentials) != num_leads:
                raise ValueError(
                    'Expected len(chemical_potentials) == len(self.negf_device.leads);'
                    f' got {len(chemical_potentials)} != {num_leads}.'
                )

            if self.k_batch_size:
                H_batches = torch.split(H_site, self.k_batch_size, dim=-3)
                lead_se_batches = torch.split(lead_self_energies, self.k_batch_size, dim=-4)
            else:
                H_batches = (H_site,)
                lead_se_batches = (lead_self_energies,)

            warm_lesser_batches = None
            warm_greater_batches = None
            if warm_start is not None:
                if warm_start.lesser_self_energies is None or warm_start.greater_self_energies is None:
                    raise ValueError(
                        'Warm start requires both lesser_self_energies and greater_self_energies.'
                    )

                def _warm_to_batches(x):
                    if isinstance(x, list):
                        return x
                    if self.k_batch_size:
                        return list(torch.split(x.to_dense(), self.k_batch_size, dim=-4))
                    return [x]

                warm_lesser_batches = _warm_to_batches(warm_start.lesser_self_energies)
                warm_greater_batches = _warm_to_batches(warm_start.greater_self_energies)

                if not (len(warm_lesser_batches) == len(warm_greater_batches) == len(H_batches)):
                    raise ValueError(
                        'Warm-start self-energy batches must match solver k-batching.'
                    )

            scatterers_resolved = list(scatterers) if scatterers else []

            # initialize batches:
            g_lesser_batches = []
            g_greater_batches = []
            lesser_se_batches = []
            greater_se_batches = []
            site_gf_batches = []

            # initialize additional file batch lists:
            hamiltonian_file_batches = []
            lead_se_file_batches = []

            # Initialize lead occupation probabilities f_l(E) used to build
            # lead correlation self-energies Sigma_l^{</>}.
            occupation_probs = None
            if chemical_potentials:
                occupation_probs = torch.stack([
                    1.0 / (1.0 + torch.exp((energies - mu) / kt))
                    for mu in chemical_potentials
                ]) 

                f_lesser = occupation_probs.view(occupation_probs.shape + (1, 1))
                f_greater = 1 - f_lesser
                f_lesser_e = f_lesser.unsqueeze(1)
                f_greater_e = f_greater.unsqueeze(1)

            if scatterers_resolved and verbosity >= 1:
                print('Solving self-consistent scattering contributions.')

            for n, (H_batch, lead_se_batch) in enumerate(zip(H_batches, lead_se_batches)):                    

                site_gf = self.dyson_solver.solve(
                    energies=energies,
                    hamiltonian_matrix=H_batch,
                    self_energies=lead_se_batch,
                    eta=eta,
                )

                # Total lead-induced correlation self-energies:
                #   Sigma_leads^< = i * sum_l[Gamma_l f_l]
                #   Sigma_leads^> = i * sum_l[Gamma_l (1-f_l)]
                sigma_lesser_leads = torch.zeros_like(site_gf)
                sigma_greater_leads = torch.zeros_like(site_gf)

                if chemical_potentials:
                    gamma_leads = 1.j * (lead_se_batch - lead_se_batch.mH)
                    sigma_lesser_leads = 1.j * (gamma_leads * f_lesser_e).sum(0)
                    sigma_greater_leads = -1.j * (gamma_leads * f_greater_e).sum(0)

                sigma_lesser_scattering = torch.zeros_like(sigma_lesser_leads)
                sigma_greater_scattering = torch.zeros_like(sigma_lesser_leads)
                if warm_lesser_batches is not None and warm_greater_batches is not None:
                    sigma_lesser_scattering = warm_start._tensor(warm_lesser_batches[n]).to(
                        device=sigma_lesser_leads.device,
                        dtype=sigma_lesser_leads.dtype,
                    )
                    sigma_greater_scattering = warm_start._tensor(warm_greater_batches[n]).to(
                        device=sigma_greater_leads.device,
                        dtype=sigma_greater_leads.dtype,
                    )
                    if sigma_lesser_scattering.shape != sigma_lesser_leads.shape:
                        raise ValueError(
                            'Warm-start lesser self-energy shape mismatch for batch '
                            f'{n}: expected {tuple(sigma_lesser_leads.shape)}, '
                            f'got {tuple(sigma_lesser_scattering.shape)}.'
                        )
                    if sigma_greater_scattering.shape != sigma_greater_leads.shape:
                        raise ValueError(
                            'Warm-start greater self-energy shape mismatch for batch '
                            f'{n}: expected {tuple(sigma_greater_leads.shape)}, '
                            f'got {tuple(sigma_greater_scattering.shape)}.'
                        )

                site_gf_new = site_gf
                g_lesser = site_gf_new @ sigma_lesser_leads @ site_gf_new.mH
                g_greater = site_gf_new @ sigma_greater_leads @ site_gf_new.mH

                residue = 0

                if scatterers_resolved:
                    build_matrix = getattr(self.dyson_solver, 'build_matrix', None)
                    can_build_matrix = callable(build_matrix)
                    g0_inv = None
                    if not can_build_matrix:
                        # Fallback: recover G0^{-1} by explicitly inverting G0.
                        # Avoid casting to complex128 to limit peak memory use.
                        g0_inv = torch.linalg.inv(site_gf_new)

                    mixing_alpha = float(self.mixing_alpha)
                    residue_history: list[float] = []
                    last_alpha_decay_iter = -10**9
                    for i in range(self.max_iter):

                        # Build the retarded scattering self-energy Σ^r from the
                        # correlation self-energies Σ^{</>}.
                        #
                        # Physical broadening:
                        #   Γ(E) = i(Σ^>(E) - Σ^<(E)) = i(Σ^r - Σ^a) = -2 Im Σ^r
                        #
                        # Kramers-Kronig relation:
                        #   Re Σ^r(E) = (1/2) H[Γ](E)
                        #
                        # so:
                        #   Σ^r(E) = (1/2) H[Γ](E) - i Γ(E)/2
                        gamma_scattering = 1j * (sigma_greater_scattering - sigma_lesser_scattering)
                        gamma_scattering = 0.5 * (gamma_scattering + gamma_scattering.mH)

                        lambda_scattering = 0.5 * hilbert_transform(gamma_scattering)
                        lambda_scattering = 0.5 * (lambda_scattering + lambda_scattering.mH)

                        scattering_self_energy = lambda_scattering - 0.5j * gamma_scattering

                        if can_build_matrix:
                            A_ret = build_matrix(
                                energies=energies,
                                hamiltonian_matrix=H_batch,
                                self_energies=lead_se_batch,
                                eta=eta,
                            )
                            A_ret.sub_(scattering_self_energy)
                            site_gf_new = torch.linalg.inv(A_ret)
                        else:
                            site_gf_new = torch.linalg.inv(g0_inv - scattering_self_energy)

                        g_lesser = site_gf_new @ (sigma_lesser_leads + sigma_lesser_scattering) @ site_gf_new.mH
                        g_greater = site_gf_new @ (sigma_greater_leads + sigma_greater_scattering) @ site_gf_new.mH

                        sigma_lesser_scattering_new, sigma_greater_scattering_new = correlation_self_energies_from_scatterers(
                            scatterers=scatterers_resolved,
                            g_lesser=g_lesser,
                            g_greater=g_greater,
                            energies=energies,
                            dE=dE,
                            kt=kt,
                            **solver_kwargs,
                        )

                        # Use a relative convergence metric so the same `tol`
                        # scale works across devices/unit systems.
                        #
                        # Normalize by both old and new scales; otherwise, if
                        # the fixed-point solution is ~0, dividing by only the
                        # new scale can artificially inflate the residue and
                        # prevent convergence.
                        abs_err = (
                            torch.abs(sigma_lesser_scattering - sigma_lesser_scattering_new).amax()
                            + torch.abs(sigma_greater_scattering - sigma_greater_scattering_new).amax()
                        )
                        scale_old = (
                            torch.abs(sigma_lesser_scattering).amax()
                            + torch.abs(sigma_greater_scattering).amax()
                        )
                        scale_new = (
                            torch.abs(sigma_lesser_scattering_new).amax()
                            + torch.abs(sigma_greater_scattering_new).amax()
                        )
                        scale = torch.maximum(scale_old, scale_new)
                        residue = abs_err / torch.clamp(scale, min=1e-30)
                        residue_history.append(float(residue.item()))
                        if (
                            self.adaptive_mixing
                            and i >= self.mixing_burn_in
                            and (i - last_alpha_decay_iter) >= self.mixing_decay_cooldown
                        ):
                            mixing_alpha_new = self._adaptive_mixing_alpha(residue_history, mixing_alpha)
                            if mixing_alpha_new < mixing_alpha:
                                last_alpha_decay_iter = i
                            mixing_alpha = mixing_alpha_new

                        if verbosity >= 1:
                            if verbosity >= 2 and self.adaptive_mixing:
                                print_overwrite(
                                    f'SCBA (batch: {n}, iter: {i}, err: {residue:5e}, alpha: {mixing_alpha:0.3g}) '
                                )
                            else:
                                print_overwrite(f'SCBA (batch: {n}, iter: {i}, err: {residue:5e}) ')

                        if residue <= self.tol:
                            sigma_lesser_scattering = sigma_lesser_scattering_new
                            sigma_greater_scattering = sigma_greater_scattering_new
                            break

                        sigma_lesser_scattering.mul_(1 - mixing_alpha).add_(
                            sigma_lesser_scattering_new,
                            alpha=mixing_alpha,
                        )
                        sigma_greater_scattering.mul_(1 - mixing_alpha).add_(
                            sigma_greater_scattering_new,
                            alpha=mixing_alpha,
                        )

                if residue > self.tol:
                    if verbosity >= 1:
                        print('[Failed]')
                    raise RuntimeError(f'SCBA failed to Converge. (batch: {n}, iter: {i}, err: {residue:5e}).')

                if cache_dir:

                    # write batch fragments to disk to conserve RAM:
                    hamiltonian_file = os.path.join(cache_dir, f'hamiltonian.k{n}.pt')
                    lead_se_file = os.path.join(cache_dir, f'lead_self_energies.k{n}.pt')
                    site_gf_file = os.path.join(cache_dir, f'site_gf.k{n}.pt')
                    lesser_se_file = os.path.join(cache_dir, f'lesser_self_energies.k{n}.pt')
                    greater_se_file = os.path.join(cache_dir, f'greater_self_energies.k{n}.pt')
                    g_lesser_file = os.path.join(cache_dir, f'g_lesser.k{n}.pt')
                    g_greater_file = os.path.join(cache_dir, f'g_greater.k{n}.pt')

                    torch.save(H_batch.to_sparse(), hamiltonian_file)
                    torch.save(lead_se_batch.to_sparse(), lead_se_file)
                    torch.save(site_gf_new.to_sparse(), site_gf_file)
                    torch.save(sigma_lesser_scattering.to_sparse(), lesser_se_file)
                    torch.save(sigma_greater_scattering.to_sparse(), greater_se_file)
                    torch.save(g_lesser.to_sparse(), g_lesser_file)
                    torch.save(g_greater.to_sparse(), g_greater_file)

                    site_gf_batches.append(site_gf_file)
                    lesser_se_batches.append(lesser_se_file)
                    greater_se_batches.append(greater_se_file)
                    g_lesser_batches.append(g_lesser_file)
                    g_greater_batches.append(g_greater_file)

                    hamiltonian_file_batches.append(hamiltonian_file)
                    lead_se_file_batches.append(lead_se_file)

                else:
                    site_gf_batches.append(site_gf_new.to_sparse())
                    lesser_se_batches.append(sigma_lesser_scattering.to_sparse())
                    greater_se_batches.append(sigma_greater_scattering.to_sparse())
                    g_lesser_batches.append(g_lesser.to_sparse())
                    g_greater_batches.append(g_greater.to_sparse())


        if verbosity >= 1:
            print('[Converged]')

        if cache_dir:

            # serialize energies and occupations:
            energies_file = os.path.join(cache_dir, 'energies.pt')
            occupations_file = os.path.join(cache_dir, 'lead_occupation_probs.pt')
            k_pts_file = os.path.join(cache_dir, 'k_pts.pt')

            torch.save(energies, energies_file)
            torch.save(occupation_probs, occupations_file)
            torch.save(k_pts, k_pts_file)

            result = TorchNEGFResult(
                energies=energies_file,
                hamiltonian=hamiltonian_file_batches,
                g_ret=site_gf_batches,
                lead_self_energies=lead_se_file_batches,
                lead_occupation_probs=occupations_file,
                lesser_self_energies=lesser_se_batches,
                greater_self_energies=greater_se_batches,
                g_lesser=g_lesser_batches,
                g_greater=g_greater_batches,
                k_pts=k_pts_file,
            )

        else:

            result = TorchNEGFResult(
                energies=energies,
                hamiltonian=H_site,
                g_ret=torch.cat(site_gf_batches, dim=-4),
                lead_self_energies=lead_self_energies,
                lead_occupation_probs=occupation_probs,
                lesser_self_energies=torch.cat(lesser_se_batches, dim=-4),
                greater_self_energies=torch.cat(greater_se_batches, dim=-4),
                g_lesser=torch.cat(g_lesser_batches, dim=-4),
                g_greater=torch.cat(g_greater_batches, dim=-4),
                k_pts=k_pts,
            )

        return result


class TorchNEGFResult:


    def __init__(
        self,
        energies: torch.Tensor | str,
        hamiltonian: torch.Tensor | list[str],
        g_ret: torch.Tensor | list[str],
        lead_self_energies: torch.Tensor | list[str] | None = None,
        lead_occupation_probs: torch.Tensor | list[str] | None = None,
        lesser_self_energies: torch.Tensor | list[str] | None = None,
        greater_self_energies: torch.Tensor | list[str] | None = None,
        g_lesser: torch.Tensor | list[str] | None = None,
        g_greater: torch.Tensor | list[str] | None = None,
        k_pts: torch.Tensor | str | None = None):
    

        self.energies = energies
        self.hamiltonian = hamiltonian
        self.g_ret = g_ret

        self.lead_self_energies = lead_self_energies
        self.lead_occupation_probs = lead_occupation_probs
        self.lesser_self_energies = lesser_self_energies
        self.greater_self_energies = greater_self_energies
        self.g_lesser = g_lesser
        self.g_greater = g_greater
        self.k_pts = k_pts

    @staticmethod
    def load(cache_dir) -> TorchNEGFResult:

        energies_file = os.path.join(cache_dir, 'energies.pt')
        
        def _k_sort_key(path: str) -> int:
            base = os.path.basename(path)
            try:
                return int(base.split('k')[-1].split('.')[0])
            except ValueError:
                return 0

        hamiltonian_files = sorted(
            glob(os.path.join(cache_dir, 'hamiltonian.k*.pt')),
            key=_k_sort_key,
        )

        lead_se_files = sorted(
            glob(os.path.join(cache_dir, 'lead_self_energies.k*.pt')),
            key=_k_sort_key,
        )
        site_gf_files = sorted(
            glob(os.path.join(cache_dir, 'site_gf.k*.pt')),
            key=_k_sort_key,
        )
        lesser_se_files = sorted(
            glob(os.path.join(cache_dir, 'lesser_self_energies.k*.pt')),
            key=_k_sort_key,
        )
        greater_se_files = sorted(
            glob(os.path.join(cache_dir, 'greater_self_energies.k*.pt')),
            key=_k_sort_key,
        )
        g_lesser_files = sorted(
            glob(os.path.join(cache_dir, 'g_lesser.k*.pt')),
            key=_k_sort_key,
        )
        g_greater_files = sorted(
            glob(os.path.join(cache_dir, 'g_greater.k*.pt')),
            key=_k_sort_key,
        )

        occupations_file = os.path.join(cache_dir, 'lead_occupation_probs.pt')
        k_pts_file = os.path.join(cache_dir, 'k_pts.pt')

        return TorchNEGFResult(
            energies=energies_file,
            hamiltonian=hamiltonian_files,
            g_ret=site_gf_files,
            lead_self_energies=lead_se_files or None,
            lead_occupation_probs=occupations_file if os.path.exists(occupations_file) else None,
            lesser_self_energies=lesser_se_files or None,
            greater_self_energies=greater_se_files or None,
            g_lesser=g_lesser_files or None,
            g_greater=g_greater_files or None,
            k_pts=k_pts_file if os.path.exists(k_pts_file) else None,
        )

    def as_dense(self, t, **to_kwargs):
        return torch.cat([
            self._tensor(b).to(**to_kwargs)
            for b in self._as_batches(t)], dim=0)
    def as_sparse(self, t, **to_kwargs):
        return torch.cat([
            self._tensor(b).to(**to_kwargs).to_sparse() 
            for b in self._as_batches(t)
        ], dim=0)

    def _tensor(self, t):
        return torch.load(t).to_dense() if isinstance(t, str) else t.to_dense()

    def _device(self):
        g0 = self.g_ret[0] if isinstance(self.g_ret, list) else self.g_ret
        return self._tensor(g0).device

    def _lead_count(self) -> int:
        if self.lead_self_energies is None:
            return 0
        se0 = self.lead_self_energies[0] if isinstance(self.lead_self_energies, list) else self.lead_self_energies
        return self._tensor(se0).shape[0]

    def _as_batches(self, obj):
        return obj if isinstance(obj, list) else [obj]

    def _fermi_dirac(self, energies, mu, kt=1e-10):
        return 1.0 / (1.0 + torch.exp((energies - mu) / kt))

    def _resolve_in_out_idxs(self, in_idx, out_idx):

        N = self._lead_count()

        if N <= 0:
            raise ValueError('Cannot resolve lead indices without lead self energies.')
        
        in_unresolved = in_idx is None
        out_unresolved = out_idx is None
        in_const = isinstance(in_idx, int)
        out_const = isinstance(out_idx, int)
        device = self._device()
        
        if in_unresolved and out_unresolved:
            raise ValueError('Either an input or output lead index must be specified (both are None).')

        def _to_tensor(idx, name):
            if isinstance(idx, int):
                return torch.tensor([idx], device=device, dtype=torch.int64)
            if isinstance(idx, torch.Tensor):
                if idx.ndim != 1:
                    raise ValueError(f'Expected {name} to have ndim == 1; (got ndim == {idx.ndim}).')
                return idx.to(device=device, dtype=torch.int64)
            if idx is None:
                return None
            raise ValueError(f'Unsupported type for {name}: {type(idx)}')

        in_idx_t = _to_tensor(in_idx, 'in_idx')
        out_idx_t = _to_tensor(out_idx, 'out_idx')

        # return if in/out indices are both integers
        if in_const and out_const:
            return in_idx_t, out_idx_t, False
        
        # broadcast function to fit in_idx onto out_idx, or vice versa:
        def _broadcast(const_idx, tgt_idx, sum_over_tgt):
            if sum_over_tgt:
                tgt_idx = torch.arange(N, device=device, dtype=torch.int64)
                tgt_idx = tgt_idx[tgt_idx != const_idx.item()]
            elif tgt_idx is None:
                raise ValueError('Unresolved target indices.')
            elif tgt_idx.ndim != 1:
                raise ValueError(f'Expected broadcast target to have ndim == 1; got {tgt_idx.ndim}.')

            src_idx = torch.full_like(tgt_idx, const_idx.item())
            
            return src_idx, tgt_idx

        # apply broadcasting
        if in_const:
            in_idx_t, out_idx_t = _broadcast(in_idx_t, out_idx_t, out_unresolved)
        elif out_const:
            out_idx_t, in_idx_t = _broadcast(out_idx_t, in_idx_t, in_unresolved)
            
        # validate shapes of indices
        if in_idx_t.ndim != 1:
            raise ValueError(
                'Expected in_idx to have ndim == 1; '
                f'(got ndim == {in_idx_t.ndim})')
        if out_idx_t.ndim != 1:
            raise ValueError(
                'Expected out_idx to have ndim == 1; '
                f'(got ndim == {out_idx_t.ndim})')
        if in_idx_t.shape != out_idx_t.shape:
            raise ValueError('Expected in_idx and out_idx to have same shape '
                            f'(got {in_idx_t.shape} and {out_idx_t.shape})')

        sum_result = (in_unresolved or out_unresolved)

        return in_idx_t, out_idx_t, sum_result

    def ldos(
            self,
            band_mult: float = 1.0,
            k_resolved: bool = False,
            **solver_kwargs) -> torch.Tensor:
        """Evaluates the local density of states (LDOS) of the site.

        Args:
            k_resolved (bool, optional): if True, returns the LDOS without 
                integrating over the :math:`\\mathbf{k}` degrees of freedom.
                Defaults to False.

        Returns:
            torch.Tensor: The evaluated LDOS 
                (shape: [E, N], or [M, E, N] if `k_resolved`).
        """

        g_ret_batches = [self.g_ret] if not isinstance(self.g_ret, list) else self.g_ret


        if k_resolved:
            return torch.cat([
                (-1/torch.pi) * torch.diagonal(self._tensor(g_ret_batch).imag, dim1=-2, dim2=-1) * band_mult
                for g_ret_batch in g_ret_batches
            ], dim=-2)

        else:
            Nk = 0
            ldos = 0
            
            for g_ret_batch in g_ret_batches:
                g_ret = self._tensor(g_ret_batch)
                ldos += (-1/torch.pi) * (
                    torch.diagonal(g_ret.imag, dim1=-2, dim2=-1).sum(-3)
                )
                Nk += g_ret.shape[-4]
            
            ldos *= band_mult / Nk

        return ldos

    def dos(
            self,
            band_mult: float = 1.0,
            k_resolved: bool = False):
        """Evaluates the density of states (DOS) of the site.

        Args:
            band_mult (float, optional): Band multiplicity. Defaults to 1.0.
            k_resolved (bool, optional): if True, returns the LDOS without 
                integrating over the :math:`\\mathbf{k}` degrees of freedom.
                Defaults to False.

        Returns:
            torch.Tensor: The evaluated DOS 
                (shape: [E], or [M, E] if `k_resolved`).
        """

        g_ret_batches = [self.g_ret] if not isinstance(self.g_ret, list) else self.g_ret


        if k_resolved:
            per_batch = []
            for g_ret_batch in g_ret_batches:
                g_ret = self._tensor(g_ret_batch)
                # sum over orbital dimension, keep k dimension
                per_batch.append(
                    (-1/torch.pi) * torch.diagonal(g_ret.imag, dim1=-2, dim2=-1).sum(dim=-1) * band_mult
                )  # shape: [K, E]
            return torch.cat(per_batch, dim=0)  # concat along k batches

        else:
            Nk = 0
            dos = 0
            
            for g_ret_batch in g_ret_batches:
                g_ret = self._tensor(g_ret_batch)
                dos += (-1/torch.pi) * (
                    torch.diagonal(g_ret.imag, dim1=-2, dim2=-1).sum(dim=(-3, -1))
                )
                Nk += g_ret.shape[-4]
            
            dos *= band_mult / Nk

        return dos

    def occupied_ldos(
            self,
            band_mult: float = 1.0,
            k_resolved: bool = False,
            **solver_kwargs) -> torch.Tensor:
        """Evaluates the occupied local density of states from G^<.

        Args:
            k_resolved (bool, optional): if True, returns the occupied LDOS
                without integrating over the :math:`\\mathbf{k}` degrees of
                freedom. Defaults to False.

        Returns:
            torch.Tensor: The evaluated occupied LDOS
                (shape: [E, N], or [M, E, N] if `k_resolved`).
        """

        if self.g_lesser is None:
            raise ValueError('Cannot calculate occupied LDOS without g_lesser.')

        g_lesser_batches = self._as_batches(self.g_lesser)

        if k_resolved:
            return torch.cat([
                (1 / (2 * torch.pi)) *
                torch.diagonal(self._tensor(g_lesser_batch).imag, dim1=-2, dim2=-1) *
                band_mult
                for g_lesser_batch in g_lesser_batches
            ], dim=-2)

        Nk = 0
        occ_ldos = 0

        for g_lesser_batch in g_lesser_batches:
            g_lesser = self._tensor(g_lesser_batch)
            occ_ldos += (1 / (2 * torch.pi)) * (
                torch.diagonal(g_lesser.imag, dim1=-2, dim2=-1).sum(-3)
            )
            Nk += g_lesser.shape[-4]

        occ_ldos *= band_mult / Nk

        return occ_ldos

    def occupied_dos(
            self,
            band_mult: float = 1.0,
            k_resolved: bool = False):
        """Evaluates the occupied density of states from G^<.

        Args:
            band_mult (float, optional): Band multiplicity. Defaults to 1.0.
            k_resolved (bool, optional): if True, returns occupied DOS without
                integrating over the :math:`\\mathbf{k}` degrees of freedom.
                Defaults to False.

        Returns:
            torch.Tensor: The evaluated occupied DOS
                (shape: [E], or [M, E] if `k_resolved`).
        """

        if self.g_lesser is None:
            raise ValueError('Cannot calculate occupied DOS without g_lesser.')

        g_lesser_batches = self._as_batches(self.g_lesser)

        if k_resolved:
            per_batch = []
            for g_lesser_batch in g_lesser_batches:
                g_lesser = self._tensor(g_lesser_batch)
                per_batch.append(
                    (1 / (2 * torch.pi)) *
                    torch.diagonal(g_lesser.imag, dim1=-2, dim2=-1).sum(dim=-1) *
                    band_mult
                )
            return torch.cat(per_batch, dim=0)

        Nk = 0
        occ_dos = 0

        for g_lesser_batch in g_lesser_batches:
            g_lesser = self._tensor(g_lesser_batch)
            occ_dos += (1 / (2 * torch.pi)) * (
                torch.diagonal(g_lesser.imag, dim1=-2, dim2=-1).sum(dim=(-3, -1))
            )
            Nk += g_lesser.shape[-4]

        occ_dos *= band_mult / Nk

        return occ_dos

    def bond_current_density(
            self,
            band_mult: float = 1.0,
            greens_function: str = 'lesser',
            filter: torch.Tensor | list[torch.Tensor] | None = None,
            k_resolved: bool = False) -> torch.Tensor:
        """Evaluates the energy-resolved bond current density matrix from ``G^<``.

        The returned tensor corresponds to an antisymmetrized localized-basis
        bond-current expression in an orbital tight-binding basis.

        .. math::

            J_{i\\to j}(E) \\propto \\mathrm{Im}\\left[
                H_{ij} G^{</>}_{ji}(E) - H_{ji} G^{</>}_{ij}(E)
            \\right]

        using the Hamiltonian stored in this result object. 
        
        With the internal
        convention ``q = \\hbar = 1``, this normalization reports the spectral
        bond current in units of the charge-conductance quantum ``e^2 / h``.
        Any spin degeneracy should be folded into ``band_mult``.

        Args:
            band_mult (float, optional): Band multiplicity factor. Defaults to
                1.0.
            greens_function (str, optional): Which correlation Green's
                function to use in the bond-current expression. Supported
                values are ``'lesser'`` and ``'greater'``. Defaults to
                ``'lesser'``.
            filter (torch.Tensor | list[torch.Tensor] | None, optional):
                Element-wise weighting applied to ``G^<`` before evaluating the
                bond current. The filter must match the full ``g_lesser`` shape
                ``[K, E, N, N]`` or be provided as a list of k-batched tensors
                matching the stored batching layout. Defaults to None.
            k_resolved (bool, optional): If True, return one bond-current matrix
                per sampled ``k`` point. Otherwise average over the sampled
                ``k`` points. Defaults to False.

        Returns:
            torch.Tensor: Energy-resolved bond-current matrix with shape
                ``[K, E, N, N]`` if ``k_resolved`` is True, otherwise
                ``[E, N, N]``. Values are normalized to ``e^2 / h`` units.
        """

        if self.hamiltonian is None:
            raise ValueError('Cannot calculate bond current without hamiltonian.')

        hamiltonian_batches = self._as_batches(self.hamiltonian)
        if greens_function == 'lesser':
            if self.g_lesser is None:
                raise ValueError('Cannot calculate bond current without g_lesser.')
            g_batches = self._as_batches(self.g_lesser)
        elif greens_function == 'greater':
            if self.g_greater is None:
                raise ValueError('Cannot calculate bond current without g_greater.')
            g_batches = self._as_batches(self.g_greater)
        else:
            raise ValueError(
                "Unsupported greens_function. Expected 'lesser' or 'greater'; "
                f"got {greens_function!r}."
            )

        if len(hamiltonian_batches) != len(g_batches):
            raise ValueError(
                'Hamiltonian batches and Green\'s-function batches must have the same length.'
            )

        filter_batches = [None] * len(g_batches)
        if filter is not None:
            if isinstance(filter, list):
                if len(filter) != len(g_batches):
                    raise ValueError(
                        'Filter batches and Green\'s-function batches must have the same length.'
                    )
                filter_batches = []
                for filt_batch, g_batch in zip(filter, g_batches):
                    filt_t = self._tensor(filt_batch)
                    g_t = self._tensor(g_batch)
                    if filt_t.shape != g_t.shape:
                        raise ValueError(
                            'Filter batch shape mismatch: '
                            f'expected {tuple(g_t.shape)}, got {tuple(filt_t.shape)}.'
                        )
                    filter_batches.append(filt_t.to(device=g_t.device, dtype=g_t.dtype))
            else:
                filter_t = self._tensor(filter)
                if filter_t.ndim != 4:
                    raise ValueError(
                        'Expected filter tensor to have shape [K, E, N, N]; '
                        f'got shape {tuple(filter_t.shape)}.'
                    )

                k_sizes = [self._tensor(g_batch).shape[0] for g_batch in g_batches]
                if filter_t.shape[0] != sum(k_sizes):
                    raise ValueError(
                        'Unbatched filter k-dimension must match the total number of sampled k points: '
                        f'expected {sum(k_sizes)}, got {filter_t.shape[0]}.'
                    )

                split_filters = list(torch.split(filter_t, k_sizes, dim=0))
                filter_batches = []
                for filt_t, g_batch in zip(split_filters, g_batches):
                    g_t = self._tensor(g_batch)
                    if filt_t.shape != g_t.shape:
                        raise ValueError(
                            'Filter batch shape mismatch after k-splitting: '
                            f'expected {tuple(g_t.shape)}, got {tuple(filt_t.shape)}.'
                        )
                    filter_batches.append(filt_t.to(device=g_t.device, dtype=g_t.dtype))

        J_batches = []
        J_sum = 0
        Nk = 0

        for hamiltonian_batch, g_batch, filter_batch in zip(hamiltonian_batches, g_batches, filter_batches):
            H = self._tensor(hamiltonian_batch)      # [K, N, N]
            g_corr = self._tensor(g_batch)  # [K, E, N, N]

            if H.ndim != 3:
                raise ValueError(
                    'Expected hamiltonian batch to have shape [K, N, N]; '
                    f'got shape {tuple(H.shape)}.'
                )
            if g_corr.ndim != 4:
                raise ValueError(
                    'Expected Green\'s-function batch to have shape [K, E, N, N]; '
                    f'got shape {tuple(g_corr.shape)}.'
                )
            if H.shape[0] != g_corr.shape[0] or H.shape[-2:] != g_corr.shape[-2:]:
                raise ValueError(
                    'Hamiltonian and Green\'s-function shapes are incompatible: '
                    f'{tuple(H.shape)} vs {tuple(g_corr.shape)}.'
                )

            # Scale so that integrating J(E) over energy yields a dimensionless
            # current in the internal convention q = ħ = 1 (i.e. e^2 / h
            # units). In this convention energy integrals are taken with the
            # factor 1/(2π) that matches other routines in this module.
            prefactor = torch.as_tensor(
                1.0 / (2.0 * torch.pi),
                device=g_corr.device,
                dtype=g_corr.real.dtype,
            )

            if filter_batch is not None:
                g_corr = g_corr * filter_batch

            # Use an explicitly antisymmetrized bond-current expression:
            #   J_ij(E) ~ Im[ H_ij G_ji(E) - H_ji G_ij(E) ]
            # This yields J_ij = -J_ji and J_ii = 0 (up to numerical error).
            HGt = H.unsqueeze(-3) * g_corr.transpose(-2, -1)             # H_ij G_ji
            HtG = H.transpose(-2, -1).unsqueeze(-3) * g_corr             # H_ji G_ij
            J_batch = prefactor * (HGt - HtG).imag
            J_batch = J_batch * band_mult

            if k_resolved:
                J_batches.append(J_batch)
            else:
                J_sum = J_sum + J_batch.sum(dim=0)
                Nk += J_batch.shape[0]

        return torch.cat(J_batches, dim=0) if k_resolved else (J_sum / Nk)

    def cut_current_density(
            self,
            in_orbitals: torch.Tensor | list[int],
            out_orbitals: torch.Tensor | list[int],
            band_mult: float = 1.0,
            greens_function: str = 'lesser',
            filter: torch.Tensor | list[torch.Tensor] | None = None,
            return_matrix: bool = False,
            k_resolved: bool = False) -> torch.Tensor:
        """Evaluates the bond current density across a user-defined orbital cut.

        The cut is defined by two orbital index sets. The method extracts the
        submatrix of :meth:`bond_current_density` spanning all pairs
        ``(i in in_orbitals, j in out_orbitals)`` and either:

        - returns that submatrix directly when ``return_matrix`` is True, or
        - sums over the submatrix entries to give the spectral cut current
          :math:`J_{cut}(E)`.

        Args:
            in_orbitals (torch.Tensor | list[int]): Orbital indices on the
                "input" side of the cut.
            out_orbitals (torch.Tensor | list[int]): Orbital indices on the
                "output" side of the cut.
            band_mult (float, optional): Band multiplicity factor. Defaults to
                1.0.
            greens_function (str, optional): Which correlation Green's
                function to use in the bond-current expression. Supported
                values are ``'lesser'`` and ``'greater'``. Defaults to
                ``'lesser'``.
            filter (torch.Tensor | list[torch.Tensor] | None, optional):
                Element-wise weighting applied to ``G^<`` before evaluating the
                cut current. The filter must match the full ``g_lesser`` shape
                ``[K, E, N, N]`` or be provided as a list of tensors matching
                the stored k-batching. Defaults to None.
            return_matrix (bool, optional): If True, return the selected
                cut-current submatrix rather than summing over its entries.
                Defaults to False.
            k_resolved (bool, optional): If True, return one result per sampled
                ``k`` point. Otherwise average over sampled ``k`` points.
                Defaults to False.

        Returns:
            torch.Tensor: If ``return_matrix`` is False, returns the spectral
                cut current with shape ``[K, E]`` when ``k_resolved`` is True,
                otherwise ``[E]``. If ``return_matrix`` is True, returns the
                selected bond-current submatrix with shape
                ``[K, E, N_in, N_out]`` or ``[E, N_in, N_out]``.
        """

        if isinstance(in_orbitals, list):
            in_orbitals = torch.tensor(in_orbitals, device=self._device(), dtype=torch.int64)
        elif isinstance(in_orbitals, torch.Tensor):
            in_orbitals = in_orbitals.to(device=self._device(), dtype=torch.int64)
        else:
            raise ValueError(
                'Expected in_orbitals to be a list[int] or 1D torch.Tensor; '
                f'got {type(in_orbitals)}.'
            )

        if isinstance(out_orbitals, list):
            out_orbitals = torch.tensor(out_orbitals, device=self._device(), dtype=torch.int64)
        elif isinstance(out_orbitals, torch.Tensor):
            out_orbitals = out_orbitals.to(device=self._device(), dtype=torch.int64)
        else:
            raise ValueError(
                'Expected out_orbitals to be a list[int] or 1D torch.Tensor; '
                f'got {type(out_orbitals)}.'
            )

        if in_orbitals.ndim != 1:
            raise ValueError(
                'Expected in_orbitals to have ndim == 1; '
                f'got ndim == {in_orbitals.ndim}.'
            )
        if out_orbitals.ndim != 1:
            raise ValueError(
                'Expected out_orbitals to have ndim == 1; '
                f'got ndim == {out_orbitals.ndim}.'
            )
        if in_orbitals.numel() == 0 or out_orbitals.numel() == 0:
            raise ValueError('Expected in_orbitals and out_orbitals to be non-empty.')

        J = self.bond_current_density(
            band_mult=band_mult,
            greens_function=greens_function,
            filter=filter,
            k_resolved=k_resolved,
        )

        J_cut = J.index_select(dim=-2, index=in_orbitals)
        J_cut = J_cut.index_select(dim=-1, index=out_orbitals)

        if return_matrix:
            return J_cut

        return J_cut.sum(dim=(-2, -1))

    def coherent_transmission(
            self,
            in_idx: torch.IntTensor | int,
            out_idx: torch.IntTensor | int,
            band_mult: torch.Tensor | float = 1.0,
            k_resolved: bool = False) -> torch.Tensor:
        """Computes the coherent transmission coefficients of
        the device, evaluated at the given energies.

        The transmission coefficients are evaluated between the 
        index of an input lead and an output lead for the given 
        energy values. They are physically independent of the
        lead carrier occupations.

        Args:
            in_idx (torch.IntTensor | int): The input lead(s)
                (shape: [B]).
            out_idx (torch.IntTensor | int): The output lead(s)
                (shape: [B]).
            band_mult (float, optional): Band multiplicity. Defaults to 1.0.
            k_resolved (bool, optional): If True, the transmission is
                returned without integrating over the sampled
                :math:`\\mathbf{k}` vectors. Defaults to False.
        
        Returns:
            torch.Tensor: The site transmission matrix 
                (shape: [B, E], or [B, M, E] if `k_resolved` is `True`)
        """

        if self.lead_self_energies is None:
            raise ValueError('Cannot calculate coherent transmission without lead self energies.')

        in_idx, out_idx, _ = self._resolve_in_out_idxs(in_idx, out_idx)

        lead_se_batches = self._as_batches(self.lead_self_energies)
        g_ret_batches = self._as_batches(self.g_ret)

        if len(lead_se_batches) != len(g_ret_batches):
            raise ValueError('Lead self-energy batches and Green\'s function batches must have the same length.')

        T_batches = []
        T_sum = 0
        Nk = 0

        for lead_se_batch, g_ret_batch in zip(lead_se_batches, g_ret_batches):
            lead_se = self._tensor(lead_se_batch)
            g_ret = self._tensor(g_ret_batch)

            gammas = 1.j*(lead_se - lead_se.mH)
            gammas_in = gammas[in_idx]
            gammas_out = gammas[out_idx]

            # compute T = Tr(gamma_a @ g_ret @ gamma_b @ g_adv)
            T_batch = torch.einsum(
                '...ij, ...jk, ...kl, ...li -> ...',
                gammas_in,
                g_ret,
                gammas_out,
                g_ret.mH
            ).real

            if k_resolved:
                T_batches.append(T_batch)
            else:
                T_sum = T_sum + T_batch.sum(-2)
                Nk += T_batch.shape[-2]

        T = torch.cat(T_batches, dim=-2) if k_resolved else (T_sum / Nk)

        # apply band multiplicity
        T *= band_mult

        # zero out any transition matrix terms where the in and out 
        # leads are the same
        same_sites = (in_idx == out_idx)
        if isinstance(same_sites, bool):
            if same_sites:
                T.zero_()
        else:
            T[same_sites] = 0

        return T

    def coherent_current(
            self,
            in_idx: torch.IntTensor | int,
            out_idx: torch.IntTensor | int,
            band_mult: torch.Tensor | float = 1.0,
            k_resolved: bool = False) -> torch.Tensor:
        """Computes the coherent current between two leads in the
        the device. The coherent current describes the 
        contribution to the total current associated with 
        quantum ballistic conduction directly between leads and is
        independent of any scattering mechanisms.

        With the internal
        convention ``q = \\hbar = 1``, the current is returned
        in units of the charge-conductance quantum ``e^2 / h``.
        Any spin degeneracy should be folded into ``band_mult``.

        Args:
            in_idx (torch.IntTensor | int): The input lead(s)
                (shape: [B]).
            out_idx (torch.IntTensor | int): The output lead(s)
                (shape: [B]).
            energies (torch.Tensor): The energies at which the current
                should be evaluated (shape: [E]).
            band_mult (float, optional): Band multiplicity. Defaults to 1.0.
            k_resolved (bool, optional): If True, the transmission matrix is
                returned without integrating over the sampled
                :math:`\\mathbf{k}` vectors. Defaults to False.

        Returns:
            torch.Tensor: The site transmission matrix 
                (shape: [B, E], or [B, M, E] if `k_resolved` is `True`)
        """

        if self.lead_self_energies is None:
            raise ValueError('Cannot calculate current without lead self energies.')

        lead_occ = None
        energies = self._tensor(self.energies)
        
        in_idx, out_idx, sum_result = self._resolve_in_out_idxs(in_idx, out_idx)

        if self.lead_occupation_probs is not None:
            if isinstance(self.lead_occupation_probs, list):
                lead_occ = torch.stack([self._tensor(o) for o in self.lead_occupation_probs], dim=0)
            else:
                lead_occ = self._tensor(self.lead_occupation_probs)
        

            
        in_f = lead_occ[in_idx]
        out_f = lead_occ[out_idx]
        if in_f is None or out_f is None:
            raise ValueError('Occupation probabilities must be provided to compute coherent current.')

        T = self.coherent_transmission(
            in_idx=in_idx,
            out_idx=out_idx,
            band_mult=band_mult,
            k_resolved=k_resolved)


        I = torch.trapz((T*(in_f - out_f)), energies, dim=-1) / (2*torch.pi)

        return I.sum(0) if sum_result else I
        

    def current(
            self,
            in_idx: torch.IntTensor | int,
            out_idx: torch.IntTensor | int,
            band_mult: torch.Tensor | float = 1.0,
            k_resolved: bool = False,
            **solver_kwargs) -> torch.Tensor:
        """Computes the current flowing out of a lead into the device by
        integrating over sampled energy values.

        With the internal
        convention ``q = \\hbar = 1``, the current is returned
        in units of the charge-conductance quantum ``e^2 / h``.
        Any spin degeneracy should be folded into ``band_mult``.

        For best results, energies should be sampled uniformly from a few
        multiples of `kt` below the lowest `chemical_potential` value to
        to a few multiples of `kt` above the highest `chemical_potential` 
        value.

        Args:
            idx (int): The index of the boundary.

        Returns:
           torch.Tensor: The current coming out of boundary into the site
                (shape: [], or [M] if `k_resolved` is `True`)
        """

        if self.lead_self_energies is None:
            raise ValueError('Cannot calculate current without lead self energies.')
        if self.g_lesser is None or self.g_greater is None:
            raise ValueError('Cannot calculate current without g_lesser and g_greater.')
        if self.lead_occupation_probs is None:
            raise ValueError('Occupation probabilities must be provided to compute current.')

        energies = self._tensor(self.energies)
        in_idx, out_idx, sum_result = self._resolve_in_out_idxs(in_idx, out_idx)

        lead_occ = self._tensor(self.lead_occupation_probs)
        if lead_occ.ndim not in (2, 3):
            raise ValueError(
                'Expected lead occupation probabilities to have shape [L, E] or [L, K, E]; '
                f'got shape {tuple(lead_occ.shape)}.'
            )

        lead_se_batches = self._as_batches(self.lead_self_energies)
        g_lesser_batches = self._as_batches(self.g_lesser)
        g_greater_batches = self._as_batches(self.g_greater)

        if not (len(lead_se_batches) == len(g_lesser_batches) == len(g_greater_batches)):
            raise ValueError(
                'Lead self-energies, g_lesser, and g_greater batches must have the same length.'
            )

        I_batches = []
        I_sum = 0
        Nk = 0
        k0 = 0

        for lead_se_batch, g_lesser_batch, g_greater_batch in zip(
                lead_se_batches, g_lesser_batches, g_greater_batches):

            lead_se = self._tensor(lead_se_batch)      # [L, Kb, E, N, N]
            g_lesser = self._tensor(g_lesser_batch)    # [Kb, E, N, N]
            g_greater = self._tensor(g_greater_batch)  # [Kb, E, N, N]

            if g_lesser.shape != g_greater.shape:
                raise ValueError('Expected g_lesser and g_greater to have matching shapes per batch.')
            if lead_se.shape[-4] != g_lesser.shape[-4]:
                raise ValueError('Batch k-dim mismatch between lead self-energy and correlation GFs.')

            k_batch = g_lesser.shape[-4]

            if lead_occ.ndim == 2:
                f_batch = lead_occ
            else:
                f_batch = lead_occ[:, k0:k0 + k_batch, :]
                if f_batch.shape[1] != k_batch:
                    raise ValueError(
                        'Lead occupation probabilities [L, K, E] do not match the current k-batching.'
                    )
            k0 += k_batch

            gamma = 1.j * (lead_se - lead_se.mH)

            gamma_in = gamma[in_idx]
            gamma_out = gamma[out_idx]

            if f_batch.ndim == 2:
                f_in = f_batch[in_idx].unsqueeze(-3).unsqueeze(-1).unsqueeze(-1)
                f_out = f_batch[out_idx].unsqueeze(-3).unsqueeze(-1).unsqueeze(-1)
            else:
                f_in = f_batch[in_idx].unsqueeze(-1).unsqueeze(-1)
                f_out = f_batch[out_idx].unsqueeze(-1).unsqueeze(-1)

            sigma_in_lesser = 1.j * gamma_in * f_in
            sigma_in_greater = -1.j * gamma_in * (1 - f_in)
            sigma_out_lesser = 1.j * gamma_out * f_out
            sigma_out_greater = -1.j * gamma_out * (1 - f_out)

            g_l = g_lesser.unsqueeze(0)
            g_g = g_greater.unsqueeze(0)

            # Symmetric two-terminal Meir-Wingreen current spectral density.
            # Use explicit batched matrix products for clarity/performance.
            j_in = (sigma_in_lesser @ g_g).diagonal(dim1=-2, dim2=-1).sum(-1) - \
                   (sigma_in_greater @ g_l).diagonal(dim1=-2, dim2=-1).sum(-1)
            j_out = (sigma_out_lesser @ g_g).diagonal(dim1=-2, dim2=-1).sum(-1) - \
                    (sigma_out_greater @ g_l).diagonal(dim1=-2, dim2=-1).sum(-1)
            jE = 0.5 * (j_in - j_out).real

            I_batch = torch.trapz(jE, energies, dim=-1) * (band_mult / (2 * torch.pi))

            if k_resolved:
                I_batches.append(I_batch)
            else:
                I_sum = I_sum + I_batch.sum(-1)
                Nk += k_batch

        I = torch.cat(I_batches, dim=-1) if k_resolved else (I_sum / Nk)

        return I.sum(0) if sum_result else I
        
