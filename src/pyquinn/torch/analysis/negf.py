import torch
from abc import ABC, abstractmethod
from pyquinn.torch.types import TorchAtomGraph
from pyquinn.torch.models import TorchBandStructureModel

from pyquinn.io.stdout import print_overwrite

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

    def solve(
            self,
            energies: torch.Tensor,
            hamiltonian_matrix: torch.Tensor,
            self_energies: torch.Tensor | None = None,
            overlap_matrix: torch.Tensor | None = None,
            eta: float = 1e-5) -> torch.Tensor:
        
        E = energies.view(-1, 1, 1)
        H_site = hamiltonian_matrix.unsqueeze(-3)
        dtype = hamiltonian_matrix.dtype
        device = hamiltonian_matrix.device
        I = torch.eye(hamiltonian_matrix.shape[-1], dtype=dtype, device=device)
        S = I if overlap_matrix is None else overlap_matrix

        H_se = 0 if self_energies is None else self_energies.sum(0)

        A_ret = (E*S + 1.j*eta*I) - (H_site + H_se)
        G_ret = torch.linalg.inv(A_ret)

        return G_ret


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

        # Conduction/valence masks along the energy axis.
        theta_c = (energies.view(-1, 1, 1) > self.valence_cutoff).to(energies.dtype)
        theta_v = 1 - theta_c

        lead_ndim = g_lesser.ndim - 3
        theta_c = theta_c.view((1,) * lead_ndim + (-1, 1, 1))
        theta_v = theta_v.view((1,) * lead_ndim + (-1, 1, 1))

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
        sigL = sigL + sL
        sigG = sigG + sG
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
            k = k_hat.to(device=device, dtype=torch.float64)
            k = k / (k.norm() + 1e-30)
            I3 = torch.eye(3, device=device, dtype=torch.float64)
            rho = (I3 - torch.outer(k, k)) / 2.0
        else:
            rho = torch.eye(3, device=device, dtype=torch.float64) / 3.0
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
        torch.Tensor: Real transformed broadening matrix operator (shape: [..., E, N, N]).
    """
    
    X = torch.fft.fft(x, dim=-3)
    n = x.shape[-3]

    freqs = torch.fft.fftfreq(n)
    sgn = torch.sign(freqs)

    H = torch.fft.ifft(-1j * sgn.view(-1,1,1) * X, dim=-3)
    return H.real


def sancho_rubio(
        energies: torch.Tensor,
        coupling_matrix: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5,
        max_iter: int = 10000,
        dyson_solver: DysonSolver | None = None,
        verbosity: int = 0,
        tol: float = 1e-6,
        eta_decay=0.1,
        eta_start=1.0) -> torch.Tensor:
    """Applies the Sancho-Rubio method that numerically solves 
    for the surface Green's function of a semi-infinite boundary.

    .. warning::
            This method does not currently support backward passes,
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
    
    E = energies.view(-1, 1, 1)
    H = hamiltonian_matrix.unsqueeze(-3)
    I = torch.eye(hamiltonian_matrix.shape[-1])
    S = I if overlap_matrix is None else overlap_matrix
    V = coupling_matrix.unsqueeze(-3)
    Z0 = E*S
    eta_current = max(eta, eta_start)

    with torch.no_grad():
        Z = Z0 + 1.j*eta_current*I
        eps_surf = H
        eps = H
        alpha = V
        beta = V.mH

        for i in range(max_iter):

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
            
            if max_err < tol:
                if eta_current > eta:
                    eta_current = max(eta, eta_current * eta_decay)
                    Z = Z0 + 1.j*eta_current*I
                    eps_surf_new = H
                    eps_new = H
                    alpha_new = V
                    beta_new = V.mH
                else:
                    break

            if verbosity >= 2:
                print_overwrite(f'Sancho Rubio (iter: {i}, eta: {eta_current:5e}, err: {max_err:5e}) ')

            alpha = alpha_new
            beta = beta_new
            eps = eps_new
            eps_surf = eps_surf_new

        g_surf = torch.linalg.pinv(Z - eps_surf)

        if  max_err > tol or eta_current > eta or torch.isnan(max_err):
            if verbosity >= 1:
                print('[Failed]')

            raise RuntimeError('Sancho-Rubio failed to Converge. (iter: {i}, eta: {eta_current:5e}, err: {max_err:5e}).')
        elif verbosity >= 1:
            print('[Converged]')
            

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

    def __init__(
            self,
            device: TorchAtomGraph | None,
            leads: list[TorchAtomGraph] | None,
            orientations: list[torch.IntTensor] | None = None):
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

        # validate boundary graphs and orientations
        for i, (boundary, orientation) in \
                enumerate(zip(self.leads, self.orientations)):
            self._validate_boundary_atomgraph(
                boundary,
                orientation,
                index=i
            )

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

    # def solve_lead_self_energies(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor | None = None,
    #         eta: float = 1e-5,
    #         boundary_gfs: list[torch.Tensor] | None = None,
    #         **solver_kwargs) -> tuple[torch.Tensor, list[torch.Tensor]]:
    #     """Returns a the site Hamiltonian and a tensor of self-energy 
    #     correlations to the device Hamiltonian based on the computed 
    #     surface Green's functions of each lead.

        

    #     Args:
    #         energies (torch.Tensor): The energies at which the Green's function 
    #             should be evaluated (shape: [E]).
    #         k_pts (torch.Tensor | None, optional): Reciprocal space 
    #             :math:`\\mathbf{k}` vectors used to sample crystal momentum
    #             degrees of freedom parallel to the boundary (shape: [M, 3]). 
    #             Defaults to None.
    #         eta (float, optional): Broadening parameter of the Green's 
    #             function. Increasing this parameter may make any surface 
    #             Green's function methods converge faster for semi-infinite
    #             boundaries at the cost of broadening the energy spectrum. 
    #             Defaults to 1e-5.
    #         boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
    #             surface Green's functions returned from a call to 
    #             :meth:`NEGF.surface_greens_functions`. Defaults to None.
    #         dyson_solver (DysonSolver | None, optional): Custom Dyson solver
    #             used when building the retarded Green's function. Defaults
    #             to the dense inverse solver.

    #     Returns:
    #         tuple[torch.Tensor, list[torch.Tensor]]: A tuple containing the 
    #         following:

    #         - site_hamitonian (torch.Tensor): 
    #             The site Hamiltonian matrix (shape: [M, N, N])
    #         - self_energies (torch.Tensor):
    #             The self-energy perturbations (shape: 
    #             [N_boundaries, E, M, N, N])

    #     """

    #     k_pts = self._resolve_k_pts(k_pts, energies)

    #     if self.site is None:
    #         raise ValueError(
    #             'The site AtomGraph must be defined '
    #             'to compute site self-energies.')

    #     H_sizes = [
    #         self.model.orbital_indices(g)[0].shape[0]
    #         for g in self.leads
    #     ]

    #     # determine indices of components of the system Hamiltonian
    #     H_site_size = self.model.orbital_indices(self.site)[0].shape[0]
    #     H_boundary_idxs = []
    #     offset = H_site_size
    #     for size in H_sizes:
    #         H_boundary_idxs.append(torch.arange(
    #             offset, offset + size,
    #             device=energies.device))
    #         offset += size

    #     # compute system graph and Hamiltonian matrix
    #     system = self.system_graph()
    #     H_system = self.model.hamiltonian_matrix(system, k_pts=k_pts)
    #     H_site = H_system[:, :H_site_size, :H_site_size]
        
    #     # compute boundary Greens functions (if not supplied):
    #     if boundary_gfs is None:
    #         boundary_gfs = self.boundary_greens_functions(
    #             energies=energies,
    #             k_pts=k_pts,
    #             eta=eta,
    #             **solver_kwargs
    #         )
        
    #     if len(boundary_gfs) != len(self.leads):
    #         raise ValueError(
    #             f'Expected a list of {len(self.leads)} '
    #             f'boundary Greens functions. (got list of '
    #             f'size: {len(boundary_gfs)}')

    #     # extract device coupling and compute self-energy:
    #     self_energies_list = []
    #     for idxs, gf in \
    #             zip(H_boundary_idxs, boundary_gfs):
    #         V = H_system[:, :H_site_size, idxs].unsqueeze(-3)

    #         self_energy = V @ gf @ V.mH

    #         self_energies_list.append(self_energy)
        
    #     self_energies = torch.stack(self_energies_list)

    #     return H_site, self_energies

    # def device_greens_function(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor | None = None,
    #         eta: float = 1e-5,
    #         dyson_solver: DysonSolver | None = None,
    #         boundary_gfs: list[torch.Tensor] | None = None,
    #         **solver_kwargs) -> torch.Tensor:
    #     """Returns the site Green's function with perturbations due to 
    #     boundary coupling.

    #     Args:
    #         energies (torch.Tensor): The energies at which the Green's function 
    #             should be evaluated (shape: [E]).
    #         k_pts (torch.Tensor | None, optional): Reciprocal space 
    #             :math:`\\mathbf{k}` vectors used to sample crystal momentum
    #             degrees of freedom parallel to the boundary (shape: [M, 3]). 
    #             Defaults to None.
    #         eta (float, optional): Broadening parameter of the Green's 
    #             function. Increasing this parameter may make any surface 
    #             Green's function methods converge faster for semi-infinite
    #             boundaries at the cost of broadening the energy spectrum. 
    #             Defaults to 1e-5.
    #         boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
    #             surface Green's functions returned from a call to 
    #             :meth:`NEGF.surface_greens_functions`. Defaults to None.
    #         dyson_solver (DysonSolver | None, optional): Custom Dyson solver
    #             used to compute the perturbed site Green's function.
    #             Defaults to the dense inverse implementation.

    #     Returns:
    #         torch.Tensor: The site Green's function (shape: [M, E, N, N]).
    #     """

    #     k_pts = self._resolve_k_pts(k_pts, energies)

    #     H_site, self_energies = self.device_lead_self_energies(
    #         energies=energies,
    #         k_pts=k_pts,
    #         eta=eta,
    #         boundary_gfs=boundary_gfs,
    #         **solver_kwargs)

    #     device_gf = greens_function(
    #         energies=energies,
    #         hamiltonian_matrix=H_site,
    #         self_energies=self_energies,
    #         eta=eta,
    #         dyson_solver=dyson_solver,
    #     )

    #     return device_gf

    # def _lesser_site_greens_function_data(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor,
    #         eta: float,
    #         chemical_potentials: list[torch.Tensor] |
    #         list[float] | None = None,
    #         kt: torch.Tensor | float = 1e-5,
    #         boundary_gfs: list[torch.Tensor] | None = None,
    #         greater: bool = False,
    #         dyson_solver: DysonSolver | None = None,
    #         **solver_kwargs) -> list[torch.Tensor]:

    #     # compute self-energies
    #     H_site, self_energies = self.site_self_energies(
    #         energies=energies,
    #         k_pts=k_pts,
    #         eta=eta,
    #         boundary_gfs=boundary_gfs,
    #         **solver_kwargs)

    #     # resolve chemical potentials and occupation probabilities
    #     if chemical_potentials is None:
    #         chemical_potentials = [0.0]*len(self_energies)

    #     if len(chemical_potentials) != len(self.leads):
    #         raise ValueError(
    #             'Expected len(chemical_potentials) == len(self.leads);'
    #             f' got {len(chemical_potentials)} != {len(self.leads)}.'
    #         )

    #     occupation_probs = torch.stack([
    #         1.0 / (1.0 + torch.exp((energies - mu)/kt))
    #         for mu in chemical_potentials
    #     ])
    #     # change to greater Green's function if desired
    #     if greater:
    #         occupation_probs = 1 - occupation_probs
        
    #     occupations = occupation_probs.view(
    #         occupation_probs.shape + (1, 1))


    #     # compute occupation-weighted broadening matrices

    #     # self_energies: [B, K, E, N, N]
    #     # occupations:   [B, E] -> [B, 1, E, 1, 1]
    #     # sigma_in:      sum over boundaries -> [K, E, N, N]
    #     occ = occupations.unsqueeze(-4)
    #     sigma_in = 1.j * ((self_energies - self_energies.mH) * occ).sum(0)

    #     # compute retarded greens function
    #     g_ret = perturbed_greens_function(
    #         energies=energies,
    #         hamiltonian_matrix=H_site,
    #         self_energies=self_energies,
    #         eta=eta,
    #         dyson_solver=dyson_solver,
    #     )

    #     # assemble lesser greens function
    #     lesser_gf = g_ret @ sigma_in @ g_ret.mH

    #     return H_site, self_energies, lesser_gf

    # def lesser_site_greens_function(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor | None = None,
    #         eta: float = 1e-5,
    #         chemical_potentials: list[torch.Tensor] |
    #         list[float] | None = None,
    #         kt: torch.Tensor | float = 1e-5,
    #         boundary_gfs: list[torch.Tensor] | None = None,
    #         dyson_solver: DysonSolver | None = None,
    #         **solver_kwargs) -> torch.Tensor:
    #     """Computes the lesser Green's function of the site.

    #     Args:
    #         energies (torch.Tensor): The energies at which the density of 
    #             states should be evaluated (shape: [E]).
    #         k_pts (torch.Tensor | None, optional): Reciprocal space 
    #             :math:`\\mathbf{k}` vectors used to sample crystal momentum
    #             degrees of freedom parallel to the boundary (shape: [M, 3]). 
    #             Defaults to None.
    #         eta (float, optional): Broadening parameter of the Green's 
    #             function.
    #         chemical_potentials (list[torch.Tensor] | 
    #             list[float] | None, optional): 
    #             List of chemical potentials :math:`\\mu` of each boundary. 
    #             Defaults to None.
    #         kt (torch.Tensor | float, optional): 
    #             The Boltzmann constant times the temperature of the device
    #             (:math:`k_BT`). Defaults to 1e-5.
    #         boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
    #             surface Green's functions returned from a call to 
    #             :meth:`NEGF.surface_greens_functions`. Defaults to None.
    #         dyson_solver (DysonSolver | None, optional): Custom Dyson solver
    #             used when building the retarded Green's function. Defaults
    #             to the dense inverse solver.

    #     Returns:
    #         torch.Tensor: The lesser site Green's function
    #             (shape: [M, E, N, N])
    #     """
        
    #     k_pts = self._resolve_k_pts(k_pts, energies)

    #     _, _, lesser_gf = self._lesser_site_greens_function_data(
    #         energies=energies,
    #         k_pts=k_pts,
    #         chemical_potentials=chemical_potentials,
    #         eta=eta,
    #         kt=kt,
    #         boundary_gfs=boundary_gfs,
    #         greater=False,
    #         dyson_solver=dyson_solver,
    #         **solver_kwargs)

    #     return lesser_gf
        
    # def greater_site_greens_function(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor | None = None,
    #         eta: float = 1e-5,
    #         chemical_potentials: list[torch.Tensor] |
    #             list[float] | None = None,
    #         kt: torch.Tensor | float = 1e-5,
    #         boundary_gfs: list[torch.Tensor] | None = None,
    #         dyson_solver: DysonSolver | None = None,
    #         **solver_kwargs) -> torch.Tensor:
    #     """Computes the greater Green's function of the site.

    #     Args:
    #         energies (torch.Tensor): The energies at which the density of 
    #             states should be evaluated (shape: [E]).
    #         k_pts (torch.Tensor | None, optional): Reciprocal space 
    #             :math:`\\mathbf{k}` vectors used to sample crystal momentum
    #             degrees of freedom parallel to the boundary (shape: [M, 3]). 
    #             Defaults to None.
    #         eta (float, optional): Broadening parameter of the Green's 
    #             function.
    #         chemical_potentials (list[torch.Tensor] | 
    #             list[float] | None, optional): 
    #             List of chemical potentials :math:`\\mu` of each boundary. 
    #             Defaults to None.
    #         kt (torch.Tensor | float, optional): 
    #             The Boltzmann constant times the temperature of the device
    #             (:math:`k_BT`). Defaults to 1e-5.
    #         boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
    #             surface Green's functions returned from a call to 
    #             :meth:`NEGF.surface_greens_functions`. Defaults to None.

    #     Returns:
    #         torch.Tensor: The greater site Green's function
    #             (shape: [M, E, N, N])
    #     """

    #     k_pts = self._resolve_k_pts(k_pts, energies)

    #     _, _, greater_gf = self._lesser_site_greens_function_data(
    #         energies=energies,
    #         k_pts=k_pts,
    #         chemical_potentials=chemical_potentials,
    #         eta=eta,
    #         kt=kt,
    #         boundary_gfs=boundary_gfs,
    #         greater=True,
    #         dyson_solver=dyson_solver,
    #         **solver_kwargs)

    #     return greater_gf

            
    # def site_charge_density(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor | None = None,
    #         eta: float = 1e-5,
    #         band_mult: float = 1.0,
    #         chemical_potentials: list[torch.Tensor] |
    #         list[float] | None = None,
    #         kt: float = 1e-5,
    #         lesser_gf: torch.Tensor | None = None,
    #         dyson_solver: DysonSolver | None = None,
    #         **solver_kwargs) -> torch.Tensor:
    #     """Estimates the charge density matrix of the site by integrating over
    #     sampled energy values.

    #     For best results, energies should be sampled uniformly from the lowest 
    #     occupied energy level to a few multiples of `kt` above the highest
    #     `chemical_potential` value.

    #     Args:
    #         energies (torch.Tensor): The energies at which the density of 
    #             states should be evaluated and integrated over (shape: [E]).
    #         k_pts (torch.Tensor | None, optional): Reciprocal space 
    #             :math:`\\mathbf{k}` vectors used to sample crystal momentum
    #             degrees of freedom parallel to the boundary (shape: [M, 3]). 
    #             Defaults to None.
    #         eta (float, optional): Broadening parameter of the Green's 
    #             function.
    #         band_mult (float, optional): Band multiplicity. Defaults to 1.0.
    #         chemical_potentials (list[torch.Tensor] | 
    #             list[float] | None, optional): 
    #             List of chemical potentials :math:`\\mu` of each boundary. 
    #             Defaults to None.
    #         kt (torch.Tensor | float, optional): 
    #             Boltzman's constant times the temperature of the device. 
    #             Defaults to 1e-5.
    #         lesser_gf (torch.Tensor | None, optional): Pre-computed 
    #             lesser Green's function returned from a call to 
    #             :meth:`NEGF.lesser_greens_function`. Defaults to None.
    #         dyson_solver (DysonSolver | None, optional): Custom Dyson solver
    #             used when computing the lesser Green's function. Defaults to
    #             the dense inverse solver.

    #     Returns:
    #         torch.Tensor: The charge density of the site (shape: [N, N])
    #     """

    #     if lesser_gf is None:
    #         lesser_gf = self.lesser_site_greens_function(
    #             energies=energies,
    #             k_pts=k_pts,
    #             eta=eta,
    #             chemical_potentials=chemical_potentials,
    #             kt=kt,
    #             dyson_solver=dyson_solver,
    #             **solver_kwargs
    #         )

    #     sorted_energies, sorted_idx = torch.sort(energies)
    #     sorted_lesser_gf = lesser_gf[..., sorted_idx, :, :]

    #     rho = -1.j * torch.trapezoid(
    #         sorted_lesser_gf, 
    #         sorted_energies,
    #         dim=-3
    #     ) * band_mult / (2*torch.pi)

    #     return rho

    # def site_bond_current(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor | None,
    #         eta: float = 1e-5,
    #         band_mult: float = 1.0,
    #         chemical_potentials: list[torch.Tensor] |
    #         list[float] | None = None,
    #         kt: torch.Tensor | float = 1e-5,
    #         boundary_gfs: list[torch.Tensor] | None = None,
    #         dyson_solver: DysonSolver | None = None,
    #         **solver_kwargs) -> torch.Tensor:
    #     """Evaluates the bond current matrix of the site
    #     as a function of energy.

    #     Args:
    #         energies (torch.Tensor): The energies at which the bond current
    #             should be evaluated (shape: [E]).
    #         k_pts (torch.Tensor | None, optional): Reciprocal space 
    #             :math:`\\mathbf{k}` vectors used to sample crystal momentum
    #             degrees of freedom parallel to the boundary (shape: [M, 3]). 
    #             Defaults to None.
    #         eta (float, optional): Broadening parameter of the Green's 
    #             function.
    #         band_mult (float, optional): Band multiplicity. Defaults to 1.0.
    #         chemical_potentials (list[torch.Tensor] | 
    #             list[float] | None, optional): 
    #             List of chemical potentials :math:`\\mu` of each boundary. 
    #             Defaults to None.
    #         kt (torch.Tensor | float, optional): 
    #             Boltzman's constant times the temperature of the device. 
    #             Defaults to 1e-5.
    #         boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
    #             surface Green's functions returned from a call to 
    #             :meth:`NEGF.surface_greens_functions`. Defaults to None.
    #         dyson_solver (DysonSolver | None, optional): Custom Dyson solver
    #             used when computing the site Green's function for the bond
    #             current. Defaults to the dense inverse solver.

    #     Returns:
    #         torch.Tensor: The bond current matrix of the site 
    #             (shape: [M, E, N, N])
    #     """

    #     k_pts = self._resolve_k_pts(k_pts, energies)

    #     H_site, _, lesser_gf = \
    #         self._lesser_site_greens_function_data(
    #             energies=energies,
    #             k_pts=k_pts,
    #             eta=eta,
    #             chemical_potentials=chemical_potentials,
    #             kt=kt,
    #             boundary_gfs=boundary_gfs,
    #             dyson_solver=dyson_solver,
    #             **solver_kwargs)

    #     I_map = 2*torch.pi * band_mult * \
    #         (H_site * lesser_gf.mT).imag

    #     return I_map

    # def solve_negf(
    #         self,
    #         energies: torch.Tensor,
    #         k_pts: torch.Tensor | None = None,
    #         solver: NEGFSolver | None = None,
    #         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
    
    #     solver = NEGFSolver() if solver is None else 

    #     return solver.solve(
    #         energies=energies,
    #         k_pts=k_pts,
    #         eta=eta,
    #         chemical_potentials=chemical_potentials,
    #         kt=kt,
    #         scatterers=scatterers,
    #         boundary_gfs=boundary_gfs,
    #         dyson_solver=dyson_solver,
    #         return_self_energies=return_self_energies,
    #         **solver_kwargs,
    #     )

class NEGFSolver:
    """Steady-state NEGF solver employing the 
    self-consistent Born approximation (SCBA).

    This solver iterates scattering self-energies :math:`\\Sigma^{</>}` to
    self-consistency using a fixed-point iteration with linear mixing.
    """

    def __init__(
            self,
            negf_device: TorchNEGFDevice,
            model: TorchBandStructureModel,
            mixing_alpha: float = 0.1,
            max_iter: int = 1024,
            tol: float = 1e-6):
        
        self.model = model
        self.negf_device = negf_device
        self.mixing_alpha = mixing_alpha
        self.max_iter = max_iter
        self.tol = tol

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

    def solve_lead_greens_functions(
        self,
        energies: torch.Tensor,
        k_pts: torch.Tensor | None = None,
        eta: float = 1e-5,
        verbosity: int = 0,
        **solver_kwargs) -> list[torch.Tensor]:
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

        k_pts = self._resolve_k_pts(k_pts, energies)

        g_surfaces = []

        for i, (boundary, orientation) in \
                enumerate(zip(self.negf_device.leads, self.negf_device.orientations)):

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
                H_size = self.model.orbital_indices(boundary)[0].shape[0]
                H = self.model.hamiltonian_matrix(sc, k_pts=k_pts)
                H_cell_idxs = torch.arange(
                    H_size, 2*H_size, device=orientation.device)
                H_outer_idxs = H_cell_idxs - H_size

                V = H[:, H_cell_idxs][:, :, H_outer_idxs]
                H0 = H[:, H_cell_idxs][:, :, H_cell_idxs]

                if verbosity >= 1:
                    print(f'Solving surface Green\'s function (lead {i}).')
                
                # compute surface Greens function with Sancho Rubio method
                g_surface = sancho_rubio(
                    energies=energies,
                    coupling_matrix=V,
                    hamiltonian_matrix=H0,
                    eta=eta,
                    verbosity=verbosity,
                    **solver_kwargs
                )
            
            else:

                # if surface is not semi-infinite, use regular Greens function
                if verbosity >= 1:
                    print(f'Solving Green\'s function (lead {i}).')
                H0 = self.model.hamiltonian_matrix(boundary, k_pts=k_pts)
                g_surface = greens_function(
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
        H_site_size = self.model.orbital_indices(self.negf_device.device)[0].shape[0]
        H_boundary_idxs = []
        offset = H_site_size
        for size in H_sizes:
            H_boundary_idxs.append(torch.arange(
                offset, offset + size,
                device=energies.device))
            offset += size

        # compute system graph and Hamiltonian matrix
        system = self.negf_device.system_graph()
        H_system = self.model.hamiltonian_matrix(system, k_pts=k_pts)
        H_site = H_system[:, :H_site_size, :H_site_size]
        
        # compute boundary Greens functions (if not supplied):
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

        # extract device coupling and compute self-energy:
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
            lead_gfs: list[torch.Tensor] | None = None,
            dyson_solver: DysonSolver | None = None,
            verbosity: int = 0,
            **solver_kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if energies.numel() < 2:
            raise ValueError(
                'Energies must have size > 1 and be uniformly spaced to solve for scattering self-energy.'
            )

        if torch.diff(energies).std() > 1e-5 * (energies.max() - energies.min()):
            raise ValueError(
                'Energies must have size > 1 and be uniformly spaced to solve for scattering self-energy.'
            )

        dE = torch.diff(energies).mean()

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
        
        site_gf = greens_function(
            energies=energies,
            hamiltonian_matrix=H_site,
            self_energies=lead_self_energies,
            eta=eta,
            dyson_solver=dyson_solver,
        )
        site_gf_inv = torch.linalg.inv(site_gf)

        num_leads = len(self.negf_device.leads)

        if chemical_potentials is None:
            chemical_potentials = [0.0] * num_leads
        if len(chemical_potentials) != num_leads:
            raise ValueError(
                'Expected len(chemical_potentials) == len(self.negf_device.leads);'
                f' got {len(chemical_potentials)} != {num_leads}.'
            )

        sigma_in = torch.zeros_like(site_gf)
        sigma_out = torch.zeros_like(site_gf)
        occupation_probs = None

        if chemical_potentials:
            occupation_probs = torch.stack([
                1.0 / (1.0 + torch.exp((energies - mu) / kt))
                for mu in chemical_potentials
            ]) 

            f_in = occupation_probs.view(occupation_probs.shape + (1, 1))
            f_out = 1 - f_in
            f_in_e = f_in.unsqueeze(1)
            f_out_e = f_out.unsqueeze(1)

            sigma_in = 1.j * ((lead_self_energies - lead_self_energies.mH) * f_in_e).sum(0)
            sigma_out = 1.j * ((lead_self_energies - lead_self_energies.mH) * f_out_e).sum(0)

        sigma_in_scattering = torch.zeros_like(sigma_in)
        sigma_out_scattering = torch.zeros_like(sigma_in)

        scatterers_resolved = list(scatterers) if scatterers else []

        site_gf_new = site_gf
        g_lesser = site_gf @ sigma_in @ site_gf.mH
        g_greater = site_gf @ sigma_out @ site_gf.mH

        if scatterers_resolved:
            if verbosity >= 1:
                print('Solving self-consistent scattering contributions.')

            with torch.no_grad():
                for i in range(self.max_iter):

                    gamma_scattering = sigma_in_scattering + sigma_out_scattering
                    scattering_self_energy = -0.5j * gamma_scattering + hilbert_transform(gamma_scattering)

                    site_gf_new = torch.linalg.inv(site_gf_inv - scattering_self_energy)

                    g_lesser = site_gf_new @ (sigma_in + sigma_in_scattering) @ site_gf_new.mH
                    g_greater = site_gf_new @ (sigma_out + sigma_out_scattering) @ site_gf_new.mH

                    sigma_in_scattering_new = torch.zeros_like(sigma_in_scattering)
                    sigma_out_scattering_new = torch.zeros_like(sigma_out_scattering)

                    if scatterers_resolved:
                        sigL, sigG = correlation_self_energies_from_scatterers(
                            scatterers=scatterers_resolved,
                            g_lesser=g_lesser,
                            g_greater=g_greater,
                            energies=energies,
                            dE=dE,
                            kt=kt,
                            **solver_kwargs,
                        )
                        sigma_in_scattering_new = sigL
                        sigma_out_scattering_new = sigG

                    residue = torch.abs(sigma_in_scattering - sigma_in_scattering_new).max() + \
                            torch.abs(sigma_out_scattering - sigma_out_scattering_new).max()

                    if verbosity >= 1:
                        print_overwrite(f'SCBA (iter: {i}, err: {residue}) ')

                    if residue < self.tol:
                        sigma_in_scattering = sigma_in_scattering_new
                        sigma_out_scattering = sigma_out_scattering_new
                        break

                    sigma_in_scattering = sigma_in_scattering * (1 - self.mixing_alpha) + \
                                        sigma_in_scattering_new * self.mixing_alpha
                    sigma_out_scattering = sigma_out_scattering * (1 - self.mixing_alpha) + \
                                        sigma_out_scattering_new * self.mixing_alpha

            if residue > self.tol:
                if verbosity >= 1:
                    print('[Failed]')
                raise RuntimeError(f'SCBA failed to Converge. (iter: {i}, err: {residue:5e}).')
            else:
                if verbosity >= 1:
                    print('[Converged]')

        return TorchNEGFResult(
            energies=energies,
            hamiltonian=H_site,
            g_ret=site_gf,
            lead_self_energies=lead_self_energies,
            lead_occupation_probs=occupation_probs,
            lesser_self_energies=sigma_in_scattering,
            greater_self_energies=sigma_out_scattering,
            g_lesser=g_lesser,
            g_greater=g_greater,
            k_pts=k_pts,
        )


class TorchNEGFResult:


    def __init__(
        self,
        energies: torch.Tensor,
        hamiltonian: torch.Tensor,
        g_ret: torch.Tensor,
        lead_self_energies: torch.Tensor | None = None,
        lead_occupation_probs: torch.Tensor | None = None,
        lesser_self_energies: torch.Tensor | None = None,
        greater_self_energies: torch.Tensor | None = None,
        g_lesser: torch.Tensor | None = None,
        g_greater: torch.Tensor | None = None,
        k_pts: torch.Tensor | None = None):
    

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

    def _resolve_in_out_idxs(in_idx, out_idx):

        N = len(self.lead_self_energies)

        if N <= 0:
            return in_idx, out_idx, False
        
        in_unresolved = isinstance(in_idx, None)
        out_unresolved = isinstance(out_idx, None)
        in_const = isinstance(in_idx, int)
        out_const = isinstance(out_idx, int)
        
        if in_unresolved and out_unresolved:
            raise ValueError('Either an input or output lead index must be specified (both are None)')

        # return if in/out indices are both integers
        if in_const and out_const:
            return in_idx, out_idx, False
        
        # broadcast function to fit in_idx onto out_idx, or vice versa:
        def _broadcast(const_idx, tgt_idx, sum_over_tgt):
            if sum_over_tgt:
                tgt_idx = torch.arange(N, device=self.g_ret.device)
                tgt_idx = tgt_idx[tgt_idx != const_idx]
            elif not isinstance(tgt_idx, torch.IntTensor):
                raise ValueError(f'Index type {type(tgt_idx)} incompatible with index type int.')

            src_idx = torch.full_like(tgt_idx, const_idx)
            
            return src_idx, tgt_idx

        # apply broadcasting
        if in_const:
            in_idx, out_idx = _broadcast(in_idx, out_idx, out_unresolved)
        elif out_const:
            out_idx, in_idx = _broadcast(out_idx, in_idx, in_unresolved)
            
        # validate shapes of indices
        if in_idx.ndim != 1:
            raise ValueError(
                'Expected in_idx to have ndim == 1; '
                f'(got ndim == {in_idx.ndim})')
        if out_idx.ndim != 1:
            raise ValueError(
                'Expected in_idx to have ndim == 1; '
                f'(got ndim == {out_idx.ndim})')
        if in_idx.shape != out_idx.shape:
            raise ValueError('Expected in_idx and out_idx to have same shape '
                            f'(got {in_idx.shape} and {out_idx.shape})')

        sum_result = (in_unresolved or out_unresolved)

        return in_idx, out_idx, sum_result

    def ldos(
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

        ldos = (-1/torch.pi) * torch.diagonal(
            self.g_ret.imag, dim1=-2, dim2=-1)

        if not k_resolved:
            ldos = ldos.mean(0)
        ldos *= band_mult

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
        
        dos = (-1/torch.pi) * torch.diagonal(
            self.g_ret.imag, dim1=-2, dim2=-1).sum(-1)

        if not k_resolved:
            dos = dos.mean(0)
        dos *= band_mult

        return dos

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

        if self.lead_self_energies is None or not self.lead_self_energies.numel():
            raise ValueError('Cannot calculate coherent transmission without lead self energies.')

        gammas = 1.j*(self.lead_self_energies - self.lead_self_energies.mH)
        gammas_in = gammas[in_idx]
        gammas_out = gammas[out_idx]

        # compute T = Tr(gamma_a @ g_ret @ gamma_b @ g_adv)
        T = torch.einsum(
            '...ij, ...jk, ...kl, ...li -> ...',
            gammas_in,
            self.g_ret,
            gammas_out,
            self.g_ret.mH
        ).real

        if not k_resolved:
            T = T.mean(-2)

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
            in_occupations: torch.Tensor | None = None,
            out_occupations: torch.Tensor | None = None,
            k_resolved: bool = False) -> torch.Tensor:
        """Computes the coherent current between two leads in the
        the device. The coherent current describes the 
        contribution to the total current associated with 
        quantum ballistic conduction directly between leads.

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

        if not self.lead_self_energies:
            raise ValueError('Cannot calculate current without lead self energies.')

        in_idx, out_idx, sum_result = _resolve_in_out_idxs(in_idx, out_idx)

        in_f = self.lead_occupation_probs[in_idx]
        out_f = self.lead_occupation_probs[out_idx]
        T = coherent_transmission(
            in_idx=in_idx,
            out_idx=out_idx,
            band_mult=band_mult,
            k_resolved=k_resolved)

        I = torch.trapz((T*(in_f - out_f)).sum(0), self.energies, dim=-1)

        return I.sum(0) if sum_result else I
        

    def current(
            self,
            in_idx: torch.IntTensor | int,
            out_idx: torch.IntTensor | int,
            k_pts: torch.Tensor | None = None,
            band_mult: torch.Tensor | float = 2.0,
            k_resolved: bool = False,
            **solver_kwargs) -> torch.Tensor:
        """Computes the current flowing out of a lead into the device by
        integrating over sampled energy values.

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

        if not self.lead_self_energies:
            raise ValueError('Cannot calculate coherent current without lead self energies.')
        
