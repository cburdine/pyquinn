
import numpy as np
import torch
import torch.nn as nn

from pyquinn.torch.nn.atomwise import AtomwiseConv
from pyquinn.torch.nn.eigensolver import Eigensolver
from pyquinn.torch.nn.hamiltonian import Hamiltonian
from pyquinn.torch.nn.lcao.gto import (
    _ELEMENT_CHO_VALENCE_SHELLS,
    GTO_L_MAX,
    _get_cho_orbital_name,
    cho_ng,
    cho_ng_hamiltonian_kinetic_matrix,
    cho_ng_hamiltonian_matrix,
    cho_ng_hamiltonian_potential_matrix,
    cho_ng_kinetic_overlap_matrix,
    cho_ng_momentum_overlap_matrix,
    cho_ng_norm_coeff,
    cho_ng_overlap_matrix,
    cho_ng_position_overlap_matrix,
    cho_overlap_matrix_metadata,
    potential_ng,
)
from pyquinn.torch.types.operator import build_matrix, build_nonperiodic_matrix


class LCAOHamiltonian(Hamiltonian):
    r"""A LCAO (Linear Combination of Atomic Orbitals)
    Hamiltonian Module.

    This module contains a fixed set of Gaussian
    orbital parameters, which are trainable `torch.nn.Parameter`
    values. This module constructs a mean-field Hamiltonian for
    a non-periodic atomic system with using a minimal basis of
    Gaussian cubic harmonic orbitals at each site.
    The Hamiltonian it constructs is real-valued and symmetric.

    When computing the Hamiltonian, this module computes the overlap
    matrices of each neighboring orbital state with an on-site
    pseudopotential that is also expanded as a set of Gaussian functions:

        .. math::
            H_{ij} = \langle \phi_i | -\frac{\hbar^2}{2m_e}\nabla^2 +  V_i + V_j | \phi_j \rangle

    This module currently supports the following eigensolvers:
        - `torch_eigh`

    """

    BASIS_OPTIONS = ["variable", "even_tempered", "fixed"]

    SPIN_OPTIONS = ["paired"]

    ORBITAL_OPERATOR_KERNELS = {
        "overlap": cho_ng_overlap_matrix,
        "kinetic": cho_ng_kinetic_overlap_matrix,
        "position": cho_ng_position_overlap_matrix,
        "momentum_imag": cho_ng_momentum_overlap_matrix,
    }

    HAMILTONIAN_OPERATOR_KERNELS = {
        "hamiltonian": cho_ng_hamiltonian_matrix,
        "kinetic": cho_ng_hamiltonian_kinetic_matrix,
        "potential": cho_ng_hamiltonian_potential_matrix,
    }

    @staticmethod
    def _make_shell_idxs(shells: dict):
        """Constructs a padded tensor that maps [atomic_number,i] -> l.
        l values of -1 should be ignored."""
        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        shell_idxs = torch.full(
            (z_max + 1, n_alpha_sets), -1, dtype=torch.int64
        )

        for z, l_list in shells.items():
            if z <= 0:
                raise ValueError(
                    f"Expected positive atomic number in "
                    f'"shells" dictionary, got: {z}'
                )
            for i, l in enumerate(l_list):
                if l < 0 or l > GTO_L_MAX:
                    raise ValueError(
                        f"Expected 0 <= l <= {GTO_L_MAX} "
                        f"for element {z} "
                        f"in shell {i}, got: l = {l}"
                    )

                shell_idxs[z, i] = l
        return shell_idxs

    @staticmethod
    def _make_logscale_basis_parameters(
        basis: str,
        basis_size: int,
        shells: dict,
        r_max: float = 1.0,
        r_min: float = 0.2,
        c_mean: float = 0.0,
        c_std: float = 0.25,
    ) -> torch.nn.Parameter:
        """Constructs a set of trainable log-scale Gaussian orbital
        parameters."""

        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        if basis not in LCAOHamiltonian.BASIS_OPTIONS:
            raise ValueError(
                f"Basis must be one of {LCAOHamiltonian.BASIS_OPTIONS},"
                f" got: {basis}."
            )
        if basis_size <= 0:
            raise ValueError("Basis size must be a positive integer.")
        if r_max <= 0 or r_min <= 0 or r_min >= r_max:
            raise ValueError(
                "Values r_min and r_max must satisfy" " 0 < r_min < r_max."
            )

        log_alpha_r_max = np.log(1 / (2 * r_max**2))
        log_alpha_r_min = np.log(1 / (2 * r_min**2))
        beta_min_max = (log_alpha_r_min - log_alpha_r_max) / (basis_size - 1)

        if basis == "variable":
            logscale_alphas = nn.Parameter(
                torch.ones(z_max + 1, n_alpha_sets, basis_size)
                * log_alpha_r_max,
                requires_grad=True,
            )
            logscale_betas = nn.Parameter(
                torch.zeros(z_max + 1, n_alpha_sets, 1), requires_grad=False
            )

        elif basis == "even_tempered":
            logscale_alphas = nn.Parameter(
                torch.ones(z_max + 1, n_alpha_sets, 1) * log_alpha_r_max,
                requires_grad=True,
            )
            logscale_betas = nn.Parameter(
                torch.ones(z_max + 1, n_alpha_sets, 1) * beta_min_max,
                requires_grad=True,
            )

        elif basis == "fixed":
            logscale_alphas = nn.Parameter(
                torch.ones(z_max + 1, n_alpha_sets, 1) * log_alpha_r_max,
                requires_grad=False,
            )
            logscale_betas = nn.Parameter(
                torch.ones(z_max + 1, n_alpha_sets, 1) * beta_min_max,
                requires_grad=False,
            )

        c_params = (
            torch.randn(z_max + 1, n_alpha_sets, basis_size) * c_std + c_mean
        )
        logscale_cs = nn.Parameter(
            torch.sort(c_params, descending=False)[0], requires_grad=True
        )

        return logscale_alphas, logscale_betas, logscale_cs

    @staticmethod
    def _fit_basis_params(atomic_numbers, alphas, betas, cs):
        pass

    @staticmethod
    def _make_logscale_energy_parameters(shells: dict):
        """Constructs a set of trainable log-scale orbital
        energy parameters."""

        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        logscale_energies = nn.Parameter(
            torch.sort(torch.randn(z_max + 1, n_alpha_sets))[0],
            requires_grad=True,
        )

        return logscale_energies

    def __init__(
        self,
        eigensolver: str | Eigensolver = "torch_eigh",
        spin: str = "paired",
        basis: str = "even_tempered",
        basis_size: int = 3,
        potential_basis: str = "even_tempered",
        potential_basis_size: int = 5,
        energy_scale: float = 1.0,
        orbital_shells: dict | None = None,
        hbar: float = 1.0,
        m_e: float = 1.0,
    ) -> None:
        """Constructs a LCAO (Linear Combination of Atomic Orbitals)
        Hamiltonian kernel. This module contains a fixed set of Gaussian
        orbital parameters, which are trainable `torch.nn.Parameter`
        values. Depending on the basis options specified, these
        parameters may be frozen (i.e., have `requires_grad=False`).

        Args:
            eigensolver (str | Eigensolver, optional): The Eigensolver
                used. Defaults to 'torch_eigh'.
            spin (str, optional): The orbital basis spin. Defaults to 'paired'.
            basis (str, optional): The type of basis used. Defaults to
                'even_tempered'.
            basis_size (int, optional): The number of Gaussians used in the
                basis representation. Defaults to 3.
            potential_basis (str, optional): The type of potential basis used.
                Defaults to 'even_tempered'.
            potential_basis_size (int, optional): The number of Gaussians
                used in the potential basis representation. Defaults to 5.
            energy_scale (float, optional): The scale multiple of energy used.
                If the module is trained on different energy units (such as
                Hartree versus eV units), this scalar may need to be adjusted.
                Defaults to 1.0.
            orbital_shells (dict, optional): A dictionary mapping atomic
                numbers to orbital shell angular momenta :math:`l`.
                For example, to model the 1s, 2s, and 2p orbitals of
                Carbon atoms, one would use `orbital_shells={6: [0, 0, 1]}`
                to model the :math:`l=0,l=0` and :math:`l=1` angular momenta of
                these orbitals. For any elements not specified in this
                dictionary, the valence shell orbitals are used.
                Defaults to None.
            hbar (float, optional): Planck's constant, used as a prefactor in
                returned Hamiltonian kernel entries.
            m_e (float, optaional): Electron mass, used as a prefactor in
                returned Hamiltonian kernel entries.

        Raises:
            ValueError: If invalid basis options are supplied.
        """

        super().__init__(eigensolver=eigensolver)

        self.basis = basis
        self.potential_basis = potential_basis
        self.basis_size = basis_size
        self.potential_basis_size = potential_basis_size

        if spin not in LCAOHamiltonian.SPIN_OPTIONS:
            raise ValueError(
                f"Spin must be one of {LCAOHamiltonian.SPIN_OPTIONS}, got: {spin}"
            )

        self.spin = spin

        # use default valence shells to start and override with supplied values
        orbital_shells = {} if orbital_shells is None else orbital_shells
        orbital_shells = _ELEMENT_CHO_VALENCE_SHELLS | orbital_shells

        # initialize shell index parameters
        self.register_buffer(
            "orbital_shell_idxs",
            LCAOHamiltonian._make_shell_idxs(orbital_shells),
        )

        self.z_max = max(orbital_shells.keys())

        self.energy_scale = energy_scale

        # initialize log-scale orbital gaussian parameters
        (
            self.logscale_orbital_alphas,
            self.logscale_orbital_betas,
            self.logscale_orbital_cs,
        ) = LCAOHamiltonian._make_logscale_basis_parameters(
            basis=basis,
            basis_size=basis_size,
            shells=orbital_shells,
            c_std=1 / np.sqrt(basis_size),
        )

        # initialize log-scale potential gaussian parameters
        # (use only s-type gaussian potential "shells" for now)
        potential_shells = {n: [0] for n in orbital_shells}
        (
            self.logscale_potential_gammas,
            self.logscale_potential_betas,
            self.logscale_potential_vs,
        ) = LCAOHamiltonian._make_logscale_basis_parameters(
            basis=potential_basis,
            basis_size=potential_basis_size,
            shells=potential_shells,
            c_std=0.15,
            c_mean=-np.log(potential_basis_size),
        )

        # initialize on-site logscale energies
        self.logscale_onsite_energies = (
            LCAOHamiltonian._make_logscale_energy_parameters(
                shells=orbital_shells
            )
        )

        self.hbar = hbar
        self.m_e = m_e

    def _get_c_normalization_coeffs(
        self,
        atomic_numbers: torch.IntTensor,
        alphas: torch.Tensor,
        cs: torch.Tensor,
    ):

        shell_idxs = self.orbital_shell_idxs[atomic_numbers]
        site_idxs, site_l_idxs = (shell_idxs >= 0).nonzero(as_tuple=True)

        site_ls = shell_idxs[site_idxs, site_l_idxs]
        site_alphas = alphas[site_idxs, site_l_idxs]
        site_cs = cs[site_idxs, site_l_idxs]

        site_c_norms = cho_ng_norm_coeff(
            l=site_ls.unsqueeze(-1),
            alphas=site_alphas.unsqueeze(-2),
            alpha_idxs=torch.zeros_like(site_ls),
            cs=site_cs.unsqueeze(-1),
        )

        c_norm_coeffs = torch.zeros_like(cs)
        c_norm_coeffs[site_idxs, site_l_idxs] = 1 / site_c_norms

        return c_norm_coeffs

    def _get_logscale_basis_width_parameters(
        self, atomic_numbers: torch.IntTensor
    ) -> tuple:

        # get log-scale orbital parameters
        base_logscale_alphas = self.logscale_orbital_alphas[atomic_numbers]
        base_logscale_betas = self.logscale_orbital_betas[atomic_numbers]

        beta_powers = torch.arange(
            0,
            self.logscale_orbital_cs.shape[-1],
            device=base_logscale_betas.device,
        ).view(1, 1, -1)

        base_logscale_v_gammas = self.logscale_potential_gammas[atomic_numbers]
        base_logscale_v_betas = self.logscale_potential_betas[atomic_numbers]

        v_beta_powers = torch.arange(
            0,
            self.logscale_potential_vs.shape[-1],
            device=base_logscale_betas.device,
        ).view(1, 1, -1)

        logscale_alphas = (
            base_logscale_alphas + base_logscale_betas * beta_powers
        )

        logscale_gammas = (
            base_logscale_v_gammas + base_logscale_v_betas * v_beta_powers
        )

        return logscale_alphas, logscale_gammas

    def _load_weights(self, name, other_model, freeze=False):
        other_weights = getattr(other_model, name)
        self_weights = getattr(self, name)
        grad = False if freeze else other_weights.requires_grad

        if self_weights.shape != other_weights.shape:
            raise ValueError(
                f'Expected weights "{name}" '
                f"to have shape {self_weights.shape}; "
                f"got {other_weights.shape}."
            )

        setattr(
            self,
            name,
            nn.Parameter(other_weights.detach().clone(), requires_grad=grad),
        )

    def _load_base_model_weights(
        self, base_model: "LCAOHamiltonian", freeze=False
    ):

        self._load_weights("logscale_orbital_alphas", base_model, freeze)
        self._load_weights("logscale_orbital_betas", base_model, freeze)
        self._load_weights("logscale_orbital_cs", base_model, freeze)

        self._load_weights("logscale_potential_gammas", base_model, freeze)
        self._load_weights("logscale_potential_betas", base_model, freeze)
        self._load_weights("logscale_potential_vs", base_model, freeze)

        self._load_weights("logscale_onsite_energies", base_model, freeze)

    def get_orbital_shells_dict(self) -> dict:
        """Returns a dictionary of the orbital shells supported in this model.

        Returns:
            dict: Orbital shell dictionary.
        """

        shells = {
            i: list(l_vals[l_vals >= 0].detach().cpu().numpy())
            for i, l_vals in enumerate(self.orbital_shell_idxs)
            if i > 0
        }

        return shells

    def get_basis_parameters(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
    ) -> tuple:
        """Returns the Gaussian basis parameters of an atomic system.


        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).


        Returns:
            tuple: A tuple containing the following:

            - alphas (torch.Tensor): A tensor of Gaussian orbital
                width parameters (shape: [N_atoms, N_ls, N_basis]).
            - cs (torch.Tensor): A tensor of Gaussian orbital
                coefficients (shape: [N_atoms, N_cs, N_basis]).
            - gammas (torch.Tensor): A tensor of Gaussian
                potential width parameters (shape: [N_atoms, 1, M_basis]).
            - vs (torch.Tensor): A tensor of Gaussian potential
                coefficients (shape: [N_atoms, 1, N_basis])
        """

        logscale_alphas, logscale_gammas = (
            self._get_logscale_basis_width_parameters(
                atomic_numbers=atomic_numbers
            )
        )

        alphas = torch.exp(logscale_alphas)

        # get normalized c coefficients:
        logscale_cs = self.logscale_orbital_cs[atomic_numbers]
        cs_unnormed = torch.sinh(logscale_cs)
        cs = cs_unnormed * self._get_c_normalization_coeffs(
            atomic_numbers=atomic_numbers, alphas=alphas, cs=cs_unnormed
        )

        gammas = torch.exp(logscale_gammas)

        logscale_vs = self.logscale_potential_vs[atomic_numbers]
        vs = -self.energy_scale * torch.exp(logscale_vs)

        return alphas, cs, gammas, vs

    def get_onsite_energies(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
    ) -> torch.Tensor:
        """Computes the on-site energies of an atomic system.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).

        Returns:
            torch.Tensor: The on-site energies of each atom's
                shell orbitals (shape: [N_atoms, N_ls]).
        """

        # get log-scale on-site energies
        logscale_energies = self.logscale_onsite_energies[atomic_numbers]

        return self.energy_scale * torch.sinh(logscale_energies)

    def _get_block_basis_idxs(
        self, idxs: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> tuple:

        block_atoms = atomic_numbers[idxs]
        block_idxs, block_l_idxs = (
            self.orbital_shell_idxs[block_atoms] >= 0
        ).nonzero(as_tuple=True)

        block_idxs = idxs[block_idxs]

        block_ls = self.orbital_shell_idxs[
            atomic_numbers[block_idxs], block_l_idxs
        ]

        return block_idxs, block_l_idxs, block_ls

    def _get_hamiltonian_idxs(self, atomic_numbers: torch.IntTensor) -> tuple:

        H_blocksizes = torch.tensor(
            [
                sum(2 * l + 1 for l in self.orbital_shell_idxs[z] if l >= 0)
                for z in atomic_numbers
            ],
            dtype=atomic_numbers.dtype,
            device=atomic_numbers.device,
            requires_grad=False,
        )

        H_idxs = torch.cumsum(H_blocksizes, dim=0) - H_blocksizes

        return H_idxs, H_blocksizes

    def _get_neighborhood_hamiltonian_idxs(
        self,
        idxs: torch.IntTensor,
        H_idxs: torch.IntTensor,
        H_blocksizes: torch.IntTensor,
    ) -> torch.IntTensor:

        H_neighborhood_idxs = torch.cat(
            [
                H_idxs[i] + torch.arange(H_blocksizes[i], device=H_idxs.device)
                for i in idxs
            ]
        )

        return H_neighborhood_idxs

    def _build_hamiltonian(
        self,
        diagonals: list[torch.Tensor],
        components: list[torch.Tensor],
        component_idxs: list[torch.IntTensor],
        shape: torch.Tensor,
        sparse: bool = False,
    ):

        H_ij = torch.cat(component_idxs, dim=0).T
        H_values = torch.cat(components, dim=0)

        H_diagonal = torch.cat(diagonals, dim=0)
        H_ii = torch.arange(H_diagonal.shape[0]).view(1, -1).expand(2, -1)

        if sparse:
            # append values and hermitian conjugate values
            H_ji = H_ij[torch.tensor([1, 0])]
            H_indices_total = torch.cat((H_ii, H_ij, H_ji), dim=-1)
            H_values_total = torch.cat(
                (H_diagonal, H_values, H_values.conj()), dim=-1
            )

            # build sparse tensor
            H = torch.sparse_coo_tensor(
                indices=H_indices_total,
                values=H_values_total,
                size=shape,
                device=H_values_total.device,
            ).coalesce()

        else:
            # build dense tensor:
            H = torch.zeros(
                shape[0] * shape[1],
                dtype=H_values.dtype,
                device=H_values.device,
            )

            H_ii_idxs = shape[0] * H_ii[0] + H_ii[1]
            H.put_(H_ii_idxs, H_diagonal, accumulate=True)

            H_ij_idxs = shape[0] * H_ij[0] + H_ij[1]
            H.put_(H_ij_idxs, H_values, accumulate=True)

            H_ji_idxs = shape[0] * H_ij[1] + H_ij[0]
            H.put_(H_ji_idxs, H_values, accumulate=True)

            H = H.reshape(shape)

        return H

    def _get_site_orbital_data(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
        site_idxs: torch.IntTensor,
        orbital_idxs: torch.IntTensor,
        **kwargs,
    ):

        data = dict(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            **kwargs,
        )

        a, c, _, _ = self.get_basis_parameters(**data)

        block_indices, _ = self.orbital_indices(**data)
        quantum_numbers = self.orbital_quantum_numbers(**data)

        shells = self.orbital_shell_idxs[atomic_numbers]
        shell_site_idxs, shell_block_idxs = torch.nonzero(
            shells >= 0, as_tuple=True
        )
        shell_ls = shells[shell_site_idxs, shell_block_idxs]

        shell_block_idxs = torch.repeat_interleave(
            shell_block_idxs, repeats=2 * shell_ls + 1
        )

        block_start_idxs = torch.searchsorted(block_indices, input=site_idxs)

        orbital_lms = quantum_numbers[block_start_idxs + orbital_idxs]

        orbital_mus = positions[site_idxs]

        orbital_shell_block_idxs = shell_block_idxs[
            block_start_idxs + orbital_idxs
        ]

        orbital_alphas = a[site_idxs, orbital_shell_block_idxs]
        orbital_cs = c[site_idxs, orbital_shell_block_idxs]

        return orbital_mus, orbital_lms, orbital_alphas, orbital_cs

    def _get_site_potential_data(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
        site_idxs: torch.IntTensor,
        **kwargs,
    ):

        data = dict(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            **kwargs,
        )

        _, _, g, v = self.get_basis_parameters(**data)

        potential_mus = positions[site_idxs]
        potential_gammas = g[site_idxs]
        potential_vs = v[site_idxs]

        return potential_mus, potential_gammas, potential_vs

    def atomic_orbital_amplitudes(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
        site_idxs: torch.IntTensor,
        orbital_idxs: torch.IntTensor,
        x_pts: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Evaluates the atomic orbital amplitudes at points in space.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            site_idxs (torch.IntTensor): Sites at which orbital
                amplitudes should be evaluated. (shape: [N_idxs])
            orbital_idxs (torch.IntTensor): Indices of orbitals which
                should be evaluated at the corresponding site.
                (shape: [N_idxs]).
            x_pts (torch.Tensor): Cartesian coordinates in space
                where the amplitudes should be evaluated. (shape: [N_pts, 3])


        Returns:
            torch.Tensor: The evaluated orbital amplitudes (shape: [N_pts, N_idxs])
        """

        orbital_mus, orbital_lms, orbital_alphas, orbital_cs = (
            self._get_site_orbital_data(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                site_idxs=site_idxs,
                orbital_idxs=orbital_idxs,
            )
        )

        n_idxs = site_idxs.shape[0]
        orbital_amplitudes = torch.zeros(
            (n_idxs, x_pts.shape[0]),
            device=positions.device,
            dtype=positions.dtype,
        )

        for i in range(n_idxs):
            orbital_amplitudes[i] = cho_ng(
                x=x_pts,
                mu=orbital_mus[i],
                l=orbital_lms[i, 0],
                m=orbital_lms[i, 1],
                alphas=orbital_alphas[i],
                cs=orbital_cs[i],
            )

        return orbital_amplitudes.transpose(0, 1)

    def atomic_potentials(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
        site_idxs: torch.IntTensor,
        x_pts: torch.Tensor,
        **kwargs,
    ):
        """Evaluates the effective atomic potential at points in space.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            site_idxs (torch.IntTensor): Sites at which the potential
                should be evaluated. (shape: [N_idxs])
            x_pts (torch.Tensor): Cartesian coordinates in space
                where the amplitudes should be evaluated. (shape: [N_pts, 3])

        Returns:
            torch.Tensor: The evaluated effective potentials
                (shape: [N_pts, N_idxs])
        """

        potential_mus, potential_gammas, potential_vs = (
            self._get_site_potential_data(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                site_idxs=site_idxs,
                **kwargs,
            )
        )

        n_idxs = site_idxs.shape[0]
        ijk = torch.zeros((3), device=x_pts.device, dtype=site_idxs.dtype)

        potential = torch.zeros(
            (n_idxs, x_pts.shape[0]),
            device=positions.device,
            dtype=positions.dtype,
        )

        for i in range(n_idxs):
            potential[i] = potential_ng(
                x=x_pts,
                mu=potential_mus[i],
                ijk=ijk,
                gammas=potential_gammas[i],
                vs=potential_vs[i],
            )

        return potential

    def hamiltonian_matrix(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        kernel: str = "hamiltonian",
        **kwargs,
    ) -> torch.Tensor:
        """Computes the Hamiltonian matrix.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            sparse (bool, optional): If True, returns a sparse matrix
                representation of the Hamiltonian. Defaults to False.

        Returns:
            torch.Tensor: The Hamiltonian matrix (shape: [H_size, H_size]).
        """

        onsite_values, couplings, coupling_values = (
            self.hamiltonian_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                kernel=kernel,
                **kwargs,
            )
        )

        return build_nonperiodic_matrix(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
        )

    def hamiltonian_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        kernel: str = "hamiltonian",
        **kwargs,
    ) -> tuple:

        _kernel = kernel.lower()
        if _kernel not in LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS:
            choices = list(LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS.keys())
            raise ValueError(
                f'Unknown hamiltonian operator kernel name "{kernel}" (expected one of {choices}).'
            )

        cho_ng_kernel = LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS[_kernel]

        # get orbital parameters:
        alphas, cs, gammas, vs = self.get_basis_parameters(
            atomic_numbers, positions, neighbors
        )

        # get on-site energies:
        onsite_energies = self.get_onsite_energies(
            atomic_numbers, positions, neighbors
        )

        H_idxs, H_blocksizes = self._get_hamiltonian_idxs(atomic_numbers)

        H_diagonals = []
        H_components = []
        H_component_idxs = []

        n_sites = atomic_numbers.shape[0]

        for i in range(n_sites):

            # retrieve site data:
            site = torch.tensor(
                [i],
                dtype=neighbors.dtype,
                device=neighbors.device,
                requires_grad=False,
            )
            site_idxs, site_l_idxs, site_ls = self._get_block_basis_idxs(
                site, atomic_numbers
            )

            site_data = cho_overlap_matrix_metadata(site_ls, sparse=True)

            H_site_idxs = self._get_neighborhood_hamiltonian_idxs(
                site, H_idxs, H_blocksizes
            )

            # retrieve neighborhood data:
            site_neighbors_pad = neighbors[i]
            site_neighbors = site_neighbors_pad[site_neighbors_pad >= 0]

            neighbor_idxs, neighbor_l_idxs, neighbor_ls = (
                self._get_block_basis_idxs(site_neighbors, atomic_numbers)
            )

            neighbor_data = cho_overlap_matrix_metadata(
                neighbor_ls, sparse=True
            )

            H_neighbor_idxs = self._get_neighborhood_hamiltonian_idxs(
                site_neighbors, H_idxs, H_blocksizes
            )

            # compute on-site diagonals:
            H_site_diagonal = torch.repeat_interleave(
                input=onsite_energies[site_idxs, site_l_idxs],
                repeats=2 * site_ls + 1,
            )

            # compute neighborhood Hamiltonian matrix:
            H_neighborhood = cho_ng_kernel(
                mu_1=positions[site_idxs],
                l_1=site_ls,
                l_alphas_1=alphas[site_idxs, site_l_idxs],
                l_cs_1=cs[site_idxs, site_l_idxs],
                gammas_1=gammas[i],
                vs_1=vs[i],
                mu_2=positions[neighbor_idxs],
                l_2=neighbor_ls,
                l_alphas_2=alphas[neighbor_idxs, neighbor_l_idxs],
                l_cs_2=cs[neighbor_idxs, neighbor_l_idxs],
                cho_data_1=site_data,
                cho_data_2=neighbor_data,
            ).flatten(start_dim=-2)

            H_neighborhood_idxs = torch.cartesian_prod(
                H_site_idxs, H_neighbor_idxs
            )

            H_offdiag = torch.cat(
                (H_neighborhood, H_neighborhood.conj()), dim=-1
            )
            H_offdiag_idxs = torch.cat(
                (H_neighborhood_idxs, H_neighborhood_idxs.flip(-1))
            )

            H_diagonals.append(H_site_diagonal)
            H_components.append(H_offdiag)
            H_component_idxs.append(H_offdiag_idxs)

        onsite_values = torch.cat(H_diagonals, dim=-1)
        couplings = torch.cat(H_component_idxs)
        coupling_values = torch.cat(H_components, dim=-1)

        return onsite_values, couplings, coupling_values

    def orbital_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        kernel: str = "overlap",
        **kwargs,
    ) -> tuple:

        _kernel = kernel.lower()
        if _kernel not in LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS:
            choices = list(LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS.keys())
            raise ValueError(
                f'Unknown orbital operator kernel name "{_kernel}" (expected one of {choices}).'
            )

        cho_ng_kernel = LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS[_kernel]

        # get orbital parameters:
        alphas, cs, gammas, vs = self.get_basis_parameters(
            atomic_numbers, positions, neighbors
        )

        # get on-site energies:
        # onsite_energies = self.get_onsite_energies(
        #    atomic_numbers, positions, neighbors)

        H_idxs, H_blocksizes = self._get_hamiltonian_idxs(atomic_numbers)

        H_diagonals = []
        H_components = []
        H_component_idxs = []

        n_sites = atomic_numbers.shape[0]

        for i in range(n_sites):

            # retrieve site data:
            site = torch.tensor(
                [i],
                dtype=neighbors.dtype,
                device=neighbors.device,
                requires_grad=False,
            )
            site_idxs, site_l_idxs, site_ls = self._get_block_basis_idxs(
                site, atomic_numbers
            )

            site_data = cho_overlap_matrix_metadata(site_ls, sparse=True)

            H_site_idxs = self._get_neighborhood_hamiltonian_idxs(
                site, H_idxs, H_blocksizes
            )

            H_onsite = cho_ng_kernel(
                mu_1=positions[site_idxs],
                l_1=site_ls,
                l_alphas_1=alphas[site_idxs, site_l_idxs],
                l_cs_1=cs[site_idxs, site_l_idxs],
                mu_2=positions[site_idxs],
                l_2=site_ls,
                l_alphas_2=alphas[site_idxs, site_l_idxs],
                l_cs_2=cs[site_idxs, site_l_idxs],
                cho_data_1=site_data,
                cho_data_2=site_data,
            )

            H_onsite_idxs = torch.cartesian_prod(H_site_idxs, H_site_idxs)
            _onsite_off_diag = H_onsite_idxs[:, 0] != H_onsite_idxs[:, 1]

            H_onsite_diagonal = torch.diagonal(H_onsite, dim1=-2, dim2=-1)
            H_onsite_offdiag_idxs = H_onsite_idxs[_onsite_off_diag]
            H_onsite_offdiag = H_onsite.flatten(start_dim=-2)[
                ..., _onsite_off_diag
            ]

            # retrieve neighborhood data:
            site_neighbors_pad = neighbors[i]
            site_neighbors = site_neighbors_pad[site_neighbors_pad >= 0]

            neighbor_idxs, neighbor_l_idxs, neighbor_ls = (
                self._get_block_basis_idxs(site_neighbors, atomic_numbers)
            )

            neighbor_data = cho_overlap_matrix_metadata(
                neighbor_ls, sparse=True
            )

            H_neighbor_idxs = self._get_neighborhood_hamiltonian_idxs(
                site_neighbors, H_idxs, H_blocksizes
            )

            # compute neighborhood kernel matrix:
            H_neighborhood = cho_ng_kernel(
                mu_1=positions[site_idxs],
                l_1=site_ls,
                l_alphas_1=alphas[site_idxs, site_l_idxs],
                l_cs_1=cs[site_idxs, site_l_idxs],
                mu_2=positions[neighbor_idxs],
                l_2=neighbor_ls,
                l_alphas_2=alphas[neighbor_idxs, neighbor_l_idxs],
                l_cs_2=cs[neighbor_idxs, neighbor_l_idxs],
                cho_data_1=site_data,
                cho_data_2=neighbor_data,
            ).flatten(start_dim=-2)

            H_neighborhood_idxs = torch.cartesian_prod(
                H_site_idxs, H_neighbor_idxs
            )

            H_offdiag = torch.cat(
                (H_onsite_offdiag, H_neighborhood, H_neighborhood.conj()),
                dim=-1,
            )
            H_offdiag_idxs = torch.cat(
                (
                    H_onsite_offdiag_idxs,
                    H_neighborhood_idxs,
                    H_neighborhood_idxs.flip(-1),
                )
            )

            H_diagonals.append(H_onsite_diagonal)
            H_components.append(H_offdiag)
            H_component_idxs.append(H_offdiag_idxs)

        onsite_values = torch.cat(H_diagonals, dim=-1)
        couplings = torch.cat(H_component_idxs)
        coupling_values = torch.cat(H_components, dim=-1)

        return onsite_values, couplings, coupling_values

    def orbital_names(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> list[str]:
        """Returns the names of all orbitals in the Hamiltonian matrix.
        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).

        Returns:
            list[str]: A list of orbital names with the same size as the
                Hamiltonian matrix.
        """

        key = []
        for z in atomic_numbers:

            l_values = self.orbital_shell_idxs[z]
            l_nz_values = l_values[l_values >= 0]

            for l in l_nz_values:
                for i in range(2 * l + 1):
                    key.append(_get_cho_orbital_name(l, i))

        return key

    def orbital_indices(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> tuple:
        """Returns the indices of all sites and corresponding orbitals
        in the Hamiltonian matrix.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).

        Returns:
            tuple: A tuple containing the following:

            - site_idxs (torch.IntTensor):
                A list of site indices (shape: [H_size])
            - orbital_idxs (torch.IntTensor):
                A list of corresponding orbital indices (shape: [H_size])
        """

        # compute index tensor of l values
        l_values_padded = self.orbital_shell_idxs[atomic_numbers]

        l_site_idxs, l_orbital_idxs = torch.nonzero(
            l_values_padded >= 0, as_tuple=True
        )
        l_values = l_values_padded[l_site_idxs, l_orbital_idxs]

        site_idxs = torch.repeat_interleave(
            l_site_idxs, repeats=2 * l_values + 1
        )

        # compute orbital indices
        n_orbitals = torch.zeros_like(atomic_numbers)
        n_orbitals.put_(l_site_idxs, 2 * l_values + 1, accumulate=True)

        site_idxs, orbital_idxs = torch.nonzero(
            torch.arange(
                torch.max(n_orbitals), device=atomic_numbers.device
            ).view(1, -1)
            < n_orbitals.view(-1, 1),
            as_tuple=True,
        )

        return site_idxs, orbital_idxs

    def orbital_quantum_numbers(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Returns the quantum numbers of each orbital in the Hamiltonian
        matrix.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).

        Returns:
            torch.Tensor: A list of quantum numbers (n,l) for each
                orbital in the Hamiltonian matrix (shape: [H_size, 2])
        """

        # compute index tensor of l values
        l_values_padded = self.orbital_shell_idxs[atomic_numbers]

        l_site_idxs, l_orbital_idxs = torch.nonzero(
            l_values_padded >= 0, as_tuple=True
        )
        l_values = l_values_padded[l_site_idxs, l_orbital_idxs]
        l_blocksizes = 2 * l_values + 1

        quantum_ls = l_values.repeat_interleave(l_blocksizes)
        m_offsets = torch.cumsum(l_blocksizes, dim=-1) - l_blocksizes

        quantum_ms = (
            torch.arange(quantum_ls.shape[0], device=quantum_ls.device)
            - torch.repeat_interleave(m_offsets, l_blocksizes)
            - quantum_ls
        )

        orbital_quantum_numbers = torch.vstack(
            (
                quantum_ls,  # l (ang. quantum number)
                quantum_ms,  # m (mag. quantum number)
            )
        ).transpose(0, 1)

        return orbital_quantum_numbers


class TBHamiltonian(LCAOHamiltonian):
    r"""A Tight Binding Hamiltonian Module.

    This module contains a fixed set of Gaussian
    orbital parameters, which are trainable `torch.nn.Parameter`
    values. Like LCAOHamiltonian, this module uses a basis of Gaussian
    cubic harmonic orbitals, but it can support periodic systems.
    To account for periodicity, each Gaussian orbital is expanded
    as bloch orbitals with crystal momentum (k) degrees of freedom.


    The Hamiltonian it constructs is a matrix-valued function of k that
    is complex-valued and Hermitian.

    When computing the Hamiltonian, this module computes the overlap
    matrices of each neighboring orbital state with an on-site
    pseudopotential that is expanded as a set of Gaussian functions:

        .. math::
            H_{ij}(k) = e^{i(k \cdot (x_i - x_j)}\langle
            \phi_i | V_i + V_j | \phi_j \rangle

    This module currently supports the following eigensolvers:
        - `torch_eigh`

    """

    def __init__(
        self,
        eigensolver: str | Eigensolver = "torch_eigh",
        spin="paired",
        basis: str = "even_tempered",
        basis_size: int = 3,
        potential_basis: str = "even_tempered",
        potential_basis_size: int = 5,
        energy_scale: float = 1.0,
        orbital_shells: dict | None = None,
    ) -> None:

        super().__init__(
            eigensolver=eigensolver,
            spin=spin,
            basis=basis,
            basis_size=basis_size,
            potential_basis=potential_basis,
            potential_basis_size=potential_basis_size,
            energy_scale=energy_scale,
            orbital_shells=orbital_shells,
        )

    def _get_neighborhood_hamiltonian_distances(
        self, neighbor_distances: torch.Tensor, neighbor_ls: torch.IntTensor
    ) -> torch.IntTensor:

        H_neighborhood_dxs = torch.cat(
            [
                dx.unsqueeze(0).expand(2 * l + 1, -1)
                for dx, l in zip(neighbor_distances, neighbor_ls)
            ]
        )

        return H_neighborhood_dxs

    def _get_neighborhood_hamiltonian_images(
        self, neighbor_images: torch.Tensor, neighbor_ls: torch.IntTensor
    ) -> torch.IntTensor:

        H_neighborhood_images = torch.repeat_interleave(
            neighbor_images, 2 * neighbor_ls + 1, dim=0
        )

        return H_neighborhood_images

    def _get_block_image_basis_idxs(
        self,
        idxs: torch.IntTensor,
        image_idxs: torch.IntTensor,
        atomic_numbers: torch.IntTensor,
    ) -> tuple:

        block_atoms = atomic_numbers[idxs]
        _block_idxs, block_l_idxs = (
            self.orbital_shell_idxs[block_atoms] >= 0
        ).nonzero(as_tuple=True)

        block_idxs = idxs[_block_idxs]
        block_image_idxs = image_idxs[_block_idxs]

        block_ls = self.orbital_shell_idxs[
            atomic_numbers[block_idxs], block_l_idxs
        ]

        return block_idxs, block_image_idxs, block_l_idxs, block_ls

    def _build_bloch_operator(
        self,
        diagonals: list[torch.Tensor],
        components: list[torch.Tensor],
        component_idxs: list[torch.IntTensor],
        component_images: list[torch.IntTensor],
    ):

        H_ij = torch.cat(component_idxs, dim=0)
        H_ji = torch.flip(H_ij, dims=(1,))
        H_values = torch.cat(components, dim=0)
        H_images = torch.cat(component_images, dim=0)

        onsite_values = torch.cat(diagonals, dim=0)
        couplings = torch.cat((H_ij, H_ji), dim=0)
        coupling_values = torch.cat((H_values, H_values.conj()), dim=0)
        coupling_images = torch.cat((H_images, -H_images), dim=0)

        return onsite_values, couplings, coupling_values, coupling_images

    def _build_bloch_hamiltonian(
        self,
        diagonals: list[torch.Tensor],
        components: list[torch.Tensor],
        component_idxs: list[torch.IntTensor],
        component_distances: list[torch.Tensor],
        k_pts: torch.Tensor,
        shape: torch.Tensor,
        sparse: bool = False,
    ):

        H_ij = torch.cat(component_idxs, dim=0).T
        H_ji = torch.flip(H_ij, dims=(0,))
        H_values = torch.cat(components, dim=0)

        H_diagonal = torch.cat(diagonals, dim=0)
        H_ii = (
            torch.arange(H_diagonal.shape[0], device=H_diagonal.device)
            .view(1, -1)
            .expand(2, -1)
        )

        H_distances = torch.cat(component_distances, dim=0)
        k_shape = k_pts.shape[:-1]
        k_view = tuple([1] * len(k_shape))

        if sparse:

            # move batch dimensions to the trailing dimensions
            # (This is necessary due to the fact that pytorch
            #  only supports efficient sparse formats with the
            #  leading axes as sparse)
            H_thetas_k = torch.einsum(
                "id, ...d -> i...", H_distances, 2 * torch.pi * k_pts
            )

            H_values_k = torch.einsum(
                "i, i... -> i...", H_values, torch.exp(1.0j * H_thetas_k)
            )

            H_diagonal_k = H_diagonal.view(-1, *k_view).expand(-1, *k_shape)

            # append values and hermitian conjugate values
            H_ji = H_ij[torch.tensor([1, 0])]
            H_indices_total = torch.cat((H_ii, H_ij, H_ji), dim=-1)
            H_values_total = torch.cat(
                (H_diagonal_k, H_values_k, H_values_k.conj()), dim=0
            )

            # build sparse tensor
            H = torch.sparse_coo_tensor(
                indices=H_indices_total,
                values=H_values_total,
                size=shape + k_shape,
                device=H_values_total.device,
            ).coalesce()
        else:

            # build dense hamiltonian
            H = torch.zeros(
                k_shape + (shape[0] * shape[1],),
                dtype=torch.complex64,
                device=H_diagonal.device,
            )

            H_ii_idxs = (
                (shape[0] * H_ii[0] + H_ii[1])
                .view(*k_view, -1)
                .expand(*k_shape, -1)
            )

            H_diagonal_k = (
                H_diagonal.to(torch.complex64)
                .view(*k_view, -1)
                .expand(*k_shape, -1)
            )

            H.scatter_add_(-1, H_ii_idxs, H_diagonal_k)

            H_thetas_k = torch.einsum(
                "id, ...d -> ...i", H_distances, 2 * torch.pi * k_pts
            )

            H_values_k = torch.einsum(
                "i, ...i -> ...i", H_values, torch.exp(1.0j * H_thetas_k)
            )

            H_ij_idxs = (
                (shape[0] * H_ij[0] + H_ij[1])
                .view(*k_view, -1)
                .expand(*k_shape, -1)
            )

            H.scatter_add_(-1, H_ij_idxs, H_values_k)

            H_ji_idxs = (
                (shape[0] * H_ij[1] + H_ij[0])
                .view(*k_view, -1)
                .expand(*k_shape, -1)
            )

            H.scatter_add_(-1, H_ji_idxs, H_values_k.conj())

            # restore original shape
            H = H.reshape(k_shape + shape)

        return H

    def get_onsite_energies(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
    ) -> torch.Tensor:
        """Computes the on-site energies of a periodic atomic system.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            unit_cell (torch.Tensor): The unit cell matrix of the system,
                where each row corresponds to a unit cell vector in
                cartesian coordinates (shape: [3,3]).
            unit_cell_images (torch.IntTensor): Unit cell image
                where each neighbor appears. Images where the
                corresponding element of `neighbors` is negative
                are ignored (shape: [N_atoms, N_neighbors, 3]).

        Returns:
            torch.Tensor: The on-site energies of each atom's
                shell orbitals (shape: [N_atoms, N_ls]).
        """

        return super().get_onsite_energies(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

    def get_basis_parameters(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
    ) -> tuple:
        """Returns the Gaussian basis parameters of an atomic system.


        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            unit_cell (torch.Tensor): The unit cell matrix of the system,
                where each row corresponds to a unit cell vector in
                cartesian coordinates (shape: [3,3]).
            unit_cell_images (torch.IntTensor): Unit cell image
                where each neighbor appears. Images where the
                corresponding element of `neighbors` is negative
                are ignored (shape: [N_atoms, N_neighbors, 3]).


        Returns:
            tuple: A tuple containing the following:

            - alphas (torch.Tensor): A tensor of Gaussian orbital
                width parameters (shape: [N_atoms, N_ls, N_basis]).
            - cs (torch.Tensor): A tensor of Gaussian orbital
                coefficients (shape: [N_atoms, N_cs, N_basis]).
            - gammas (torch.Tensor): A tensor of Gaussian
                potential width parameters (shape: [N_atoms, 1, M_basis]).
            - vs (torch.Tensor): A tensor of Gaussian potential
                coefficients (shape: [N_atoms, 1, N_basis])
        """

        return super().get_basis_parameters(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

    def atomic_orbital_amplitudes(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        site_idxs: torch.IntTensor,
        orbital_idxs: torch.IntTensor,
        x_pts: torch.Tensor,
    ):
        """Evaluates the atomic orbital amplitudes at points within the
        unit cell.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            unit_cell (torch.Tensor): The unit cell matrix of the system,
                where each row corresponds to a unit cell vector in
                cartesian coordinates (shape: [3,3]).
            unit_cell_images (torch.IntTensor): Unit cell image
                where each neighbor appears. Images where the
                corresponding element of `neighbors` is negative
                are ignored (shape: [N_atoms, N_neighbors, 3]).
            site_idxs (torch.IntTensor): Sites at which orbital
                amplitudes should be evaluated. (shape: [N_idxs])
            orbital_idxs (torch.IntTensor): Indices of orbitals which
                should be evaluated at the corresponding site.
                (shape: [N_idxs]).
            x_pts (torch.Tensor): Cartesian coordinates in space
                where the amplitudes should be evaluated. These points
                should lie within the unit cell (shape: [N_pts, 3]).


        Returns:
            torch.Tensor: The evaluated orbital amplitudes (shape: [N_pts, N_idxs])
        """

        orbital_mus, orbital_lms, orbital_alphas, orbital_cs = (
            self._get_site_orbital_data(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                site_idxs=site_idxs,
                orbital_idxs=orbital_idxs,
            )
        )

        n_idxs = site_idxs.shape[0]
        orbital_amplitudes = torch.zeros((n_idxs, x_pts.shape[0]))
        orbital_images = unit_cell_images[site_idxs]

        for i in range(n_idxs):

            orbital_amplitudes[i] = cho_ng(
                x=x_pts,
                mu=orbital_mus[i],
                l=orbital_lms[i, 0],
                m=orbital_lms[i, 1],
                alphas=orbital_alphas[i],
                cs=orbital_cs[i],
            )

            # include any self-interactions (if exists)
            site_neighbors = neighbors[site_idxs[i]]
            self_neighbor_idxs = torch.nonzero(site_neighbors == site_idxs[i])

            for j in self_neighbor_idxs:
                orbital_amplitudes[i] += cho_ng(
                    x=x_pts,
                    mu=orbital_mus[i]
                    + (orbital_images[i, j].to(torch.float32) @ unit_cell),
                    l=orbital_lms[i, 0],
                    m=orbital_lms[i, 1],
                    alphas=orbital_alphas[i],
                    cs=orbital_cs[i],
                )

        return orbital_amplitudes.transpose(0, 1)

    def bloch_eigenstate_amplitudes(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        x_pts: torch.Tensor,
        k_pts: torch.Tensor | None = None,
        eigenstate_idxs: torch.IntTensor | None = None,
        x_cutoff: float | None = None,
    ):
        """Evaluates the Bloch eigenstate amplitudes at points in space for
        supplied k points.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            unit_cell (torch.Tensor): The unit cell matrix of the system,
                where each row corresponds to a unit cell vector in
                cartesian coordinates (shape: [3,3]).
            unit_cell_images (torch.IntTensor): Unit cell image
                where each neighbor appears. Images where the
                corresponding element of `neighbors` is negative
                are ignored (shape: [N_atoms, N_neighbors, 3]).
            x_pts (torch.Tensor): Cartesian coordinates in space
                where the amplitudes should be evaluated. (shape: [N_pts, 3])
            k_pts (torch.Tensor | None, optional):
                Reciprocal space points (shape: [N_k, 3]). If None, only k=0
                is used. Defaults to None.
            eigenstate_idxs (torch.IntTensor | None, optional):
                Indices of Bloch eigenstates to evaluate.
                (shape: [N_idxs]). Defaults to None.
            x_cutoff: (float | None, optional): Cutoff distance for
                ignoring orbital contributions that are far away
                from the evaluation points (may speed up evaluation).
                If None, a recommended default cutoff is used.
        Returns:
            torch.Tensor: The evaluated Bloch eigenstate amplitudes
            (shape: [N_pts, N_k, N_idxs])
        """

        _, eigenstates = self.eigenstates(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
        )

        if x_cutoff is None:
            x_cutoff = 5.0

        site_idxs, orbital_idxs = self.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        x_center = torch.mean(x_pts, axis=0, keepdims=True)
        x_radius = torch.max(torch.sqrt(torch.square(x_pts - x_center).sum(-1)))

        if k_pts is None:
            k_pts = torch.zeros(3, device=positions.device)

        if eigenstate_idxs is not None:
            eigenstates = eigenstates[..., eigenstate_idxs]

        nearby_sites = (
            torch.sqrt(torch.square(positions - x_center).sum(-1))
            < x_radius + x_cutoff
        )

        nearby_idx = nearby_sites[site_idxs]
        nearby_site_idxs = site_idxs[nearby_idx]
        nearby_orbital_idxs = orbital_idxs[nearby_idx]
        nearby_eigenstates = eigenstates[..., nearby_idx, :]

        amps = self.atomic_orbital_amplitudes(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            site_idxs=nearby_site_idxs,
            orbital_idxs=nearby_orbital_idxs,
            x_pts=x_pts,
        )

        eigenstate_amps = torch.einsum(
            "xi, ...ij -> ...xj", amps.to(torch.complex64), nearby_eigenstates
        )

        return eigenstate_amps

    def hamiltonian_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        kernel: str = "potential",
    ) -> tuple:

        _kernel = kernel.lower()
        if _kernel not in LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS:
            choices = list(LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS.keys())
            raise ValueError(
                f'Unknown hamiltonian operator kernel name "{kernel}" (expected one of {choices}).'
            )

        cho_ng_kernel = LCAOHamiltonian.HAMILTONIAN_OPERATOR_KERNELS[_kernel]

        # get orbital parameters:
        alphas, cs, gammas, vs = self.get_basis_parameters(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )

        # get on-site energies:
        onsite_energies = self.get_onsite_energies(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )

        # convert unit_cell_images to float:
        unit_cell_images = unit_cell_images.to(torch.float)

        H_idxs, H_blocksizes = self._get_hamiltonian_idxs(atomic_numbers)

        H_diagonals = []
        H_components = []
        H_component_images = []
        H_component_idxs = []
        n_sites = atomic_numbers.shape[0]

        for i in range(n_sites):

            # retrieve site data:
            site = torch.tensor(
                [i],
                dtype=neighbors.dtype,
                device=neighbors.device,
                requires_grad=False,
            )
            site_idxs, site_l_idxs, site_ls = self._get_block_basis_idxs(
                site, atomic_numbers
            )
            site_data = cho_overlap_matrix_metadata(site_ls, sparse=True)

            H_site_idxs = self._get_neighborhood_hamiltonian_idxs(
                site, H_idxs, H_blocksizes
            )

            # compute on-site diagonals:
            H_site_diagonal = torch.repeat_interleave(
                onsite_energies[site_idxs, site_l_idxs], repeats=2 * site_ls + 1
            )

            # retrieve neighborhood data:
            site_neighbors_nz = neighbors[i] >= 0
            site_neighbors = neighbors[i][site_neighbors_nz]
            site_neighbor_images = unit_cell_images[i][site_neighbors_nz]

            # handle if neighborhood is empty:
            if site_neighbors.shape[0] == 0:
                H_components.append(
                    torch.zeros(
                        H_site_diagonal.shape[:-1] + (0,),
                        device=H_site_diagonal.device,
                        dtype=H_site_diagonal.dtype,
                    )
                )
                H_component_idxs.append(
                    torch.zeros(
                        (0, 2),
                        dtype=site_neighbors.dtype,
                        device=site_neighbors.device,
                    )
                )
                H_component_images.append(
                    torch.zeros(
                        (0, 3),
                        dtype=unit_cell_images.dtype,
                        device=unit_cell_images.device,
                    )
                )
            else:
                # otherwise, compute neighborhood overlap matrices:
                neighbor_idxs, neighbor_images, neighbor_l_idxs, neighbor_ls = (
                    self._get_block_image_basis_idxs(
                        site_neighbors, site_neighbor_images, atomic_numbers
                    )
                )

                neighbor_data = cho_overlap_matrix_metadata(
                    neighbor_ls, sparse=True
                )

                H_neighbor_idxs = self._get_neighborhood_hamiltonian_idxs(
                    site_neighbors, H_idxs, H_blocksizes
                )

                # compute distances to neighbors across unit cell boundaries:
                neighbor_positions = (
                    positions[neighbor_idxs] + neighbor_images @ unit_cell
                )

                # compute neighborhood Hamiltonian matrix:
                H_neighborhood = cho_ng_kernel(
                    mu_1=positions[site_idxs],
                    l_1=site_ls,
                    l_alphas_1=alphas[site_idxs, site_l_idxs],
                    l_cs_1=cs[site_idxs, site_l_idxs],
                    gammas_1=gammas[i],
                    vs_1=vs[i],
                    mu_2=neighbor_positions,
                    l_2=neighbor_ls,
                    l_alphas_2=alphas[neighbor_idxs, neighbor_l_idxs],
                    l_cs_2=cs[neighbor_idxs, neighbor_l_idxs],
                    cho_data_1=site_data,
                    cho_data_2=neighbor_data,
                )

                # H_neighborhood_dxs = \
                #     self._get_neighborhood_hamiltonian_distances(
                #         neighbor_positions - positions[i],
                #         neighbor_ls
                #     ).view(1, -1, 3).expand(*H_neighborhood.shape, 3)

                H_neighborhood_images = (
                    self._get_neighborhood_hamiltonian_images(
                        neighbor_images,
                        neighbor_ls,
                    )
                    .view(1, -1, 3)
                    .expand(
                        H_neighborhood.shape[-2], H_neighborhood.shape[-1], 3
                    )
                    .reshape(-1, 3)
                )

                H_neighborhood = H_neighborhood.flatten(start_dim=-2)

                H_neighborhood_idxs = torch.cartesian_prod(
                    H_site_idxs, H_neighbor_idxs
                )

                H_diagonals.append(H_site_diagonal)
                H_components.append(
                    torch.cat((H_neighborhood, H_neighborhood.conj()), dim=-1)
                )
                H_component_idxs.append(
                    torch.cat(
                        (H_neighborhood_idxs, H_neighborhood_idxs.flip(-1))
                    )
                )
                H_component_images.append(
                    torch.cat((H_neighborhood_images, -H_neighborhood_images))
                )

        onsite_values = torch.cat(H_diagonals, dim=-1)
        couplings = torch.cat(H_component_idxs)
        coupling_values = torch.cat(H_components, dim=-1)
        coupling_images = torch.cat(H_component_images)

        return onsite_values, couplings, coupling_values, coupling_images

    def orbital_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        kernel: str = "overlap",
    ) -> tuple:

        _kernel = kernel.lower()
        if _kernel not in LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS:
            choices = list(LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS.keys())
            raise ValueError(
                f'Unknown operator kernel name "{kernel}" (expected one of {choices}).'
            )
        cho_ng_kernel = LCAOHamiltonian.ORBITAL_OPERATOR_KERNELS[_kernel]

        # get orbital parameters:
        alphas, cs, gammas, vs = self.get_basis_parameters(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )

        # convert unit_cell_images to float:
        unit_cell_images = unit_cell_images.to(torch.float)

        H_idxs, H_blocksizes = self._get_hamiltonian_idxs(atomic_numbers)

        H_diagonals = []
        H_components = []
        H_component_images = []
        H_component_idxs = []
        n_sites = atomic_numbers.shape[0]

        for i in range(n_sites):

            # retrieve site data:
            site = torch.tensor(
                [i],
                dtype=neighbors.dtype,
                device=neighbors.device,
                requires_grad=False,
            )
            site_idxs, site_l_idxs, site_ls = self._get_block_basis_idxs(
                site, atomic_numbers
            )
            site_data = cho_overlap_matrix_metadata(site_ls, sparse=True)

            H_site_idxs = self._get_neighborhood_hamiltonian_idxs(
                site, H_idxs, H_blocksizes
            )

            # retrieve neighborhood data:
            site_neighbors_nz = neighbors[i] >= 0
            site_neighbors = neighbors[i][site_neighbors_nz]
            site_neighbor_images = unit_cell_images[i][site_neighbors_nz]

            # compute site kernel matrix:
            H_onsite = cho_ng_kernel(
                mu_1=positions[site_idxs],
                l_1=site_ls,
                l_alphas_1=alphas[site_idxs, site_l_idxs],
                l_cs_1=cs[site_idxs, site_l_idxs],
                mu_2=positions[site_idxs],
                l_2=site_ls,
                l_alphas_2=alphas[site_idxs, site_l_idxs],
                l_cs_2=cs[site_idxs, site_l_idxs],
                cho_data_1=site_data,
                cho_data_2=site_data,
            )

            H_onsite_idxs = torch.cartesian_prod(H_site_idxs, H_site_idxs)
            _onsite_off_diag = H_onsite_idxs[:, 0] != H_onsite_idxs[:, 1]

            H_onsite_diagonal = torch.diagonal(H_onsite, dim1=-2, dim2=-1)
            H_onsite_offdiag_idxs = H_onsite_idxs[_onsite_off_diag]
            H_onsite_offdiag = H_onsite.flatten(start_dim=-2)[
                ..., _onsite_off_diag
            ]
            H_onsite_offdiag_images = torch.zeros(
                (H_onsite_offdiag_idxs.shape[0], 3),
                device=site_neighbor_images.device,
                dtype=site_neighbor_images.dtype,
            )

            # handle if neighborhood is empty:
            if site_neighbors.shape[0] == 0:
                H_neighborhood = torch.zeros(
                    (0),
                    device=H_onsite_diagonal.device,
                    dtype=H_onsite_diagonal.dtype,
                )
                H_neighborhood_idxs = torch.zeros(
                    (0, 2),
                    dtype=H_onsite_offdiag_idxs.dtype,
                    device=H_onsite_offdiag_idxs.device,
                )
                H_neighborhood_images = torch.zeros(
                    (0, 3),
                    dtype=H_onsite_offdiag_images.dtype,
                    device=H_onsite_offdiag_images.device,
                )
            else:
                # otherwise, compute neighborhood overlap matrices:
                neighbor_idxs, neighbor_images, neighbor_l_idxs, neighbor_ls = (
                    self._get_block_image_basis_idxs(
                        site_neighbors, site_neighbor_images, atomic_numbers
                    )
                )

                neighbor_data = cho_overlap_matrix_metadata(
                    neighbor_ls, sparse=True
                )

                H_neighbor_idxs = self._get_neighborhood_hamiltonian_idxs(
                    site_neighbors, H_idxs, H_blocksizes
                )

                # compute distances to neighbors across unit cell boundaries:
                neighbor_positions = (
                    positions[neighbor_idxs] + neighbor_images @ unit_cell
                )

                # compute neighborhood Hamiltonian matrix:
                H_neighborhood = cho_ng_kernel(
                    mu_1=positions[site_idxs],
                    l_1=site_ls,
                    l_alphas_1=alphas[site_idxs, site_l_idxs],
                    l_cs_1=cs[site_idxs, site_l_idxs],
                    mu_2=neighbor_positions,
                    l_2=neighbor_ls,
                    l_alphas_2=alphas[neighbor_idxs, neighbor_l_idxs],
                    l_cs_2=cs[neighbor_idxs, neighbor_l_idxs],
                    cho_data_1=site_data,
                    cho_data_2=neighbor_data,
                )

                H_neighborhood_images = (
                    self._get_neighborhood_hamiltonian_images(
                        neighbor_images,
                        neighbor_ls,
                    )
                    .view(1, -1, 3)
                    .expand(
                        H_neighborhood.shape[-2], H_neighborhood.shape[-1], 3
                    )
                    .reshape(-1, 3)
                )

                H_neighborhood = H_neighborhood.flatten(start_dim=-2)

                H_neighborhood_idxs = torch.cartesian_prod(
                    H_site_idxs, H_neighbor_idxs
                )

            H_offdiag = torch.cat(
                (H_onsite_offdiag, H_neighborhood, H_neighborhood.conj()),
                dim=-1,
            )
            H_offdiag_idxs = torch.cat(
                (
                    H_onsite_offdiag_idxs,
                    H_neighborhood_idxs,
                    H_neighborhood_idxs.flip(-1),
                )
            )
            H_offdiag_images = torch.cat(
                (
                    H_onsite_offdiag_images,
                    H_neighborhood_images,
                    -H_neighborhood_images,
                )
            )

            H_diagonals.append(H_onsite_diagonal)
            H_components.append(H_offdiag)
            H_component_idxs.append(H_offdiag_idxs)
            H_component_images.append(H_offdiag_images)

        onsite_values = torch.cat(H_diagonals, dim=-1)
        couplings = torch.cat(H_component_idxs)
        coupling_values = torch.cat(H_components, dim=-1)
        coupling_images = torch.cat(H_component_images)

        return onsite_values, couplings, coupling_values, coupling_images

    def hamiltonian_matrix(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        k_pts: torch.Tensor | None = None,
        sparse: bool = False,
    ) -> torch.Tensor:
        """Computes the Hamiltonian matrices for supplied k points.

        Args:
            atomic_numbers (torch.IntTensor): The atomic numbers of
                the atoms (shape: [N_atoms]).
            positions (torch.Tensor): The cartesian coordinates of
                the atoms (shape: [N_atoms,3]).
            neighbors (torch.IntTensor): A list of indexes of atoms
                that are neighbors to each atom. Negative indexes are
                ignored, allowing each atom to have a variable number
                of neighbors up to at most `N_neighbors` neighbors.
                (shape: [N_atoms, N_neighbors]).
            unit_cell (torch.Tensor): The unit cell matrix of the system,
                where each row corresponds to a unit cell vector in
                cartesian coordinates (shape: [3,3]).
            unit_cell_images (torch.IntTensor): Unit cell image
                where each neighbor appears. Images where the
                corresponding element of `neighbors` is negative
                are ignored (shape: [N_atoms, N_neighbors, 3]).
            k_pts (torch.Tensor | None, optional):
                Reciprocal space points (shape: [N_k, 3]). If None, only k=0
                is used. Defaults to None.

        Returns:
            torch.Tensor: The Hamiltonian matrices for each k point
            (shape: [N_k, H_size, H_size]).
        """

        if k_pts is None:
            k_pts = torch.zeros((1, 3), device=positions.device)

        frac_positions = positions @ torch.linalg.pinv(unit_cell)

        site_indices, _ = self.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        onsite_values, couplings, coupling_values, images = (
            self.hamiltonian_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
            )
        )

        return build_matrix(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            images=images,
            k_pts=k_pts,
            frac_positions=frac_positions,
            sparse=sparse,
        )

    def eigenstates(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        k_pts: torch.Tensor | None = None,
    ):

        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
        )

        return self.eigensolver.diagonalize(H)

    def eigenenergies(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        k_pts: torch.Tensor | None = None,
    ):

        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
        )

        return self.eigensolver(H)

    def forward(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor,
        unit_cell_images: torch.IntTensor,
        k_pts: torch.Tensor | None = None,
        sparse: bool = False,
    ):

        return self.eigenenergies(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
        )


def _make_atomiwise_nn(
    nn: str | AtomwiseConv,
    out_features: int,
    in_features: int | None = None,
    name: str = "nn",
    z_max: int = 118,
    nn_kwargs: dict | None = None,
):

    nn_kwargs = {} if nn_kwargs is None else nn_kwargs

    # if AtomwiseConv instance given, check that input/output features are correct:
    if isinstance(nn, AtomwiseConv):
        if nn.in_features != in_features:
            raise ValueError(
                f"Error in creating AtomwiseConv model. "
                f"Expected {name}.in_features == {in_features} (got {nn.in_features})."
            )

        if nn.out_features != out_features:
            raise ValueError(
                f"Error in creating AtomwiseConv model. "
                f"Expected {name}.out_features == {out_features} (got {nn.out_features})."
            )
    elif isinstance(nn, str):
        nn = AtomwiseConv.from_name(
            module_name=nn,
            out_features=out_features,
            in_features=in_features,
            z_max=z_max,
            **nn_kwargs,
        )
    else:
        raise ValueError(
            f'Unexpected type for module "{name}" (got {type(nn)}).'
        )

    return nn


class VariationalLCAOHamiltonian(LCAOHamiltonian):

    def __init__(
        self,
        variational_nn: str | AtomwiseConv = "nnconv",
        onsite_nn: str | AtomwiseConv = "nnconv",
        eigensolver: str | Eigensolver = "torch_eigh",
        spin: str = "paired",
        basis: str = "even_tempered",
        basis_size: int = 3,
        potential_basis: str = "even_tempered",
        potential_basis_size: int = 5,
        energy_scale: float = 1.0,
        onsite_weight: float = 1.0,
        variational_weight: float = 1.0,
        orbital_shells: dict | None = None,
        variational_nn_kwargs: dict | None = None,
        onsite_nn_kwargs: dict | None = None,
    ):

        super().__init__(
            eigensolver=eigensolver,
            spin=spin,
            basis=basis,
            basis_size=basis_size,
            potential_basis=potential_basis,
            potential_basis_size=potential_basis_size,
            energy_scale=energy_scale,
            orbital_shells=orbital_shells,
        )

        n_shells = self.orbital_shell_idxs.shape[-1]

        in_features = (n_shells * self.basis_size) + self.potential_basis_size
        variational_out_features = (
            n_shells * self.basis_size
        ) + self.potential_basis_size
        onsite_out_features = n_shells

        self.onsite_weight = onsite_weight
        self.variational_weight = variational_weight

        if onsite_nn_kwargs is None:
            onsite_nn_kwargs = {}
        if variational_nn_kwargs is None:
            variational_nn_kwargs = {}

        self.onsite_nn = _make_atomiwise_nn(
            nn=onsite_nn,
            out_features=onsite_out_features,
            in_features=in_features,
            name="onsite_nn",
            nn_kwargs=onsite_nn_kwargs,
        )

        self.variational_nn = _make_atomiwise_nn(
            nn=variational_nn,
            out_features=variational_out_features,
            in_features=in_features,
            name="variational_nn",
            nn_kwargs=variational_nn_kwargs,
        )

    def get_onsite_energies(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
    ) -> torch.Tensor:

        # get alpha & gamma orbital parameters
        logscale_alphas, logscale_gammas = (
            self._get_logscale_basis_width_parameters(
                atomic_numbers=atomic_numbers
            )
        )

        logscale_alphas_flat = logscale_alphas.view(atomic_numbers.shape[0], -1)
        logscale_gammas_flat = logscale_gammas.view(atomic_numbers.shape[0], -1)

        # combine to obtain input feature tensor to variational nn
        nn_features_in = torch.cat(
            (logscale_alphas_flat, logscale_gammas_flat), dim=-1
        )

        # compute neighbor distances
        neighbor_distances = positions[neighbors] - positions.unsqueeze(-2)

        # apply variational nn
        onsite_energy_perts = self.onsite_weight * self.onsite_nn(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=nn_features_in,
        )

        # compute total on-site energy
        logscale_energies = self.logscale_onsite_energies[atomic_numbers]

        return self.energy_scale * (
            torch.sinh(logscale_energies) + onsite_energy_perts
        )

    def get_basis_parameters(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
    ) -> tuple:

        # get alpha & gamma orbital parameters
        logscale_alphas, logscale_gammas = (
            self._get_logscale_basis_width_parameters(
                atomic_numbers=atomic_numbers
            )
        )

        alphas = torch.exp(logscale_alphas)
        gammas = torch.exp(logscale_gammas)

        logscale_alphas_flat = logscale_alphas.view(atomic_numbers.shape[0], -1)
        logscale_gammas_flat = logscale_gammas.view(atomic_numbers.shape[0], -1)

        # combine to obtain input feature tensor to variational nn
        nn_features_in = torch.cat(
            (logscale_alphas_flat, logscale_gammas_flat), dim=-1
        )

        # compute neighbor distances
        neighbor_distances = positions[neighbors] - positions.unsqueeze(-2)

        # apply variational nn
        nn_features_out = self.variational_weight * self.variational_nn(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=nn_features_in,
        )

        # unpack basis coefficient perturbations
        c_perts_flat, v_perts_flat = torch.split(
            nn_features_out,
            (logscale_alphas_flat.shape[-1], logscale_gammas_flat.shape[-1]),
            dim=1,
        )

        # compute normalized c coeffficients
        logscale_cs = self.logscale_orbital_cs[atomic_numbers]
        cs_unnormed = torch.sinh(logscale_cs) + c_perts_flat.reshape(
            logscale_cs.shape
        )
        cs = cs_unnormed * self._get_c_normalization_coeffs(
            atomic_numbers=atomic_numbers, alphas=alphas, cs=cs_unnormed
        )

        # compute v coefficients
        logscale_vs = self.logscale_potential_vs[atomic_numbers]
        vs = -self.energy_scale * (
            torch.exp(logscale_vs) + v_perts_flat.reshape(logscale_vs.shape)
        )

        return alphas, cs, gammas, vs


class VariationalTBHamiltonian(TBHamiltonian):

    def __init__(
        self,
        base_model: TBHamiltonian | None = None,
        freeze: bool = False,
        variational_nn: str | AtomwiseConv = "nnconv",
        onsite_nn: str | AtomwiseConv = "nnconv",
        onsite_weight: float = 1.0,
        variational_weight: float = 1.0,
        basis_weight: float = 1.0,
        variational_nn_kwargs: dict | None = None,
        onsite_nn_kwargs: dict | None = None,
        **base_model_kwargs,
    ):

        if base_model is not None:
            super().__init__(
                eigensolver=base_model.eigensolver,
                spin=base_model.spin,
                basis=base_model.basis,
                basis_size=base_model.basis_size,
                potential_basis=base_model.potential_basis,
                potential_basis_size=base_model.potential_basis_size,
                energy_scale=base_model.energy_scale,
                orbital_shells=base_model.get_orbital_shells_dict(),
                **base_model_kwargs,
            )
            self._load_base_model_weights(base_model=base_model, freeze=freeze)
        else:
            super().__init__(**base_model_kwargs)

        n_shells = self.orbital_shell_idxs.shape[-1]

        in_features = (n_shells * self.basis_size) + self.potential_basis_size

        variational_out_features = (
            (n_shells * self.basis_size)
            + (n_shells * self.logscale_orbital_alphas.size(-1))
            + self.potential_basis_size
            + self.logscale_potential_gammas.size(-1)
        )

        onsite_out_features = n_shells

        self.onsite_weight = onsite_weight
        self.variational_weight = variational_weight
        self.basis_weight = basis_weight

        if onsite_nn_kwargs is None:
            onsite_nn_kwargs = {}
        if variational_nn_kwargs is None:
            variational_nn_kwargs = {}

        self.onsite_nn = _make_atomiwise_nn(
            nn=onsite_nn,
            out_features=onsite_out_features,
            in_features=None,
            name="onsite_nn",
            nn_kwargs=onsite_nn_kwargs,
        )

        self.variational_nn = _make_atomiwise_nn(
            nn=variational_nn,
            out_features=variational_out_features,
            in_features=None,
            name="variational_nn",
            nn_kwargs=variational_nn_kwargs,
        )

    def get_onsite_energies(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.IntTensor,
        unit_cell_images: torch.Tensor,
    ) -> torch.Tensor:

        # get alpha & gamma orbital parameters
        logscale_alphas, logscale_gammas = (
            self._get_logscale_basis_width_parameters(
                atomic_numbers=atomic_numbers
            )
        )

        logscale_alphas_flat = logscale_alphas.view(atomic_numbers.shape[0], -1)
        logscale_gammas_flat = logscale_gammas.view(atomic_numbers.shape[0], -1)

        # combine to obtain input feature tensor to variational nn
        nn_features_in = torch.cat(
            (logscale_alphas_flat, logscale_gammas_flat), dim=-1
        )

        # compute neighbor distances
        unit_cell_images = unit_cell_images.to(torch.float)
        neighbor_distances = (
            positions[neighbors]
            + (unit_cell_images @ unit_cell)
            - positions.unsqueeze(-2)
        )

        # apply variational nn
        onsite_energy_perts = self.onsite_weight * self.onsite_nn(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=None,
        )

        # compute total on-site energy
        logscale_energies = self.logscale_onsite_energies[atomic_numbers]

        return self.energy_scale * (
            torch.sinh(logscale_energies) + onsite_energy_perts
        )

    def get_basis_parameters(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.IntTensor,
        unit_cell_images: torch.Tensor,
    ) -> torch.Tensor:

        n_alpha_params = self.logscale_orbital_alphas[0].flatten().size(-1)
        n_gamma_params = self.logscale_potential_gammas[0].flatten().size(-1)
        alpha_params_shape = self.logscale_orbital_alphas[atomic_numbers].shape
        gamma_params_shape = self.logscale_potential_gammas[
            atomic_numbers
        ].shape

        # get alpha & gamma orbital parameters
        logscale_alphas, logscale_gammas = (
            self._get_logscale_basis_width_parameters(
                atomic_numbers=atomic_numbers
            )
        )

        logscale_alphas_flat = logscale_alphas.view(atomic_numbers.shape[0], -1)
        logscale_gammas_flat = logscale_gammas.view(atomic_numbers.shape[0], -1)

        n_c_params = logscale_alphas_flat.shape[-1]
        n_v_params = logscale_gammas_flat.shape[-1]

        # combine to obtain input feature tensor to variational nn
        # nn_features_in = torch.cat(
        #     (logscale_alphas_flat, logscale_gammas_flat), dim=-1)

        # compute neighbor distances
        unit_cell_images = unit_cell_images.to(torch.float)
        neighbor_distances = (
            positions[neighbors]
            + (unit_cell_images @ unit_cell)
            - positions.unsqueeze(-2)
        )

        # apply variational nn
        nn_features_out = self.variational_nn(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=None,
        )

        # unpack basis coefficient perturbations
        c_perts_flat, alpha_perts_flat, v_perts_flat, gamma_perts_flat = (
            torch.split(
                nn_features_out,
                (n_c_params, n_alpha_params, n_v_params, n_gamma_params),
                dim=1,
            )
        )

        # compute perturbed alpha and gammas
        alphas = torch.clamp(
            torch.exp(logscale_alphas)
            + self.basis_weight * alpha_perts_flat.reshape(alpha_params_shape),
            min=1e-3,
        )
        gammas = torch.clamp(
            torch.exp(logscale_gammas)
            + self.basis_weight * gamma_perts_flat.reshape(gamma_params_shape),
            min=1e-3,
        )

        # compute normalized c coeffficients
        logscale_cs = self.logscale_orbital_cs[atomic_numbers]
        cs_unnormed = torch.sinh(
            logscale_cs
        ) + self.variational_weight * c_perts_flat.reshape(logscale_cs.shape)
        cs = cs_unnormed * self._get_c_normalization_coeffs(
            atomic_numbers=atomic_numbers, alphas=alphas, cs=cs_unnormed
        )

        # compute v coefficients
        logscale_vs = self.logscale_potential_vs[atomic_numbers]
        vs = -self.energy_scale * (
            torch.exp(logscale_vs)
            + self.variational_weight * v_perts_flat.reshape(logscale_vs.shape)
        )

        return alphas, cs, gammas, vs
