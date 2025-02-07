from pyquinn.torch.hamiltonian import HamiltonianModel
from pyquinn.torch.eigensolver import Eigensolver

from pyquinn.torch.lcao.gto import (
    _ELEMENT_CHO_VALENCE_SHELLS,
    GTO_L_MAX,
    cho_ng_potential_overlap_matrix,
    cho_overlap_matrix_metadata,
    _get_cho_orbital_name
)

import torch
import torch.nn as nn

import numpy as np


class LCAOModel(HamiltonianModel):
    r"""LCAO (Linear Combination of Atomic Orbitals) Hamiltonian.
    This model contains a fixed set of Gaussian 
    orbital parameters, which are trainable `torch.nn.Parameter`
    values. This model constructs a mean-field Hamiltonian for 
    a non-periodic atomic system with using a minimal basis of 
    Gaussian cubic harmonic orbitals at each site. 
    The Hamiltonian it contructs is real-valued and symmetric.

    When computing the Hamiltonian, this model computes the overlap
    matrices of each neighboring orbital state with an on-site 
    pseudopotential that is expanded as a set of Gaussian functions:

        .. math::
            H_{ij} = \langle \phi_i | V_i + V_j \phi_j \rangle

    This model currently supports the following eigensolvers:
        - `torch_eigh`

    """

    BASIS_OPTIONS = [
        'variable',
        'even_tempered',
        'fixed'
    ]

    SPIN_OPTIONS = [
        'paired'
    ]

    @classmethod
    def _make_shell_idxs(cls, shells: dict):
        """ Constructs a padded tensor that maps [atomic_number,i] -> l. 
        l values of -1 should be ignored."""
        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        shell_idxs = torch.full((z_max+1, n_alpha_sets), -1, dtype=torch.int64)

        for z, l_list in shells.items():
            if z <= 0:
                raise ValueError(f'Expected positive atomic number in '
                                 f'"shells" dictionary, got: {z}')
            for i, l in enumerate(l_list):
                if l < 0 or l > GTO_L_MAX:
                    raise ValueError(f'Expected 0 <= l <= {GTO_L_MAX} '
                                     f'for element {z} '
                                     f'in shell {i}, got: l = {l}')

                shell_idxs[z, i] = l
        return shell_idxs

    @classmethod
    def _make_logscale_basis_parameters(
            cls, 
            basis: str, basis_size: int, shells: dict, 
            r_max: float = 5.0, r_min: float = 5e-2) -> torch.nn.Parameter:
        """Constructs a set of trainable log-scale Gaussian orbital 
        parameters. """

        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        if basis not in LCAOModel.BASIS_OPTIONS:
            raise ValueError(f'Basis must be one of {LCAOModel.BASIS_OPTIONS},'
                             f' got: {basis}.')
        if basis_size <= 0:
            raise ValueError('Basis size must be a positive integer.')
        if r_max <= 0 or r_min <= 0 or r_min >= r_max:
            raise ValueError('Values r_min and r_max must satisfy'
                             ' 0 < r_min < r_max.')

        log_alpha_r_max = np.log(1/(2*r_max**2))
        log_alpha_r_min = np.log(1/(2*r_min**2))
        beta_min_max = (log_alpha_r_min - log_alpha_r_max) / (basis_size-1)

        if basis == 'variable':
            logscale_alphas = nn.Parameter(
                torch.ones(z_max+1, n_alpha_sets, basis_size)*log_alpha_r_max)
            logscale_betas = nn.Parameter(
                torch.zeros(z_max+1, n_alpha_sets, 1), requires_grad=False)

        elif basis == 'even_tempered':
            logscale_alphas = nn.Parameter(
                torch.ones(z_max+1, n_alpha_sets, 1)*log_alpha_r_max, 
                requires_grad=True)
            logscale_betas = nn.Parameter(
                torch.ones(z_max+1, n_alpha_sets, 1)*beta_min_max, 
                requires_grad=True)

        elif basis == 'fixed':
            logscale_alphas = nn.Parameter(
                torch.ones(z_max+1, n_alpha_sets, 1)*log_alpha_r_max, 
                requires_grad=False)
            logscale_betas = nn.Parameter(
                torch.ones(z_max+1, n_alpha_sets, 1)*beta_min_max, 
                requires_grad=False)

        logscale_cs = torch.randn(z_max, n_alpha_sets, basis_size)

        return logscale_alphas, logscale_betas, logscale_cs

    @classmethod
    def _make_logscale_energy_parameters(cls, shells: dict):
        """Constructs a set of trainable log-scale orbital 
        energy parameters."""

        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        logscale_energies = nn.Parameter(
            torch.sort(torch.randn(z_max+1, n_alpha_sets))[0],
            requires_grad=True
        )

        return logscale_energies

    def __init__(self, 
                 eigensolver: str | Eigensolver | None = 'torch_eigh',
                 spin: str = 'paired',
                 basis: str = 'even_tempered',
                 basis_size: int = 3,
                 potential_basis: str = 'even_tempered',
                 potential_basis_size: int = 5,
                 energy_scale: float = 1.0,
                 orbital_shells: dict = {},
                 ) -> None:
        """Constructs a LCAO (Linear Combination of Atomic Orbitals) 
        Hamiltonian model. This model contains a fixed set of Gaussian 
        orbital parameters, which are trainable `torch.nn.Parameter`
        values. Depending on the basis options specified, these 
        parameters may be frozen (i.e., have `requires_grad=False`).

        Args:
            eigensolver (str | Eigensolver | None, optional): The Eigensolver 
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
                If the model is trained on different energy units (such as
                Hartree versus eV units), this scalar may need to be adjusted.
                Defaults to 1.0.
            orbital_shells (dict, optional): A dictionary mapping atomic 
                numbers to orbital shell angular momenta :math:`l`. 
                For example, to model the 1s, 2s, and 2p orbitals of 
                Carbon atoms, one would use `orbital_shells={6: [0, 0, 1]}`
                to model the :math:`l=0,l=0` and :math:`l=1` angular momenta of 
                these orbitals. For any elements not specified in this 
                dictionary, the valence shell orbitals are used.
                Defaults to {}.

        Raises:
            ValueError: If invalid basis options are supplied.
        """

        super().__init__(eigensolver=eigensolver)

        self.basis = basis
        self.basis_size = basis_size
        self.potential_basis_size = potential_basis_size

        if spin not in LCAOModel.SPIN_OPTIONS:
            raise ValueError(
                f'Spin must be one of {LCAOModel.SPIN_OPTIONS}, got: {spin}')

        self.spin = spin

        # use default valence shells to start and override with supplied values
        orbital_shells = _ELEMENT_CHO_VALENCE_SHELLS | orbital_shells

        # initialize shell index parameters
        self.orbital_shell_idxs = LCAOModel._make_shell_idxs(orbital_shells)

        self.z_max = max(orbital_shells.keys())

        self.energy_scale = energy_scale

        # initialize log-scale orbital gaussian parameters
        self.logscale_orbital_alphas,\
            self.logscale_orbital_betas,\
            self.logscale_orbital_cs = \
            LCAOModel._make_logscale_basis_parameters(
                basis=basis,
                basis_size=basis_size,
                shells=orbital_shells
            )

        # initialize log-scale potential gaussian parameters
        # (use only s-type gaussian potential "shells" for now)
        potential_shells = {n: [0] for n in orbital_shells}
        self.logscale_potential_gammas,\
            self.logscale_potential_betas,\
            self.logscale_potential_vs = \
            LCAOModel._make_logscale_basis_parameters(
                basis=potential_basis,
                basis_size=potential_basis_size,
                shells=potential_shells
            )

        # initialize on-site logscale energies
        self.logscale_onsite_energies = \
            LCAOModel._make_logscale_energy_parameters(shells=orbital_shells)

    def get_basis_parameters(self, 
                             atomic_numbers: torch.IntTensor, 
                             positions: torch.Tensor, 
                             neighbors: torch.IntTensor
                             ) -> tuple:
        """Computes the Gaussian basis parameters of an atomic system.


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
                coefficients (shape: [N_atoms, N_ls, N_basis]).
            - gammas (torch.Tensor): A tensor of Gaussian
                potential width parameters (shape: [N_atoms, 1, M_basis]).
            - vs (torch.Tensor): A tensor of Gaussian potential
                coefficients (shape: [N_atoms, 1, N_basis])
        """

        # get log-scale orbital parameters
        logscale_alphas = self.logscale_orbital_alphas[atomic_numbers]
        logscale_betas = self.logscale_orbital_betas[atomic_numbers]
        logscale_cs = self.logscale_orbital_cs[atomic_numbers]

        beta_powers = torch.arange(0, logscale_cs.shape[-1])\
            .view(1, 1, -1)
        alphas = torch.exp(logscale_alphas + logscale_betas*beta_powers)

        cs = torch.sinh(logscale_cs)

        # get log-scale potential parameters:
        logscale_v_gammas = self.logscale_potential_gammas[atomic_numbers]
        logscale_v_betas = self.logscale_potential_betas[atomic_numbers]
        logscale_vs = self.logscale_potential_vs[atomic_numbers]

        v_beta_powers = torch.arange(0, logscale_vs.shape[-1])\
            .view(1, 1, -1)

        gammas = torch.exp(logscale_v_gammas + logscale_v_betas*v_beta_powers)

        vs = -self.energy_scale * torch.exp(logscale_vs)

        return alphas, cs, gammas, vs

    def get_onsite_energies(self,
                            atomic_numbers: torch.IntTensor, 
                            positions: torch.Tensor, 
                            neighbors: torch.IntTensor
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

        return self.energy_scale*torch.sinh(logscale_energies)

    def _get_block_basis_idxs(
            self,
            idxs: torch.Tensor,
            atomic_numbers: torch.Tensor) -> tuple:

        block_atoms = atomic_numbers[idxs]
        block_idxs, block_l_idxs = \
            (self.orbital_shell_idxs[block_atoms] >= 0)\
            .nonzero(as_tuple=True)

        block_idxs = idxs[block_idxs]

        block_ls = self.orbital_shell_idxs[
            atomic_numbers[block_idxs], 
            block_l_idxs
        ]

        return block_idxs, block_l_idxs, block_ls

    def _get_hamiltonian_idxs(
            self, 
            atomic_numbers: torch.IntTensor) -> tuple:

        H_blocksizes = torch.tensor(
            [sum(2*l + 1 for l in self.orbital_shell_idxs[z] if l >= 0)
                for z in atomic_numbers],
            dtype=atomic_numbers.dtype,
            device=atomic_numbers.device,
            requires_grad=False)

        H_idxs = torch.cumsum(H_blocksizes, dim=0) - H_blocksizes

        return H_idxs, H_blocksizes

    def _get_neighborhood_hamiltonian_idxs(
            self,
            idxs: torch.IntTensor,
            H_idxs: torch.IntTensor,
            H_blocksizes: torch.IntTensor) -> torch.IntTensor:

        H_neighborhood_idxs = torch.cat([
            H_idxs[i] + torch.arange(H_blocksizes[i])
            for i in idxs
        ])

        return H_neighborhood_idxs

    def _build_hamiltonian(
            self,
            diagonals: list[torch.Tensor],
            components: list[torch.Tensor], 
            component_idxs: list[torch.IntTensor],
            shape: torch.Tensor,
            sparse: bool = False):

        H_ij = torch.cat(component_idxs, dim=0).T
        H_values = torch.cat(components, dim=0)

        H_diagonal = torch.cat(diagonals, dim=0)
        H_ii = torch.arange(H_diagonal.shape[0])\
            .view(1, -1).expand(2, -1)

        if sparse:
            # append values and hermitian conjugate values
            H_ji = H_ij[torch.tensor([1, 0])]
            H_indices_total = torch.cat((H_ii, H_ij, H_ji), dim=-1)
            H_values_total = torch.cat(
                (H_diagonal, H_values, H_values.conj()), dim=-1)

            # build sparse tensor
            H = torch.sparse_coo_tensor(
                indices=H_indices_total, 
                values=H_values_total,
                size=shape,
                device=H_values_total.device).coalesce()

        else:
            # build dense tensor:
            H = torch.zeros(
                shape[0]*shape[1],
                dtype=H_values.dtype,
                device=H_values.device)

            H_ii_idxs = shape[0]*H_ii[0] + H_ii[1]
            H.put_(H_ii_idxs, H_diagonal, accumulate=True)

            H_ij_idxs = shape[0]*H_ij[0] + H_ij[1]
            H.put_(H_ij_idxs, H_values, accumulate=True)

            H_ji_idxs = shape[0]*H_ij[1] + H_ij[0]
            H.put_(H_ji_idxs, H_values, accumulate=True)

            H = H.reshape(shape)

        return H

    def hamiltonian_matrix(self, 
                           atomic_numbers: torch.IntTensor, 
                           positions: torch.Tensor, 
                           neighbors: torch.IntTensor,
                           sparse: bool = False,
                           ) -> torch.Tensor:
        """_summary_

        Args:
            atomic_numbers (torch.IntTensor): _description_
            positions (torch.Tensor): _description_
            neighbors (torch.IntTensor): _description_
            sparse (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """

        # get orbital parameters:
        alphas, cs, gammas, vs = \
            self.get_basis_parameters(atomic_numbers, positions, neighbors)

        # get on-site energies:
        onsite_energies = self.get_onsite_energies(
            atomic_numbers, positions, neighbors)

        H_idxs, H_blocksizes = self._get_hamiltonian_idxs(atomic_numbers)
        H_size = H_blocksizes.sum()

        H_diagonals = []
        H_components = []
        H_component_idxs = []

        n_sites = atomic_numbers.shape[0]

        for i in range(n_sites):

            # retrieve site data:
            site = torch.tensor(
                [i], dtype=neighbors.dtype, 
                device=neighbors.device,
                requires_grad=False)
            site_idxs, site_l_idxs, site_ls = \
                self._get_block_basis_idxs(site, atomic_numbers)
            site_data = cho_overlap_matrix_metadata(site_ls, sparse=sparse)

            H_site_idxs = self._get_neighborhood_hamiltonian_idxs(
                site, H_idxs, H_blocksizes)

            # retrieve neighborhood data:
            site_neighbors_pad = neighbors[i]
            site_neighbors = site_neighbors_pad[site_neighbors_pad >= 0]

            neighbor_idxs, neighbor_l_idxs, neighbor_ls = \
                self._get_block_basis_idxs(site_neighbors, atomic_numbers)

            neighbor_data = cho_overlap_matrix_metadata(
                neighbor_ls, sparse=sparse)

            H_neighbor_idxs = self._get_neighborhood_hamiltonian_idxs(
                site_neighbors, H_idxs, H_blocksizes)

            # compute on-site diagonals:
            H_site_diagonal = onsite_energies[site_idxs, site_l_idxs]

            # compute neighborhood Hamiltonian matrix:
            H_neighborhood = cho_ng_potential_overlap_matrix(
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
            )

            H_neighborhood_idxs = torch.cartesian_prod(
                H_site_idxs, H_neighbor_idxs)

            H_diagonals.append(H_site_diagonal)
            H_components.append(H_neighborhood.flatten())
            H_component_idxs.append(H_neighborhood_idxs)

        return self._build_hamiltonian(
            H_diagonals, H_components, H_component_idxs, 
            shape=(H_size, H_size), sparse=sparse)

    def hamiltonian_key(self, 
                        atomic_numbers: torch.IntTensor, 
                        positions: torch.Tensor, 
                        neighbors: torch.IntTensor
                        ) -> list[str]:

        key = []
        for z in atomic_numbers:

            l_values = self.orbital_shell_idxs[z]
            l_nz_values = l_values[l_values >= 0]

            for l in l_nz_values:
                for i in range(2*l+1):
                    key.append(
                        _get_cho_orbital_name(l, i))

        return key


class TightBindingModel(LCAOModel):

    def __init__(self, 
                 eigensolver: str | Eigensolver | None = None,
                 spin='paired',
                 basis: str = 'even_tempered',
                 basis_size: int = 3,
                 potential_basis: str = 'even_tempered',
                 potential_basis_size: int = 5,
                 energy_scale: float = 1.0,
                 orbital_shells: dict = {},
                 ) -> None:

        super().__init__(
            eigensolver=eigensolver,
            spin=spin,
            basis=basis,
            basis_size=basis_size,
            potential_basis=potential_basis,
            potential_basis_size=potential_basis_size,
            energy_scale=energy_scale,
            orbital_shells=orbital_shells
        )

    def _get_neighborhood_hamiltonian_distances(
            self,
            neighbor_distances: torch.Tensor,
            neighbor_ls: torch.IntTensor) -> torch.IntTensor:

        H_neighborhood_dxs = torch.cat([
            dx.unsqueeze(0).expand(2*l+1, -1)
            for dx, l in zip(neighbor_distances, neighbor_ls)            
        ])

        return H_neighborhood_dxs

    def _build_bloch_hamiltonian(
            self,
            diagonals: list[torch.Tensor],
            components: list[torch.Tensor],
            component_idxs: list[torch.IntTensor],
            component_distances: list[torch.Tensor],
            k_pts: torch.Tensor,
            shape: torch.Tensor,
            sparse: bool = False):

        H_ij = torch.cat(component_idxs, dim=0).T
        H_values = torch.cat(components, dim=0)

        H_diagonal = torch.cat(diagonals, dim=0)
        H_ii = torch.arange(H_diagonal.shape[0])\
            .view(1, -1).expand(2, -1)

        H_distances = torch.cat(component_distances, dim=0)
        k_shape = k_pts.shape[:-1]
        k_view = tuple([1]*len(k_shape))

        if sparse:

            # move batch dimensions to the trailing dimensions
            # (This is necessary due to the fact that pytorch
            #  only supports efficient sparse formats with the
            #  leading axes as sparse)
            H_thetas_k = torch.einsum(
                'id, ...d -> i...',
                H_distances,
                2*torch.pi*k_pts
            )

            H_values_k = torch.einsum(
                'i, i... -> i...',
                H_values,
                torch.exp(1.j*H_thetas_k))

            H_diagonal_k = H_diagonal.view(-1, *k_view).expand(-1, *k_shape)

            # append values and hermitian conjugate values
            H_ji = H_ij[torch.tensor([1, 0])]
            H_indices_total = torch.cat((H_ii, H_ij, H_ji), dim=-1)
            H_values_total = torch.cat(
                (H_diagonal_k, H_values_k, H_values_k.conj()), dim=0)

            # build sparse tensor
            H = torch.sparse_coo_tensor(
                indices=H_indices_total, 
                values=H_values_total,
                size=shape + k_shape,
                device=H_values_total.device).coalesce()
        else:

            # build dense hamiltonian
            H = torch.zeros(k_shape + (shape[0]*shape[1],), 
                            dtype=torch.complex64)

            H_ii_idxs = (shape[0]*H_ii[0] + H_ii[1])\
                .view(*k_view, -1).expand(*k_shape, -1)

            H_diagonal_k = H_diagonal.to(torch.complex64)\
                .view(*k_view, -1).expand(*k_shape, -1)

            H.scatter_add_(-1, H_ii_idxs, H_diagonal_k)

            H_thetas_k = torch.einsum(
                'id, ...d -> ...i',
                H_distances,
                2*torch.pi*k_pts
            )

            H_values_k = torch.einsum(
                'i, ...i -> ...i',
                H_values,
                torch.exp(1.j*H_thetas_k))

            H_ij_idxs = (shape[0]*H_ij[0] + H_ij[1])\
                .view(*k_view, -1).expand(*k_shape, -1)

            H.scatter_add_(-1, H_ij_idxs, H_values_k)

            H_ji_idxs = (shape[0]*H_ij[1] + H_ij[0])\
                .view(*k_view, -1).expand(*k_shape, -1)

            H.scatter_add_(-1, H_ji_idxs, H_values_k.conj())

            # restore original shape
            H = H.reshape(k_shape + shape)

        return H

    def hamiltonian_matrix(
            self, 
            atomic_numbers: torch.IntTensor, 
            positions: torch.Tensor, 
            neighbors: torch.IntTensor,
            unit_cell: torch.IntTensor,
            unit_cell_images: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            sparse: bool = False) -> torch.Tensor:

        if k_pts is None:
            k_pts = torch.zeros(3)

        # get orbital parameters:
        alphas, cs, gammas, vs = \
            self.get_basis_parameters(atomic_numbers, positions, neighbors)

        # get on-site energies:
        onsite_energies = self.get_onsite_energies(
            atomic_numbers, positions, neighbors)

        # convert unit_cell_images to float:
        unit_cell_images = unit_cell_images.to(torch.float)

        H_idxs, H_blocksizes = self._get_hamiltonian_idxs(atomic_numbers)
        H_size = H_blocksizes.sum()

        H_diagonals = []
        H_components = []
        H_component_distances = []
        H_component_idxs = []
        n_sites = atomic_numbers.shape[0]

        for i in range(n_sites):

            # retrieve site data:
            site = torch.tensor(
                [i], dtype=neighbors.dtype, 
                device=neighbors.device,
                requires_grad=False)
            site_idxs, site_l_idxs, site_ls = \
                self._get_block_basis_idxs(site, atomic_numbers)
            site_data = cho_overlap_matrix_metadata(site_ls, sparse=True)

            H_site_idxs = self._get_neighborhood_hamiltonian_idxs(
                site, H_idxs, H_blocksizes)

            # retrieve neighborhood data:
            site_neighbors_pad = neighbors[i]
            site_neighbors = site_neighbors_pad[site_neighbors_pad >= 0]

            neighbor_idxs, neighbor_l_idxs, neighbor_ls = \
                self._get_block_basis_idxs(site_neighbors, atomic_numbers)

            neighbor_data = cho_overlap_matrix_metadata(
                neighbor_ls, sparse=True)

            H_neighbor_idxs = self._get_neighborhood_hamiltonian_idxs(
                site_neighbors, H_idxs, H_blocksizes)

            # compute on-site diagonals:
            H_site_diagonal = onsite_energies[site_idxs, site_l_idxs]

            # compute distances to neighbors across unit cell boundaries:
            neighbor_positions = \
                positions[neighbor_idxs] + \
                unit_cell_images[neighbor_idxs, neighbor_l_idxs] @ unit_cell

            # compute neighborhood Hamiltonian matrix:
            H_neighborhood = cho_ng_potential_overlap_matrix(
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
                cho_data_2=neighbor_data)

            H_neighborhood_dxs = self._get_neighborhood_hamiltonian_distances(
                neighbor_positions - positions[i],
                neighbor_ls).view(1, -1, 3).expand(*H_neighborhood.shape, 3)

            H_neighborhood_idxs = torch.cartesian_prod(
                H_site_idxs, H_neighbor_idxs)

            H_diagonals.append(H_site_diagonal)
            H_components.append(H_neighborhood.flatten())
            H_component_idxs.append(H_neighborhood_idxs)
            H_component_distances.append(H_neighborhood_dxs.reshape(-1, 3))

        return self._build_bloch_hamiltonian(
            H_diagonals, H_components, H_component_idxs,
            H_component_distances, k_pts,
            shape=(H_size, H_size), sparse=sparse)

    def eigenstates(
            self, 
            atomic_numbers: torch.IntTensor, 
            positions: torch.Tensor, 
            neighbors: torch.IntTensor,
            unit_cell: torch.IntTensor,
            unit_cell_images: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            sparse: bool = False):

        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
            sparse=sparse
        )

        if sparse and H.ndim > 2:
            raise NotImplementedError(
                'Diagonalization of sparse hamiltonians not supported '
                'for TightBindingModels with specified k points.'
                '(use sparse=False).')

        return self.eigensolver.diagonalize(H)

    def eigenenergies(
            self,
            atomic_numbers: torch.IntTensor,
            positions: torch.Tensor,
            neighbors: torch.IntTensor,
            unit_cell: torch.IntTensor,
            unit_cell_images: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            sparse: bool = False):

        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            k_pts=k_pts,
            sparse=sparse
        )

        if sparse and H.ndim > 2:
            raise NotImplementedError(
                'Diagonalization of sparse hamiltonians not supported '
                'for TightBindingModels with more than one k point. '
                '(Use sparse=False or query each k point individually.)')

        return self.eigensolver(H)
