from pyquinn.torch.hamiltonian import HamiltonianModel
from pyquinn.torch.eigensolver import Eigensolver

from pyquinn.torch.lcao.gto import (
    _CHO_BASIS_SIZES, 
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
    def _make_logscale_basis_parameters(cls, basis: str, basis_size: int, shells: dict, 
                                        r_max: float = 5.0, r_min: float = 5e-2):

        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        if basis not in LCAOModel.BASIS_OPTIONS:
            raise ValueError(f'Basis must be one of {LCAOModel.BASIS_OPTIONS}, got: {basis}.')
        if basis_size <= 0:
            raise ValueError('Basis size must be a positive integer.')
        if r_max <= 0 or r_min <= 0 or r_min >= r_max:
            raise ValueError('Values r_min and r_max must satisfy 0 < r_min < r_max.')

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
                torch.ones(z_max, n_alpha_sets, 1)*log_alpha_r_max, 
                requires_grad=True)
            logscale_betas = nn.Parameter(
                torch.ones(z_max, n_alpha_sets, 1)*beta_min_max, 
                requires_grad=True)

        elif basis == 'fixed':
            logscale_alphas = nn.Parameter(
                torch.ones(z_max, n_alpha_sets, 1)*log_alpha_r_max, 
                requires_grad=False)
            logscale_betas = nn.Parameter(
                torch.ones(z_max, n_alpha_sets, 1)*beta_min_max, 
                requires_grad=False)

        logscale_cs = torch.randn(z_max, n_alpha_sets, basis_size)

        return logscale_alphas, logscale_betas, logscale_cs

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

    def get_basis_parameters(self, 
                             atomic_numbers: torch.IntTensor, 
                             positions: torch.Tensor, 
                             neighbors: torch.IntTensor
                             ) -> tuple:

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
            components: list[torch.Tensor], 
            component_idxs: list[torch.IntTensor],
            shape: torch.Tensor,
            sparse: bool = False):

        H_ij = torch.cat(component_idxs, dim=0).T
        H_values = torch.cat(components, dim=0)
        
        if sparse:
            # append hermitian conjugate values:
            H_ji = H_ij[torch.tensor([1, 0])]
            H_ij_conj = torch.cat((H_ij, H_ji), dim=-1)

            H_values_conj = torch.cat(
                (H_values, H_values.conj()), dim=-1)

            # build sparse tensor:
            H = torch.sparse_coo_tensor(
                indices=H_ij_conj, 
                values=H_values_conj,
                size=shape,
                device=H_values_conj.device).coalesce()

        else:
            # build dense tensor:
            H = torch.zeros(
                shape, 
                dtype=H_values.dtype, 
                device=H_values.device)

            H[H_ij[0], H_ij[1]] += H_values
            H[H_ij[1], H_ij[0]] += H_values.conj()

        return H

    def hamiltonian_matrix(self, 
                           atomic_numbers: torch.IntTensor, 
                           positions: torch.Tensor, 
                           neighbors: torch.IntTensor,
                           sparse: bool = False,
                           ) -> torch.Tensor:

        # get orbital parameters:
        alphas, cs, gammas, vs = \
            self.get_basis_parameters(atomic_numbers, positions, neighbors)

        H_idxs, H_blocksizes = self._get_hamiltonian_idxs(atomic_numbers)
        H_size = H_blocksizes.sum()
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

            H_components.append(H_neighborhood.flatten())
            H_component_idxs.append(H_neighborhood_idxs)
        
        return self._build_hamiltonian(
            H_components, H_component_idxs, 
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

    def __init__(self):
        super().__init__(self)

