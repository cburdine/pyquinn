from pyquinn.torch.hamiltonian import HamiltonianModel
from pyquinn.torch.eigensolver import Eigensolver
from pyquinn.torch.lcao.gto import (
    _CHO_BASIS_SIZES, 
    _ELEMENT_CHO_VALENCE_SHELLS, 
    GTO_L_MAX
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
    def _make_shell_idxs(shells: dict):
        z_max = max(shells.keys())
        n_alpha_sets = max(len(l_list) for l_list in shells.values())

        shell_idxs = torch.full((z_max+1, n_alpha_sets), -1, dtype=torch.int64)

        for z, l_list in shells:
            if z <= 0:
                raise ValueError(f'Expected positive atomic number in "shells" dictionary, got: {z}')
            for i, l in enumerate(l_list):
                if l < 0 or l > GTO_L_MAX:
                    raise ValueError(f'Expected 0 <= l <= {GTO_L_MAX} for element {z}'
                                     f' in shell {i}, got: l = {l}')

                shell_idxs[z, i] = l

        return torch.shell_idxs

    @classmethod
    def _make_logscale_basis_parameters(basis: str, basis_size: int, shells: dict, 
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
                 orbital_shells: dict = {},
                 ) -> None:

        super().__init__(eigensolver=eigensolver)

        self.basis = basis
        self.basis_size = basis_size
        self.potential_basis_size = potential_basis_size

        if spin not in LCAOModel.SPIN_OPTIONS:
            raise ValueError(f'Spin must be one of {LCAOModel.SPIN_OPTIONS}, got: {spin}')

        self.spin = spin

        # use default valence shells to start and override with supplied values
        orbital_shells = _ELEMENT_CHO_VALENCE_SHELLS | orbital_shells

        # use only s-type gaussian potential "shells"
        potential_shells = {n: [0] for n in orbital_shells}

        # initialize shell index parameters
        self.orbital_shell_idxs = LCAOModel._make_shell_idxs(orbital_shells)

        self.z_max = max(max(orbital_shells.keys()))

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
        self.logscale_potential_gammas,\
            self.logscale_potential_betas,\
            self.logscale_potential_vs = \
            LCAOModel._make_logscale_basis_parameters(
                basis=potential_basis,
                basis_size=potential_basis_size,
                shells=potential_shells
            )

    def _get_hamiltonian_indices(self, 
                                 atomic_numbers: torch.IntTensor,
                                 ) -> torch.Tensor:

            H_idxs = torch.zeros_like(atomic_numbers, 
                                    dtype=atomic_numbers.dtype,
                                    requires_grad=False)

            H_blocksizes = torch.zeros_like(atomic_numbers, 
                                            dtype=atomic_numbers.dtype,
                                            requires_grad=False)

            for i, z in enumerate(atomic_numbers):

                H_blocksizes[i] = sum(
                    self.orbital_shells[l] for l in self.orbital_shells[z])

                if i > 0:
                    H_idxs[i] = H_idxs[i-1] + H_blocksizes[i-1]

            return H_idxs, H_blocksizes

    def get_orbital_parameters(self, 
                              atomic_numbers: torch.IntTensor, 
                              positions: torch.Tensor, 
                              neighbors: torch.IntTensor) -> torch.Tensor:

        # get log-scale orbital parameters
        logscale_alphas = self.logscale_orbital_alphas[atomic_numbers]
        logscale_betas = self.logscale_orbital_betas[atomic_numbers]
        beta_powers = torch.arange(0, logscale_betas.shape[-1])\
            .reshape(1, 1, -1)

        logscale_cs = self.logscale_orbital_cs[atomic_numbers]

        alphas = torch.exp(logscale_alphas + logscale_betas*beta_powers)
        cs = torch.sinh(logscale_cs)

        return alphas, cs



    def get_potential_parameters(self, 
                                 atomic_numbers: torch.IntTensor, 
                                 positions: torch.Tensor, 
                                 neighbors: torch.IntTensor):
        
        # get log-scale potential parameters:
        logscale_gammas = self.logscale_potential_gammas[atomic_numbers]
        logscale_betas = self.logscale_potential_betas[atomic_numbers]
        beta_powers = torch.arange(0, logscale_betas.shape[-1])\
            .reshape(1, 1, -1)


    def hamiltonian_matrix(self, 
                           atomic_numbers: torch.IntTensor, 
                           positions: torch.Tensor, 
                           neighbors: torch.IntTensor
                           ) -> torch.Tensor:
        
        H_indices, H_blocksizes = self._get_hamiltonian_indices(atomic_numbers)

        alphas, cs = self.get_orbital_parameters(self,
                                                 atomic_numbers=atomic_numbers,
                                                 positions=positions,
                                                 neighbors=neighbors)

        for i, site_atom in enumerate(atomic_numbers):
            site_neighbors = (neighbors[i] >= 0).nonzero()
            site_neighbor_alphas = alphas[site_neighbors]
            site_neighbor_cs = cs[site_neighbors]

            site_shell_ls, = (self.orbital_shell_idxs[i] >= 0)\
                .nonzero(as_tuple=True)

            neighbor_atoms = atomic_numbers[atom_neighbors]
            neighbor_idxs, neighbor_ls = self.orbital_shell_idxs[].nonzero(as_tuple=True)



    def hamiltonian_key(self, 
                        atomic_numbers: torch.IntTensor, 
                        positions: torch.Tensor, 
                        neighbors: torch.IntTensor
                        ) -> list[str]:
        pass

    def forward(self, atomic_numbers):
        pass


class TightBindingModel(LCAOHamiltonianModel):

    def __init__(self):
        super().__init__(self)

