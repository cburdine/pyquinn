from pyquinn.torch.nn.lcao import TBHamiltonian
from pyquinn.torch.nn import AtomwiseE3Conv

from pyquinn.torch.nn.atomwise_e3 import _scalar_irreps
from pyquinn.torch.nn.lcao.gto import (
    e3_cho_ng_potential_overlap_matrix, 
    e3_cho_overlap_matrix_metadata,
    cho_ng_norm_coeff
)

import e3nn.o3 as o3
import torch


def _make_atomwise_e3_nn(
        nn: AtomwiseE3Conv | None = None, 
        out_irreps: o3.Irreps | None = None,
        in_irreps: o3.Irreps | None = None,
        name: str = 'e3nn',
        nn_kwargs: dict | None = None):

    nn_kwargs = {} if nn_kwargs is None else nn_kwargs

    # if AtomwiseConv instance given, check that input/output features 
    # are correct
    if isinstance(nn, AtomwiseE3Conv):
        if nn.in_irreps.dim != in_irreps.dim:
            raise ValueError(
                f'Error in creating AtomwiseE3Conv model. '
                f'Expected {name}.in_irreps to be compatible with {in_irreps} '
                f'(got {nn.in_irreps}).')

        if nn.out_irreps.dim != out_irreps.dim:
            raise ValueError(
                f'Error in creating AtomwiseE3Conv model. '
                f'Expected {name}.out_irreps to be compatible with '
                f'{out_irreps} (got {nn.out_irreps}).')

    elif nn is None:
        nn = AtomwiseE3Conv(
            out_features=out_irreps,
            in_features=in_irreps,
            **nn_kwargs
        )
    else:
        raise ValueError(f'Unexpected type for AtomwiseE3Conv "{name}" '
                         f'(got {type(nn)}).')

    return nn


def _make_shell_irreps(unique_shells, n_coeffs=1):
    shell_irreps = o3.Irreps()
    for _ in range(n_coeffs):
        for l in unique_shells:
            _l = int(l.item())
            shell_irreps += o3.Irreps([(1, (_l, (-1)**_l))])

    return shell_irreps


def _make_shell_irrep_map(
        orbital_shell_idxs) -> torch.tensor:

    # determine how many instances of each irrep (shell) are needed
    l_max = torch.max(orbital_shell_idxs)
    l_unique_shell_counts = torch.zeros(l_max+1, dtype=torch.int32)
    max_shell_block_size = torch.tensor(0)
    for z, l_arr in enumerate(orbital_shell_idxs):
        l_counts = torch.zeros(l_max+1, dtype=torch.int32)
        ls, counts = torch.unique(l_arr, return_counts=True)
        ls_nz = (ls >= 0)
        ls, counts = ls[ls_nz], counts[ls_nz]
        l_counts[ls] += counts
        l_unique_shell_counts = \
            torch.maximum(l_unique_shell_counts, l_counts)

        block_size = torch.sum((2*ls + 1)*counts)
        max_shell_block_size = torch.maximum(
            max_shell_block_size, block_size)

    # determine the minimal set of unique shells for building all irreps
    l_unique_idxs = torch.cumsum(l_unique_shell_counts, -1) - \
        l_unique_shell_counts
    irrep_blocks = torch.zeros(
        l_unique_shell_counts.sum(), dtype=torch.int32)
    for l, count in enumerate(l_unique_shell_counts):
        idx = l_unique_idxs[l]
        irrep_blocks[idx:idx+count] = l
    
    # assemble the block indices of each unique shell
    irrep_block_sizes = l_unique_shell_counts*(2*torch.arange(l_max+1)+1)
    irrep_block_idxs = torch.cumsum(irrep_block_sizes, -1) - irrep_block_sizes
    
    # construct map of (shell idxs) -> (irrep idxs)
    z_max = orbital_shell_idxs.shape[0]
    irrep_map = torch.full((z_max, max_shell_block_size), -1)
    irrep_shell_map = torch.full((z_max, max_shell_block_size), -1)
    for z, l_arr in enumerate(orbital_shell_idxs):
        l_shell_counts = torch.zeros(l_max+1, dtype=torch.int32)
        shell_idx = 0
        for i, l in enumerate(l_arr):
            if l >= 0:
                block_size = 2*l+1
                irrep_idxs = torch.arange(block_size) + \
                    block_size*l_shell_counts[l] + irrep_block_idxs[l]
                irrep_map[z, shell_idx:shell_idx+block_size] = irrep_idxs
                irrep_shell_map[z, shell_idx:shell_idx+block_size] = i
                l_shell_counts[l] += 1
                shell_idx += block_size

    return irrep_blocks, irrep_shell_map, irrep_map


class VariationalE3TBHamiltonian(TBHamiltonian):

    def __init__(self,
                 base_model: TBHamiltonian | None = None,
                 freeze: bool = False,
                 variational_nn: AtomwiseE3Conv | None = None,
                 onsite_nn: AtomwiseE3Conv | None = None,
                 onsite_weight: float = 1.0,
                 variational_weight: float = 1.0,
                 basis_weight: float = 1.0,
                 variational_nn_kwargs: dict | None = None,
                 onsite_nn_kwargs: dict | None = None,
                 **base_model_kwargs):

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
                **base_model_kwargs
            )
            self._load_base_model_weights(base_model=base_model, freeze=freeze)
        else:
            super().__init__(**base_model_kwargs)

        self.irrep_blocks, self.irrep_shell_map, self.irrep_map = \
            _make_shell_irrep_map(self.orbital_shell_idxs)

        shell_irreps = _make_shell_irreps(
            self.irrep_blocks, 
            n_coeffs=self.basis_size)

        onsite_energy_irreps = _make_shell_irreps(
            self.irrep_blocks, 
            n_coeffs=1)

        n_alpha_params = self.logscale_orbital_alphas[0].flatten().size(-1)
        n_gamma_params = self.logscale_potential_gammas[0].flatten().size(-1)

        variational_out_irreps = (
            shell_irreps +
            _scalar_irreps(n_alpha_params) +
            _scalar_irreps(self.potential_basis_size) + 
            _scalar_irreps(n_gamma_params)
        )
        
        self.onsite_weight = onsite_weight
        self.variational_weight = variational_weight
        self.basis_weight = basis_weight

        if onsite_nn_kwargs is None:
            onsite_nn_kwargs = {}
        if variational_nn_kwargs is None:
            variational_nn_kwargs = {}

        self.onsite_nn = _make_atomwise_e3_nn(
            nn=onsite_nn,
            out_irreps=onsite_energy_irreps,
            in_irreps=None,
            name='onsite_nn',
            nn_kwargs=onsite_nn_kwargs,
        )

        self.variational_nn = _make_atomwise_e3_nn(
            nn=variational_nn,
            out_irreps=variational_out_irreps,
            in_irreps=None,
            name='variational_nn',
            nn_kwargs=variational_nn_kwargs,
        )

    def get_onsite_energies(
            self, 
            atomic_numbers: torch.IntTensor,
            positions: torch.Tensor,
            neighbors: torch.IntTensor,
            unit_cell: torch.IntTensor,
            unit_cell_images: torch.Tensor) -> torch.Tensor:

        # compute neighbor distances
        unit_cell_images = unit_cell_images.to(torch.float)
        neighbor_distances = positions[neighbors] + \
            (unit_cell_images @ unit_cell) - \
            positions.unsqueeze(-2)

        # generate irrep index mapping
        irrep_shell_map = self.irrep_shell_map[atomic_numbers]
        irrep_map = self.irrep_map[atomic_numbers]
        site_idxs, irrep_idxs = torch.nonzero(
            irrep_shell_map >= 0, as_tuple=True)

        # get on-site energy
        onsite_energies = torch.zeros_like(
            irrep_shell_map, dtype=torch.float32)
        shell_energies = torch.sinh(
            self.logscale_onsite_energies[atomic_numbers])
        
        onsite_energies[site_idxs, irrep_idxs] += \
            shell_energies[site_idxs, irrep_shell_map[site_idxs, irrep_idxs]]
        
        # apply variational nn to obtain on-site perturbations
        irrep_perts = self.onsite_weight * self.onsite_nn(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=None)

        onsite_energies[site_idxs, irrep_idxs] += \
            irrep_perts[site_idxs, irrep_map[site_idxs, irrep_idxs]]

        # combine energies and perturbations
        return self.energy_scale * onsite_energies

    def get_basis_parameters(
            self, 
            atomic_numbers: torch.IntTensor,
            positions: torch.Tensor,
            neighbors: torch.IntTensor,
            unit_cell: torch.IntTensor,
            unit_cell_images: torch.Tensor) -> torch.Tensor:
        
        n_alpha_params = self.logscale_orbital_alphas[0].flatten().size(-1)
        n_gamma_params = self.logscale_potential_gammas[0].flatten().size(-1)
        alpha_params_shape = \
            self.logscale_orbital_alphas[atomic_numbers].shape
        gamma_params_shape = \
            self.logscale_potential_gammas[atomic_numbers].shape

        # get alpha & gamma orbital parameters
        logscale_alphas, logscale_gammas = \
            self._get_logscale_basis_width_parameters(
                atomic_numbers=atomic_numbers)

        logscale_alphas_flat = \
            logscale_alphas.view(atomic_numbers.shape[0], -1)
        logscale_gammas_flat = \
            logscale_gammas.view(atomic_numbers.shape[0], -1)
            
        n_c_params = (2*self.irrep_blocks+1).sum() * self.basis_size
        n_v_params = logscale_gammas_flat.shape[-1]

        # compute neighbor distances
        unit_cell_images = unit_cell_images.to(torch.float)
        neighbor_distances = positions[neighbors] + \
            (unit_cell_images @ unit_cell) - \
            positions.unsqueeze(-2)

        # apply variational nn
        nn_features_out = self.variational_nn(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=None)

        # unpack basis coefficient perturbations
        c_perts_flat, alpha_perts_flat, v_perts_flat, gamma_perts_flat = \
            torch.split(
                nn_features_out,
                (n_c_params, n_alpha_params, n_v_params, n_gamma_params),
                dim=1)
        
        # compute perturbed alpha and gammas
        alphas = torch.clamp(
            torch.exp(logscale_alphas) + 
            self.basis_weight * alpha_perts_flat.reshape(alpha_params_shape), 
            min=1e-3)
        gammas = torch.clamp(
            torch.exp(logscale_gammas) + 
            self.basis_weight * gamma_perts_flat.reshape(gamma_params_shape), 
            min=1e-3)

        irrep_shell_map = self.irrep_shell_map[atomic_numbers]
        irrep_map = self.irrep_map[atomic_numbers]
        
        site_idxs, irrep_idxs = torch.nonzero(
            irrep_shell_map >= 0, as_tuple=True)

        # compute normalized c coeffficients
        shell_cs = torch.sinh(self.logscale_orbital_cs[atomic_numbers])
        cs_shape = irrep_shell_map.shape + (self.basis_size,)
        cs_unnormed = torch.zeros(cs_shape, dtype=torch.float32)
        cs_unnormed[site_idxs, irrep_idxs] += \
            shell_cs[site_idxs, irrep_shell_map[site_idxs, irrep_idxs]]
        
        c_perts = c_perts_flat.reshape(
            atomic_numbers.shape[0], -1, self.basis_size)
        cs_unnormed[site_idxs, irrep_idxs] += self.variational_weight * \
            c_perts[site_idxs, irrep_map[site_idxs, irrep_idxs]]

        cs = cs_unnormed * self._get_c_normalization_coeffs(
            atomic_numbers=atomic_numbers,
            alphas=alphas,
            cs=cs_unnormed)

        # compute v coefficients
        logscale_vs = self.logscale_potential_vs[atomic_numbers]
        vs = -self.energy_scale * \
            (torch.exp(logscale_vs) + self.variational_weight *
                v_perts_flat.reshape(logscale_vs.shape))

        return alphas, cs, gammas, vs

    def _get_c_normalization_coeffs(
            self,
            atomic_numbers: torch.IntTensor,
            alphas: torch.Tensor,
            cs: torch.Tensor):

        irrep_shell_map = self.irrep_shell_map[atomic_numbers]
        shell_idxs = self.orbital_shell_idxs[atomic_numbers]

        site_idxs, irrep_idxs = torch.nonzero(
            irrep_shell_map >= 0, as_tuple=True)

        irrep_alpha_idxs = irrep_shell_map[site_idxs, irrep_idxs]
        site_ls = shell_idxs[site_idxs, irrep_alpha_idxs]
        site_alphas = alphas[site_idxs, irrep_alpha_idxs]
        site_cs = cs[site_idxs, irrep_idxs]

        site_c_norms = cho_ng_norm_coeff(
            l=site_ls.unsqueeze(-1),
            alphas=site_alphas.unsqueeze(-2),
            alpha_idxs=torch.zeros_like(site_ls),
            cs=site_cs.unsqueeze(-2)
        )

        c_norm_coeffs = torch.zeros_like(cs, requires_grad=False)
        c_norm_coeffs[site_idxs, irrep_idxs] = 1/torch.sqrt(site_c_norms)
        
        return c_norm_coeffs

    def _get_orbital_basis_idxs(
            self,
            idxs: torch.Tensor,
            atomic_numbers: torch.Tensor) -> tuple:

        block_atoms = atomic_numbers[idxs]
        orbital_shells = self.orbital_shell_idxs[block_atoms]

        block_idxs, block_l_idxs = \
            (orbital_shells >= 0)\
            .nonzero(as_tuple=True)
        block_ls = orbital_shells[block_idxs, block_l_idxs]
        block_l_sizes = 2*block_ls+1

        orbital_shell_size_totals = torch.zeros(
            orbital_shells.shape[:1],
            dtype=orbital_shells.dtype,
            device=orbital_shells.device
        ).put_(block_idxs, block_l_sizes, accumulate=True)

        orbital_shell_totals_idxs = \
            orbital_shell_size_totals.cumsum(dim=0) - \
            orbital_shell_size_totals
        
        orbital_site_idxs = idxs[block_idxs].repeat_interleave(block_l_sizes)

        orbital_idxs = (
            torch.arange(
                orbital_site_idxs.shape[0], 
                device=orbital_site_idxs.device
            ) - orbital_shell_totals_idxs.repeat_interleave(
                orbital_shell_size_totals)
        )

        return orbital_site_idxs, orbital_idxs

    def _get_orbital_image_basis_idxs(
            self,
            idxs: torch.IntTensor,
            image_idxs: torch.IntTensor,
            atomic_numbers: torch.IntTensor) -> tuple:

        block_atoms = atomic_numbers[idxs]
        orbital_shells = self.orbital_shell_idxs[block_atoms]

        block_idxs, block_l_idxs = \
            (orbital_shells >= 0)\
            .nonzero(as_tuple=True)
        block_ls = orbital_shells[block_idxs, block_l_idxs]
        block_l_sizes = 2*block_ls+1

        orbital_shell_size_totals = torch.zeros(
            orbital_shells.shape[:1],
            dtype=orbital_shells.dtype,
            device=orbital_shells.device
        ).put_(block_idxs, block_l_sizes, accumulate=True)

        orbital_shell_totals_idxs = \
            orbital_shell_size_totals.cumsum(dim=0) - \
            orbital_shell_size_totals
        
        orbital_site_idxs = idxs[block_idxs].repeat_interleave(block_l_sizes)
        orbital_image_idxs = image_idxs[orbital_site_idxs]

        orbital_idxs = (
            torch.arange(
                orbital_site_idxs.shape[0], 
                device=orbital_site_idxs.device
            ) - orbital_shell_totals_idxs.repeat_interleave(
                orbital_shell_size_totals)
        )

        return orbital_site_idxs, orbital_image_idxs, orbital_idxs

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
            k_pts = torch.zeros(3, device=positions.device)

        # get orbital parameters:
        alphas, cs, gammas, vs = \
            self.get_basis_parameters(
                atomic_numbers=atomic_numbers, 
                positions=positions, 
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images)

        # get on-site energies:
        onsite_energies = self.get_onsite_energies(
            atomic_numbers=atomic_numbers, 
            positions=positions, 
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images)

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
            
            site_e3_idxs, site_e3_orbital_idxs = \
                self._get_orbital_basis_idxs(site, atomic_numbers)
            site_idxs, site_l_idxs, site_ls = \
                self._get_block_basis_idxs(site, atomic_numbers)
            
            site_data = e3_cho_overlap_matrix_metadata(site_ls, sparse=True)

            H_site_idxs = self._get_neighborhood_hamiltonian_idxs(
                site, H_idxs, H_blocksizes)

            # compute on-site diagonals:
            H_site_diagonal = \
                onsite_energies[site_e3_idxs, site_e3_orbital_idxs]
            H_diagonals.append(H_site_diagonal)

            # retrieve neighborhood data:
            site_neighbors_nz = (neighbors[i] >= 0)
            site_neighbors = neighbors[i][site_neighbors_nz]
            site_neighbor_images = unit_cell_images[i][site_neighbors_nz]

            # handle if neighborhood is empty
            if site_neighbors.shape[0] == 0:
                H_components.append(
                    torch.zeros(
                        (0), 
                        device=H_site_diagonal.device,
                        dtype=H_site_diagonal.dtype))
                H_component_idxs.append(
                    torch.zeros(
                        (0, 2),
                        dtype=site_neighbors.dtype,
                        device=site_neighbors.device))
                H_component_distances.append(
                    torch.zeros(
                        (0, 3),
                        dtype=positions.dtype,
                        device=positions.device))
            else:
                # otherwise, compute neighborhood overlap matrices
                neighbor_e3_idxs, _, neighbor_e3_orbital_idxs = \
                    self._get_orbital_image_basis_idxs(
                        site_neighbors, 
                        site_neighbor_images, 
                        atomic_numbers)

                neighbor_idxs, neighbor_images, \
                    neighbor_l_idxs, neighbor_ls = \
                    self._get_block_image_basis_idxs(
                        site_neighbors, 
                        site_neighbor_images, 
                        atomic_numbers)

                neighbor_data = e3_cho_overlap_matrix_metadata(
                    neighbor_ls, sparse=True)

                H_neighbor_idxs = self._get_neighborhood_hamiltonian_idxs(
                    site_neighbors, H_idxs, H_blocksizes)

                # compute distances to neighbors across unit cell boundaries
                neighbor_positions = \
                    positions[neighbor_idxs] + \
                    neighbor_images @ unit_cell

                # compute neighborhood Hamiltonian matrix
                H_neighborhood = e3_cho_ng_potential_overlap_matrix(
                    mu_1=positions[site_idxs],
                    l_1=site_ls,
                    l_alphas_1=alphas[site_idxs, site_l_idxs],
                    e3_cs_1=cs[site_e3_idxs, site_e3_orbital_idxs],
                    gammas_1=gammas[i],
                    vs_1=vs[i],
                    mu_2=neighbor_positions,
                    l_2=neighbor_ls,
                    l_alphas_2=alphas[neighbor_idxs, neighbor_l_idxs],
                    e3_cs_2=cs[neighbor_e3_idxs, neighbor_e3_orbital_idxs],
                    e3_cho_data_1=site_data,
                    e3_cho_data_2=neighbor_data)

                H_neighborhood_dxs = \
                    self._get_neighborhood_hamiltonian_distances(
                        neighbor_positions - positions[i],
                        neighbor_ls
                    ).view(1, -1, 3).expand(*H_neighborhood.shape, 3)

                H_neighborhood_idxs = torch.cartesian_prod(
                    H_site_idxs, H_neighbor_idxs)

                H_components.append(H_neighborhood.flatten())
                H_component_idxs.append(H_neighborhood_idxs)
                H_component_distances.append(H_neighborhood_dxs.reshape(-1, 3))

        return self._build_bloch_hamiltonian(
            H_diagonals, H_components, H_component_idxs,
            H_component_distances, k_pts,
            shape=(H_size, H_size), sparse=sparse)