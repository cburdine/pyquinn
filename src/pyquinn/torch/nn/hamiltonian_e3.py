from __future__ import annotations

import torch
import torch.nn as nn

import e3nn.o3 as o3

from pyquinn.torch.nn.hamiltonian import Hamiltonian
from pyquinn.torch.nn.eigensolver import Eigensolver
from pyquinn.torch.nn.atomwise_e3 import AtomwiseE3Conv
from pyquinn.torch.nn.lcao.gto import _ELEMENT_CHO_VALENCE_SHELLS, GTO_L_MAX
from pyquinn.torch.types import TorchOperator
from pyquinn.torch.types.operator import build_matrix, build_nonperiodic_matrix


def _make_shell_idxs(shells: dict[int, list[int]]) -> torch.IntTensor:
    """Construct a padded tensor mapping `[atomic_number, shell_idx] -> l`.

    Unused entries are `-1` and should be ignored.
    """
    if not shells:
        raise ValueError('Expected non-empty `shells`.')

    z_max = max(shells.keys())
    n_shells_max = max(len(v) for v in shells.values())

    shell_idxs = torch.full((z_max + 1, n_shells_max), -1, dtype=torch.int64)
    for z, l_list in shells.items():
        if z <= 0:
            raise ValueError(f'Expected positive atomic number in `shells`, got: {z}')
        for i, l in enumerate(l_list):
            if l < 0 or l > GTO_L_MAX:
                raise ValueError(
                    f'Expected 0 <= l <= {GTO_L_MAX} for element {z} in shell {i}, got: l={l}'
                )
            shell_idxs[z, i] = int(l)

    return shell_idxs


def _make_shell_irreps(unique_shells: torch.IntTensor) -> o3.Irreps:
    """Build an Irreps representing a concatenated list of orbital shells."""
    irreps = o3.Irreps()
    for l in unique_shells:
        _l = int(l.item())
        irreps += o3.Irreps([(1, (_l, (-1) ** _l))])
    return irreps


def _make_shell_irrep_map(orbital_shell_idxs: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
    """Build a mapping from element-specific orbital order -> unique shell irrep order.

    Returns:
        irrep_blocks: (S,) tensor with the l value for each unique shell instance.
        irrep_map: (Z, M) tensor mapping local orbital component indices to unique-shell indices.
            `-1` entries should be ignored.
    """
    l_max = int(torch.max(orbital_shell_idxs).item())
    if l_max < 0:
        raise ValueError('Expected at least one orbital shell with l >= 0.')

    l_unique_shell_counts = torch.zeros(l_max + 1, dtype=torch.int32)
    max_shell_block_size = torch.tensor(0, dtype=torch.int32)

    # Determine max number of copies of each l needed across elements.
    for l_arr in orbital_shell_idxs:
        l_counts = torch.zeros(l_max + 1, dtype=torch.int32)
        ls, counts = torch.unique(l_arr, return_counts=True)
        valid = ls >= 0
        ls, counts = ls[valid], counts[valid]
        if ls.numel() > 0:
            l_counts[ls.to(torch.int64)] += counts.to(torch.int32)
            l_unique_shell_counts = torch.maximum(l_unique_shell_counts, l_counts)

            block_size = torch.sum((2 * ls + 1) * counts)
            max_shell_block_size = torch.maximum(max_shell_block_size, block_size.to(torch.int32))

    l_unique_idxs = torch.cumsum(l_unique_shell_counts, -1) - l_unique_shell_counts
    irrep_blocks = torch.zeros(int(l_unique_shell_counts.sum().item()), dtype=torch.int32)
    for l, count in enumerate(l_unique_shell_counts.tolist()):
        if count <= 0:
            continue
        idx = int(l_unique_idxs[l].item())
        irrep_blocks[idx:idx + count] = l

    irrep_block_sizes = l_unique_shell_counts * (2 * torch.arange(l_max + 1, dtype=torch.int32) + 1)
    irrep_block_idxs = torch.cumsum(irrep_block_sizes, -1) - irrep_block_sizes

    z_dim = orbital_shell_idxs.shape[0]
    max_shell_block_size_int = int(max_shell_block_size.item())
    irrep_map = torch.full((z_dim, max_shell_block_size_int), -1, dtype=torch.int64)

    # Map each element's local orbital component index -> unique-shell component index.
    for z, l_arr in enumerate(orbital_shell_idxs):
        l_shell_counts = torch.zeros(l_max + 1, dtype=torch.int32)
        shell_idx = 0
        for l in l_arr.tolist():
            if l < 0:
                continue
            block_size = 2 * l + 1
            base = int(irrep_block_idxs[l].item()) + block_size * int(l_shell_counts[l].item())
            irrep_idxs = torch.arange(block_size, dtype=torch.int64) + base
            irrep_map[z, shell_idx:shell_idx + block_size] = irrep_idxs
            l_shell_counts[l] += 1
            shell_idx += block_size

    return irrep_blocks.to(torch.int64), irrep_map


def _neighbor_distances(
    positions: torch.Tensor,
    neighbors: torch.IntTensor,
    unit_cell: torch.Tensor | None = None,
    unit_cell_images: torch.IntTensor | None = None,
) -> torch.Tensor:
    """Compute neighbor displacement vectors `r_j - r_i` for a -1 padded neighbor list.

    If `unit_cell_images` is provided, displacements are computed periodically as:

        dx = (r_j + image @ unit_cell) - r_i
    """
    neighbors_safe = neighbors.clamp_min(0)
    dx = positions[neighbors_safe] - positions.unsqueeze(-2)

    if unit_cell_images is not None:
        if unit_cell is None:
            raise ValueError('`unit_cell_images` requires `unit_cell`.')
        shift = unit_cell_images.to(dtype=positions.dtype, device=positions.device) @ unit_cell.to(
            dtype=positions.dtype, device=positions.device
        )
        dx = dx + shift

    invalid = neighbors < 0
    if invalid.any():
        dx = dx.masked_fill(invalid.unsqueeze(-1), 0.0)
    return dx


class E3Hamiltonian(Hamiltonian):
    """Equivariant Hamiltonian model built from a rank-1 outer-product expansion.

    This module predicts a real symmetric Hamiltonian in a (real) cubic-harmonic
    orbital basis using an E(3)-equivariant atomwise network.

    The Hamiltonian is parameterized as:

        H = energy_scale * sum_r w_r * v_r v_r^T

    where each v_r transforms under global rotations/reflections in the same
    representation as the orbital basis. This guarantees equivariance of H:

        H(Rx) = D(R) H(x) D(R)^T

    Note: e3nn uses an `xyz` real basis for `l=1` (p orbitals). If you need a
    different orbital ordering or basis convention, adjust the basis mapping.
    """

    SPIN_OPTIONS = ['paired']

    def __init__(
        self,
        eigensolver: str | Eigensolver = 'torch_eigh',
        spin: str = 'paired',
        energy_scale: float = 1.0,
        orbital_shells: dict[int, list[int]] | None = None,
        rank: int = 8,
        equivariant_nn: AtomwiseE3Conv | None = None,
        equivariant_nn_kwargs: dict | None = None,
        z_max: int = 118,
    ) -> None:
        super().__init__(eigensolver=eigensolver)

        if spin not in self.SPIN_OPTIONS:
            raise ValueError(f'Spin must be one of {self.SPIN_OPTIONS}, got: {spin}')
        if rank <= 0:
            raise ValueError('`rank` must be a positive integer.')

        self.spin = spin
        self.energy_scale = float(energy_scale)
        self.rank = int(rank)

        # Fill in unspecified elements from the default valence shells.
        shells = {int(z): list(v) for z, v in _ELEMENT_CHO_VALENCE_SHELLS.items()}
        if orbital_shells is not None:
            for z, l_list in orbital_shells.items():
                shells[int(z)] = list(l_list)

        shell_idxs = _make_shell_idxs(shells)
        if shell_idxs.shape[0] < (z_max + 1):
            pad = torch.full((z_max + 1 - shell_idxs.shape[0], shell_idxs.shape[1]), -1, dtype=shell_idxs.dtype)
            shell_idxs = torch.cat((shell_idxs, pad), dim=0)
        elif shell_idxs.shape[0] > (z_max + 1):
            shell_idxs = shell_idxs[: z_max + 1]

        self.register_buffer('orbital_shell_idxs', shell_idxs)

        irrep_blocks, irrep_map = _make_shell_irrep_map(shell_idxs)
        self.register_buffer('irrep_blocks', irrep_blocks)
        self.register_buffer('irrep_map', irrep_map)

        self.unique_shell_irreps = _make_shell_irreps(self.irrep_blocks)
        self.unique_shell_dim = self.unique_shell_irreps.dim

        out_irreps = self.unique_shell_irreps * self.rank

        equivariant_nn_kwargs = {} if equivariant_nn_kwargs is None else dict(equivariant_nn_kwargs)

        if equivariant_nn is None:
            self.equivariant_nn = AtomwiseE3Conv(
                out_features=out_irreps,
                in_features=None,
                z_max=z_max,
                **equivariant_nn_kwargs,
            )
        elif isinstance(equivariant_nn, AtomwiseE3Conv):
            if equivariant_nn.out_irreps.dim != out_irreps.dim:
                raise ValueError(
                    'Incompatible `equivariant_nn.out_irreps`: '
                    f'expected dim {out_irreps.dim}, got {equivariant_nn.out_irreps}.'
                )
            self.equivariant_nn = equivariant_nn
        else:
            raise ValueError(f'Unexpected type for `equivariant_nn` (got: {type(equivariant_nn)}).')

        # Learnable scalar weights for each outer-product term.
        self.rank_weights = nn.Parameter(torch.randn(self.rank) * 0.05)

    def _orbital_counts(self, atomic_numbers: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
        l_values_padded = self.orbital_shell_idxs[atomic_numbers]
        l_site_idxs, l_shell_idxs = torch.nonzero(l_values_padded >= 0, as_tuple=True)
        ls = l_values_padded[l_site_idxs, l_shell_idxs]

        n_orbitals = torch.zeros_like(atomic_numbers, dtype=torch.int64)
        n_orbitals.put_(l_site_idxs, (2 * ls + 1).to(torch.int64), accumulate=True)

        offsets = torch.cumsum(n_orbitals, dim=0) - n_orbitals
        return n_orbitals, offsets

    def orbital_indices(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> tuple[torch.IntTensor, torch.IntTensor]:
        n_orbitals, _offsets = self._orbital_counts(atomic_numbers)

        site_idxs, orbital_idxs = torch.nonzero(
            torch.arange(int(torch.max(n_orbitals).item()), device=atomic_numbers.device).view(1, -1)
            < n_orbitals.view(-1, 1),
            as_tuple=True,
        )

        return site_idxs.to(torch.int64), orbital_idxs.to(torch.int64)

    def orbital_names(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> list[str]:
        # Minimal labels; ordering follows the per-site shell order in `orbital_shell_idxs`.
        # NOTE: e3nn's `l=1` basis is `x,y,z` (commonly associated with p_x, p_y, p_z).
        names: list[str] = []

        l_values_padded = self.orbital_shell_idxs[atomic_numbers]
        for l_arr in l_values_padded:
            for l in l_arr.tolist():
                if l < 0:
                    continue
                if l == 0:
                    names.append('s')
                elif l == 1:
                    names.extend(['p_x', 'p_y', 'p_z'])
                else:
                    names.extend([f'l{l}_{i}' for i in range(2 * l + 1)])

        return names

    def orbital_quantum_numbers(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        # Returns (l, m) with a conventional m=-l..l ordering per shell.
        l_values_padded = self.orbital_shell_idxs[atomic_numbers]
        l_site_idxs, l_shell_idxs = torch.nonzero(l_values_padded >= 0, as_tuple=True)
        ls = l_values_padded[l_site_idxs, l_shell_idxs]
        block_sizes = (2 * ls + 1).to(torch.int64)

        quantum_ls = ls.repeat_interleave(block_sizes)
        offsets = torch.cumsum(block_sizes, dim=0) - block_sizes

        quantum_ms = (
            torch.arange(quantum_ls.shape[0], device=quantum_ls.device, dtype=torch.int64)
            - torch.repeat_interleave(offsets, block_sizes)
            - quantum_ls.to(torch.int64)
        )

        return torch.stack((quantum_ls.to(torch.int64), quantum_ms), dim=-1)

    def _predict_orbital_rank_vectors(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        features: torch.Tensor | None = None,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
    ) -> torch.Tensor:
        """Return v with shape `[rank, n_orbitals_total]` in orbital basis order."""
        neighbor_distances = _neighbor_distances(
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )

        per_atom = self.equivariant_nn(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=features,
        )

        n_sites = atomic_numbers.shape[0]
        per_atom = per_atom.view(n_sites, self.rank, self.unique_shell_dim)

        site_idxs, orbital_idxs = self.orbital_indices(
            atomic_numbers=atomic_numbers, positions=positions, neighbors=neighbors
        )

        unique_idxs = self.irrep_map[atomic_numbers][site_idxs, orbital_idxs]
        if (unique_idxs < 0).any():
            raise RuntimeError('Encountered invalid orbital->irrep map index while building orbital features.')

        gathered = per_atom[site_idxs]  # [H, rank, unique_dim]
        v = torch.gather(
            gathered,
            dim=2,
            index=unique_idxs.view(-1, 1, 1).expand(-1, self.rank, 1),
        ).squeeze(-1)  # [H, rank]

        return v.transpose(0, 1).contiguous()  # [rank, H]

    def _coupling_pairs(
        self,
        atomic_numbers: torch.IntTensor,
        neighbors: torch.IntTensor,
        include_onsite_offdiag: bool = True,
    ) -> torch.IntTensor:
        """Return undirected orbital coupling pairs (a,b) with a != b.

        The returned pairs do not include the symmetric counterpart; callers
        can add it by concatenating `pairs.flip(1)`.
        """
        n_orbitals, offsets = self._orbital_counts(atomic_numbers)
        device = atomic_numbers.device

        coupling_pairs: list[torch.IntTensor] = []

        if include_onsite_offdiag:
            for i in range(int(atomic_numbers.shape[0])):
                n_i = int(n_orbitals[i].item())
                if n_i <= 1:
                    continue
                idx_i = torch.arange(n_i, device=device, dtype=torch.int64) + offsets[i]
                coupling_pairs.append(torch.combinations(idx_i, r=2))

        # Unique undirected site pairs from neighbor list.
        site_i, adj = torch.nonzero(neighbors >= 0, as_tuple=True)
        if site_i.numel() > 0:
            site_j = neighbors[site_i, adj].to(torch.int64)
            a = torch.minimum(site_i.to(torch.int64), site_j)
            b = torch.maximum(site_i.to(torch.int64), site_j)
            site_pairs = torch.unique(torch.stack((a, b), dim=1), dim=0)
            site_pairs = site_pairs[site_pairs[:, 0] != site_pairs[:, 1]]

            for i, j in site_pairs.tolist():
                n_i = int(n_orbitals[i].item())
                n_j = int(n_orbitals[j].item())
                if n_i == 0 or n_j == 0:
                    continue
                idx_i = torch.arange(n_i, device=device, dtype=torch.int64) + offsets[i]
                idx_j = torch.arange(n_j, device=device, dtype=torch.int64) + offsets[j]
                coupling_pairs.append(torch.cartesian_prod(idx_i, idx_j))

        if not coupling_pairs:
            return torch.empty((0, 2), dtype=torch.int64, device=device)

        return torch.cat(coupling_pairs, dim=0)

    def hamiltonian_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        features: torch.Tensor | None = None,
        include_onsite_offdiag: bool = True,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.IntTensor, torch.Tensor]:
        v = self._predict_orbital_rank_vectors(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            features=features,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
        )  # [rank, H]

        w = self.rank_weights.to(dtype=v.dtype)
        onsite_values = (v.square() * w.view(-1, 1)).sum(dim=0) * self.energy_scale

        couplings_u = self._coupling_pairs(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            include_onsite_offdiag=include_onsite_offdiag,
        )

        if couplings_u.numel() == 0:
            return onsite_values, couplings_u, torch.empty((0,), dtype=onsite_values.dtype, device=onsite_values.device)

        v_a = v[:, couplings_u[:, 0]]
        v_b = v[:, couplings_u[:, 1]]
        values_u = (v_a * v_b * w.view(-1, 1)).sum(dim=0) * self.energy_scale

        couplings = torch.cat((couplings_u, couplings_u.flip(1)), dim=0)
        coupling_values = torch.cat((values_u, values_u), dim=0)

        return onsite_values, couplings, coupling_values

    def hamiltonian_operator(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        features: torch.Tensor | None = None,
        include_onsite_offdiag: bool = True,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        **kwargs,
    ) -> TorchOperator:
        onsite_values, couplings, coupling_values = self.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            features=features,
            include_onsite_offdiag=include_onsite_offdiag,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            **kwargs,
        )

        site_indices, orbital_indices = self.orbital_indices(
            atomic_numbers=atomic_numbers, positions=positions, neighbors=neighbors
        )

        images = torch.zeros((couplings.shape[0], 3), device=couplings.device, dtype=torch.int64)
        if unit_cell_images is not None:
            unit_cell_images = unit_cell_images.to(device=couplings.device, dtype=torch.int64)
            neighbors = neighbors.to(device=couplings.device, dtype=torch.int64)
            images = self._coupling_images(
                couplings=couplings,
                site_indices=site_indices,
                neighbors=neighbors,
                unit_cell_images=unit_cell_images,
            )

        return TorchOperator(
            onsite_values=onsite_values,
            couplings=couplings.to(torch.int64),
            coupling_values=coupling_values,
            site_indices=site_indices.to(torch.int64),
            orbital_indices=orbital_indices.to(torch.int64),
            images=images,
        )

    @staticmethod
    def _coupling_images(
        couplings: torch.IntTensor,
        site_indices: torch.IntTensor,
        neighbors: torch.IntTensor,
        unit_cell_images: torch.IntTensor,
    ) -> torch.IntTensor:
        """Resolve per-coupling unit-cell images for a Bloch Hamiltonian.

        Assumes `couplings` are ordered as `[couplings_u; couplings_u.flip(1)]`,
        so the second half is the Hermitian counterpart and uses `-images_u`.
        """
        if couplings.ndim != 2 or couplings.shape[1] != 2:
            raise ValueError(f'Expected `couplings` to have shape [E,2], got: {tuple(couplings.shape)}')
        if couplings.shape[0] % 2 != 0:
            raise ValueError(f'Expected even number of couplings, got: E={int(couplings.shape[0])}')
        if unit_cell_images.ndim != 3 or unit_cell_images.shape[-1] != 3:
            raise ValueError(
                'Expected `unit_cell_images` to have shape [N_sites, N_neighbors, 3], '
                f'got: {tuple(unit_cell_images.shape)}'
            )
        if neighbors.shape != unit_cell_images.shape[:2]:
            raise ValueError(
                'Expected `neighbors` and `unit_cell_images` to agree in the first two dims '
                f'(got: {tuple(neighbors.shape)} vs {tuple(unit_cell_images.shape[:2])})'
            )

        device = couplings.device
        half = couplings.shape[0] // 2

        couplings_u = couplings[:half]
        site_a = site_indices[couplings_u[:, 0]].to(torch.int64)
        site_b = site_indices[couplings_u[:, 1]].to(torch.int64)

        images_u = torch.zeros((half, 3), dtype=torch.int64, device=device)
        offsite = site_a != site_b
        if offsite.any():
            pairs = torch.stack((site_a[offsite], site_b[offsite]), dim=1)
            unique_pairs, inv = torch.unique(pairs, dim=0, return_inverse=True)

            src = unique_pairs[:, 0]
            dst = unique_pairs[:, 1]

            # Prefer src->dst match; fall back to -(dst->src) if needed.
            nbrs_src = neighbors[src]
            match_src = nbrs_src == dst.unsqueeze(1)
            has_src = match_src.any(dim=1)
            slot_src = match_src.to(torch.int64).argmax(dim=1)
            img_src = unit_cell_images[src, slot_src].to(torch.int64)

            nbrs_dst = neighbors[dst]
            match_dst = nbrs_dst == src.unsqueeze(1)
            has_dst = match_dst.any(dim=1)
            slot_dst = match_dst.to(torch.int64).argmax(dim=1)
            img_dst = (-unit_cell_images[dst, slot_dst]).to(torch.int64)

            has_any = has_src | has_dst
            if not bool(has_any.all().item()):
                bad = torch.nonzero(~has_any, as_tuple=False).view(-1)
                bad_pairs = unique_pairs[bad].detach().cpu().tolist()
                raise ValueError(
                    'Could not resolve `unit_cell_images` for some neighbor pairs. '
                    f'Missing pairs (src,dst): {bad_pairs}'
                )

            if bool((has_src & has_dst).any().item()):
                both = has_src & has_dst
                if not bool((img_src[both] == img_dst[both]).all().item()):
                    raise ValueError(
                        'Inconsistent `unit_cell_images` for some neighbor pairs: '
                        'src->dst image does not equal -(dst->src) image.'
                    )

            img = torch.where(has_src.view(-1, 1), img_src, img_dst)
            images_u[offsite] = img[inv]

        images = torch.cat((images_u, -images_u), dim=0)
        return images

    def hamiltonian_matrix(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        features: torch.Tensor | None = None,
        include_onsite_offdiag: bool = True,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        k_pts: torch.Tensor | None = None,
        sparse: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        onsite_values, couplings, coupling_values = self.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            features=features,
            include_onsite_offdiag=include_onsite_offdiag,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            **kwargs,
        )

        periodic = (unit_cell is not None) or (unit_cell_images is not None) or (k_pts is not None)
        if not periodic:
            return build_nonperiodic_matrix(
                onsite_values=onsite_values,
                couplings=couplings,
                coupling_values=coupling_values,
            )

        if unit_cell is None or unit_cell_images is None:
            raise ValueError('Periodic `hamiltonian_matrix` requires both `unit_cell` and `unit_cell_images`.')

        if k_pts is None:
            k_pts = torch.zeros((1, 3), device=positions.device, dtype=positions.dtype)
        elif not isinstance(k_pts, torch.Tensor):
            k_pts = torch.as_tensor(k_pts, device=positions.device, dtype=positions.dtype)
        else:
            k_pts = k_pts.to(device=positions.device, dtype=positions.dtype)

        frac_positions = positions @ torch.linalg.pinv(unit_cell.to(device=positions.device, dtype=positions.dtype))

        site_indices, _orbital_indices = self.orbital_indices(
            atomic_numbers=atomic_numbers, positions=positions, neighbors=neighbors
        )

        images = self._coupling_images(
            couplings=couplings.to(torch.int64),
            site_indices=site_indices.to(torch.int64),
            neighbors=neighbors.to(device=positions.device, dtype=torch.int64),
            unit_cell_images=unit_cell_images.to(device=positions.device, dtype=torch.int64),
        )

        return build_matrix(
            onsite_values=onsite_values,
            couplings=couplings.to(torch.int64),
            coupling_values=coupling_values,
            site_indices=site_indices.to(torch.int64),
            images=images,
            k_pts=k_pts,
            frac_positions=frac_positions,
            sparse=sparse,
        )
