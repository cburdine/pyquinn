import e3nn.o3 as o3
import torch.nn as nn
import e3nn
import torch
from torch_geometric.utils import scatter
import numpy as np

from pyquinn.torch.atomwise import (
    AtomwiseConv, 
    _get_neighbor_distance_edgelist
)
from pyquinn.torch.util import get_activation_fn
from pyquinn.torch.lcao.gto import _ELEMENT_CHO_VALENCE_SHELLS


def _scalar_irreps(n: int | None):
    return o3.Irreps() if n is None \
        else o3.Irreps([(n, (0, 1))])


def _hidden_irreps(n: int | None, l_max: int = 2):
    if n is None:
        n = 1
    return o3.Irreps([(n, (l, (-1)**l)) for l in range(l_max+1)])


def _make_irrep_change_of_basis(o3_basis: str = 'xyz'):

    o3_matrix = torch.zeros((3, 3))
    o3_eye = torch.eye(3)

    if not isinstance(o3_basis, str) or len(o3_basis) > 3:
        raise ValueError(
            f'Expected o3_basis to be a string of length 3; '
            f'got: "{o3_basis}."'
        )

    for i, ch in enumerate(o3_basis):
        ax = ch.lower()
        if ax not in 'xyz':
            raise ValueError(
                f'Unknown axis "{ax}" in basis string "{o3_basis}"; '
                f'expected "x", "y", or "z"')
        o3_matrix[:, i] = o3_eye['xyz'.index(ch)]

    return o3_matrix


# class BiasedFullyConnectedNet(nn.Module):
#     def __init__(self, hs, act=None, 
#                  variance_in: int = 1, 
#                  variance_out: int = 1, 
#                  out_act: bool = False, 
#                  bias=True):
#         super().__init__()
#         self.fcn = e3nn.nn.FullyConnectedNet(
#             hs, act=act, 
#             variance_in=variance_in, variance_out=variance_out, 
#             out_act=out_act)
#         self.bias = nn.Parameter(torch.zeros(hs[-1])) if bias else None

#     def forward(self, x):
#         x = self.fcn(x)

#         if self.bias is not None:
#             x += self.bias
#         return x


@AtomwiseConv.register('e3conv')
class AtomwiseE3Conv(AtomwiseConv):

    def __init__(
            self,
            out_features: int | o3.Irreps,
            in_features: int | o3.Irreps | None = None,
            hidden_features: int | o3.Irreps | None = None,
            layers: int = 1,
            activation: str = 'leakyrelu',
            o3_basis: str = 'xyz',
            l_max: int = 2,
            r_max: float = 5.0,
            r_points: int = 32,
            z_max: int = 118):

        if not isinstance(in_features, o3.Irreps):
            in_features = _scalar_irreps(in_features)

        if not isinstance(out_features, o3.Irreps):
            out_features = _scalar_irreps(out_features)

        if not isinstance(hidden_features, o3.Irreps):
            hidden_features = _hidden_irreps(hidden_features, l_max=l_max)

        super().__init__(
            out_features=out_features.dim,
            in_features=in_features.dim,
            z_max=z_max)

        self.in_irreps = in_features
        self.out_irreps = out_features
        self.hidden_irreps = hidden_features
        self.atom_irreps = o3.Irreps([(z_max+1, (0, 1))])
        self.neighbor_irreps = o3.Irreps.spherical_harmonics(lmax=l_max)

        self.r_max = r_max
        self.r_points = r_points

        self.D_output_transform = self.out_irreps.D_from_matrix(
            _make_irrep_change_of_basis(o3_basis))

        self.D_input_transform = self.in_irreps.D_from_matrix(
            _make_irrep_change_of_basis(o3_basis)) \
            if self.in_irreps else None

        self.activation = get_activation_fn(activation)()

        layer_out_irreps = self.hidden_irreps \
            if layers > 1 else self.out_irreps
        
        self.irrep_tps = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                (self.atom_irreps + self.in_irreps), 
                self.neighbor_irreps, layer_out_irreps,
                shared_weights=False)
        ])

        for i in range(layers-1):
            layer_in_irreps = self.hidden_irreps
            layer_out_irreps = self.hidden_irreps if i < (layers - 2) \
                else self.out_irreps
            self.irrep_tps.append(
                o3.FullyConnectedTensorProduct(
                    layer_in_irreps, self.neighbor_irreps, 
                    layer_out_irreps,
                    shared_weights=False))
        
        self.edge_nns = nn.ModuleList([
            e3nn.nn.FullyConnectedNet(
                [r_points, r_points, tp.weight_numel], 
                self.activation)
            for tp in self.irrep_tps
        ])

    def _get_atom_features(
            self, 
            atomic_numbers, 
            features):
        
        # build atomic "one-hot" encoded features
        atom_features = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=self.z_max+1
        ).to(torch.float32)

        if features is not None and self.D_input_transform is not None:
            transformed_features = features @ self.D_input_transform.T           
            atom_features = torch.cat(
                (atom_features, transformed_features), dim=-1)

        return atom_features

    def forward(
            self,
            atomic_numbers,
            neighbors,
            neighbor_distances,
            features,
            **kwargs):

        # generate edge list between neighboring atoms (the "bond pairs")
        bond_pairs, bond_distances = \
            _get_neighbor_distance_edgelist(
                neighbors=neighbors, distances=neighbor_distances)

        atom_neighbors = torch.sum(neighbors >= 0, dim=-1, keepdim=True)
        
        # combine features into a feature tensor
        f = self._get_atom_features(atomic_numbers, features)

        # get the "bond features" from the atomic distances
        bond_features = e3nn.math.soft_one_hot_linspace(
            bond_distances.norm(dim=1),
            start=0.0,
            end=self.r_max,
            number=self.r_points,
            basis='smooth_finite',
            cutoff=True,
        )

        sh = o3.spherical_harmonics(
            self.neighbor_irreps, 
            bond_distances,
            normalize=True, 
            normalization='component')

        for tp, edge_nn in zip(self.irrep_tps, self.edge_nns):

            tp_weights = edge_nn(bond_features)
            f = scatter(
                tp(f[bond_pairs[0]], sh, tp_weights),
                bond_pairs[1], 
                dim=0,
                dim_size=atomic_numbers.shape[0]).div(atom_neighbors**0.5)

        f_out = f @ self.D_output_transform

        return f_out


class AtomicOrbitalE3Conv(AtomwiseE3Conv):

    @staticmethod
    def _make_orbital_shells_irrep_map(
            orbital_shells: dict, 
            z_max: int = 118) -> torch.tensor:

        # determine how many instances of each irrep (shell) are needed
        l_max = max(np.max(l_arr) for l_arr in orbital_shells.keys())
        l_unique_shell_counts = np.zeros(l_max+1, dtype=np.int64)
        shell_l_counts = {}

        for z, l_arr in orbital_shells.items():
            if z <= z_max:
                l_counts = np.zeros(l_max+1, dtype=np.int64)
                ls, counts = np.unique_counts(l_arr)
                ls_nz = (ls >= 0)
                ls, counts = ls[ls_nz], counts[ls_nz]
                l_counts[ls] += counts
                shell_l_counts[z] = l_counts
                l_unique_shell_counts = \
                    np.maximum(l_unique_shell_counts, l_counts)

        # determine the minimal set of unique shells for building irreps
        l_unique_idxs = np.cumsum(l_unique_shell_counts) - \
            l_unique_shell_counts
        l_unique_shells = np.zeros(l_unique_shell_counts.sum(), dtype=np.int64)
        for l, count in enumerate(l_unique_shell_counts):
            idx = l_unique_idxs[l]
            l_unique_shells[idx:idx+count] = l
        
        # populate unique shell map, which maps unique irreps -> shells
        # (shells with an index of -1 are ignored)
        l_unique_shell_map = np.full((z_max + 1, l_unique_shells.shape[0]), -1)
        for z, l_arr in orbital_shells.items():
            if z <= z_max:
                l_counts = np.zeros(l_max+1, dtype=np.int64)
                for l in l_arr:
                    if l >= 0:
                        l_unique_shell_map[z, l_unique_idxs[l]+l_counts[l]]
                        l_counts[l] += 1
        
        l_unique_shells = torch.tensor(l_unique_shells, dtype=torch.int64)
        l_unique_shell_map = torch.tensor(
            l_unique_shell_map, dtype=torch.int64)

        return l_unique_shells, l_unique_shell_map

    def __init__(
            self,
            orbital_shells: dict | None = None,
            out_orbital_features: int = 1,
            in_features: o3.Irreps | None = None,
            hidden_orbital_features: int = 16,
            layers: int = 1,
            activation: str = 'leakyrelu',
            o3_basis: str = 'xyz',
            l_max: int = 2,
            r_max: float = 5.0,
            r_points: int = 32,
            z_max: int = 118):

        if not isinstance(in_features, o3.Irreps):
            in_features = _scalar_irreps(in_features)
        self.site_feature_irreps = in_features

        # use default valence shells to start and override with supplied values
        orbital_shells = {} if orbital_shells is None else orbital_shells
        orbital_shells = _ELEMENT_CHO_VALENCE_SHELLS | orbital_shells

        self.unique_shells, self.unique_shells_map = \
            AtomicOrbitalE3Conv._make_orbital_shells_irrep_map(
                orbital_shells=orbital_shells,
                z_max=z_max)

        self.in_orbital_irreps = o3.Irreps([
            (1, (int(l), (-1)**int(l))) for l in self.unique_shells
        ])

        self.out_orbital_irreps = o3.Irreps([
            (out_orbital_features, (int(l), (-1)**int(l)))
            for l in self.unique_shells
        ])

        self.hidden_orbital_irreps = o3.Irreps([
            (hidden_orbital_features, (int(l), (-1)**int(l)))
            for l in self.unique_shells
        ])

        super().__init__(
            out_features=self.out_orbital_irreps,
            in_features=self.site_feature_irreps+self.in_orbital_irreps,
            hidden_features=self.hidden_orbital_irreps,
            layers=layers,
            activation=activation,
            o3_basis=o3_basis,
            l_max=l_max,
            r_max=r_max,
            r_points=r_points,
            z_max=z_max
        )

    def forward(
            self,
            atomic_numbers,
            neighbors,
            neighbor_distances,
            features,
            **kwargs):

        # generate edge list between neighboring atoms (the "bond pairs")
        bond_pairs, bond_distances = \
            _get_neighbor_distance_edgelist(
                neighbors=neighbors, distances=neighbor_distances)

        atom_neighbors = torch.sum(neighbors >= 0, dim=-1, keepdim=True)
        
        # combine features into a feature tensor
        atom_features = self._get_atom_features(atomic_numbers, features)

        f = torch.cat(atom_features)

        # get the "bond features" from the atomic distances
        bond_features = e3nn.math.soft_one_hot_linspace(
            bond_distances.norm(dim=1),
            start=0.0,
            end=self.r_max,
            number=self.r_points,
            basis='smooth_finite',
            cutoff=True,
        )

        sh = o3.spherical_harmonics(
            self.neighbor_irreps, 
            bond_distances,
            normalize=True, 
            normalization='component')

        for tp, edge_nn in zip(self.irrep_tps, self.edge_nns):
            tp_weights = edge_nn(bond_features)
            f = scatter(
                tp(f[bond_pairs[0]], sh, tp_weights),
                bond_pairs[1],
                dim=0,
                dim_size=atomic_numbers.shape[0]).div(atom_neighbors**0.5)

        f_out = f @ self.D_output_transform

        return f_out

        

