import e3nn
import e3nn.nn
import e3nn.o3 as o3
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from pyquinn.torch.nn.atomwise import (
    AtomwiseConv,
    _get_neighbor_distance_edgelist,
)
from pyquinn.torch.util import get_activation_fn


def _scalar_irreps(n: int | None):
    return o3.Irreps() if n is None else o3.Irreps([(n, (0, 1))])


def _hidden_irreps(n: int | None, l_max: int = 2):
    if n is None:
        n = 1
    return o3.Irreps([(n, (l, (-1) ** l)) for l in range(l_max + 1)])


def _make_irrep_change_of_basis(o3_basis: str = "xyz"):

    o3_matrix = torch.zeros((3, 3))
    o3_eye = torch.eye(3)

    if not isinstance(o3_basis, str) or len(o3_basis) > 3:
        raise ValueError(f"Expected o3_basis to be a string of length 3; " f'got: "{o3_basis}."')

    for i, ch in enumerate(o3_basis):
        ax = ch.lower()
        if ax not in "xyz":
            raise ValueError(f'Unknown axis "{ax}" in basis string "{o3_basis}"; ' f'expected "x", "y", or "z"')
        o3_matrix[:, i] = o3_eye["xyz".index(ch)]

    return o3_matrix


@AtomwiseConv.register("e3conv")
class AtomwiseE3Conv(AtomwiseConv):

    def __init__(
        self,
        out_features: int | o3.Irreps,
        in_features: int | o3.Irreps | None = None,
        hidden_features: int | o3.Irreps | None = None,
        layers: int = 1,
        activation: str = "leakyrelu",
        o3_basis: str = "xyz",
        l_max: int = 2,
        r_max: float = 10.0,
        r_points: int = 32,
        z_max: int = 118,
    ):

        if not isinstance(in_features, o3.Irreps):
            in_features = _scalar_irreps(in_features)

        if not isinstance(out_features, o3.Irreps):
            out_features = _scalar_irreps(out_features)

        if not isinstance(hidden_features, o3.Irreps):
            hidden_features = _hidden_irreps(hidden_features, l_max=l_max)

        super().__init__(
            out_features=out_features.dim,
            in_features=in_features.dim,
            z_max=z_max,
        )

        self.in_irreps = in_features
        self.out_irreps = out_features
        self.hidden_irreps = hidden_features
        self.atom_irreps = o3.Irreps([(z_max + 1, (0, 1))])
        self.neighbor_irreps = o3.Irreps.spherical_harmonics(lmax=l_max)

        self.r_max = r_max
        self.r_points = r_points

        D_output_transform = self.out_irreps.D_from_matrix(_make_irrep_change_of_basis(o3_basis))

        self.register_buffer("D_output_transform", D_output_transform)

        D_input_transform = (
            self.in_irreps.D_from_matrix(_make_irrep_change_of_basis(o3_basis)) if self.in_irreps else None
        )

        self.register_buffer("D_input_transform", D_input_transform)

        self.activation = get_activation_fn(activation)()

        layer_out_irreps = self.hidden_irreps if layers > 1 else self.out_irreps

        self.irrep_tps = nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    (self.atom_irreps + self.in_irreps),
                    self.neighbor_irreps,
                    layer_out_irreps,
                    shared_weights=False,
                )
            ]
        )

        for i in range(layers - 1):
            layer_in_irreps = self.hidden_irreps
            layer_out_irreps = self.hidden_irreps if i < (layers - 2) else self.out_irreps
            self.irrep_tps.append(
                o3.FullyConnectedTensorProduct(
                    layer_in_irreps,
                    self.neighbor_irreps,
                    layer_out_irreps,
                    shared_weights=False,
                )
            )

        self.edge_nns = nn.ModuleList(
            [e3nn.nn.FullyConnectedNet([r_points, r_points, tp.weight_numel], self.activation) for tp in self.irrep_tps]
        )

    def _get_atom_features(self, atomic_numbers, features):

        # build atomic "one-hot" encoded features
        atom_features = nn.functional.one_hot(atomic_numbers, num_classes=self.z_max + 1).to(torch.float32)

        if features is not None and self.D_input_transform is not None:
            transformed_features = features @ self.D_input_transform.T
            atom_features = torch.cat((atom_features, transformed_features), dim=-1)

        return atom_features

    def forward(self, atomic_numbers, neighbors, neighbor_distances, features, **kwargs):

        # generate edge list between neighboring atoms (the "bond pairs")
        bond_pairs, bond_distances = _get_neighbor_distance_edgelist(neighbors=neighbors, distances=neighbor_distances)

        atom_neighbors = torch.sum(neighbors >= 0, dim=-1, keepdim=True)

        # combine features into a feature tensor
        f = self._get_atom_features(atomic_numbers, features)

        # get the "bond features" from the atomic distances
        bond_features = e3nn.math.soft_one_hot_linspace(
            bond_distances.norm(dim=1),
            start=0.0,
            end=self.r_max,
            number=self.r_points,
            basis="smooth_finite",
            cutoff=True,
        )

        sh = o3.spherical_harmonics(
            self.neighbor_irreps,
            bond_distances,
            normalize=True,
            normalization="component",
        )

        for tp, edge_nn in zip(self.irrep_tps, self.edge_nns, strict=False):

            tp_weights = edge_nn(bond_features)
            f = scatter(
                tp(f[bond_pairs[0]], sh, tp_weights),
                bond_pairs[1],
                dim=0,
                dim_size=atomic_numbers.shape[0],
            ).div(atom_neighbors**0.5)

        f_out = f @ self.D_output_transform

        return f_out
