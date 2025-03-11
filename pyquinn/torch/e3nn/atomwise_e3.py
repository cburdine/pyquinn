import e3nn.o3 as o3
import torch.nn as nn
import e3nn
import torch
from torch_geometric.utils import scatter

from pyquinn.torch.atomwise import (
    AtomwiseConv, 
    _get_neighbor_distance_edgelist
)
from pyquinn.torch.util import get_activation_fn


def _scalar_irreps(n: int | None):
    return o3.Irreps() if n is None \
        else o3.Irreps([(n, (0, 1))])


@AtomwiseConv.register('e3conv')
class AtomwiseE3Conv(AtomwiseConv):

    def __init__(
            self,
            out_features: int | o3.Irreps,
            in_features: int | o3.Irreps | None = None,
            layers: int = 1,
            activation: str = 'leakyrelu',
            l_max: int = 2,
            r_max: float = 5.0,
            r_points: int = 32,
            z_max: int = 118):
        
        if not isinstance(in_features, o3.Irreps):
            in_features = _scalar_irreps(in_features)

        if not isinstance(out_features, o3.Irreps):
            out_features = _scalar_irreps(out_features)

        super().__init__(
            out_features=out_features.dim,
            in_features=in_features.dim,
            z_max=z_max)

        self.in_irreps = in_features
        self.out_irreps = out_features
        self.atom_irreps = o3.Irreps([(z_max+1, (0, 1))])
        self.neighbor_irreps = o3.Irreps.spherical_harmonics(lmax=l_max)

        self.r_max = r_max
        self.r_points = r_points

        self.activation = get_activation_fn(activation)()

        self.irrep_tps = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                (self.atom_irreps + self.in_irreps), 
                self.neighbor_irreps, self.out_irreps,
                shared_weights=False)
        ])

        for _ in range(layers-1):
            self.irrep_tps.append(
                o3.FullyConnectedTensorProduct(
                    self.out_irreps, self.neighbor_irreps, self.out_irreps,
                    shared_weights=False))
        
        self.edge_nns = nn.ModuleList([
            e3nn.nn.FullyConnectedNet(
                [r_points, tp.weight_numel], 
                self.activation)
            for tp in self.irrep_tps
        ])

    def forward(
            self,
            atomic_numbers,
            neighbors,
            neighbor_distances,
            features,
            **kwargs):

        # generate edge list between neighboring atoms (the "bond pairs"):
        bond_pairs, bond_distances = \
            _get_neighbor_distance_edgelist(
                neighbors=neighbors, distances=neighbor_distances)

        # build atomic "one-hot" encoded features:
        atom_features = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=self.z_max+1
        ).to(torch.float32)

        atom_neighbors = torch.sum(neighbors >= 0, dim=-1, keepdim=True)
        
        # combine features into feature tensors:
        f = atom_features if features is None else \
            torch.cat((atom_features, features), dim=-1)

        # get the "bond features" from the atomic distances:
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

        print(f)

        for tp, edge_nn in zip(self.irrep_tps, self.edge_nns):
            tp_weights = edge_nn(bond_features)
            f = scatter(
                tp(f[bond_pairs[0]], sh, tp_weights),
                bond_pairs[1], 
                dim=0, 
                dim_size=atomic_numbers.shape[0]).div(atom_neighbors**0.5)

        return f

