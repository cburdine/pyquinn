
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch_geometric.nn import NNConv

from pyquinn.torch.util import get_activation_fn

def _get_neighbor_distance_edgelist(
        neighbors: torch.IntTensor, 
        distances: torch.Tensor) -> tuple[torch.IntTensor, torch.Tensor]:
    """Converts a -1 padded neighbor adjacency list and adjacency distances
       list (the default format of neighbors) to an edge
       list format with corresponding distances
    """

    site_idxs, site_adj_idxs = torch.nonzero(neighbors >= 0, as_tuple=True)
    neighbor_idxs = neighbors[site_idxs, site_adj_idxs]

    edgelist = torch.vstack([site_idxs, neighbor_idxs])
    edgelist_distances = distances[site_idxs, site_adj_idxs]

    return edgelist, edgelist_distances

class RBFEmbedding(nn.Module):
    def __init__(self, r_max: float = 5.0, r_points: int = 32):

        super(RBFEmbedding, self).__init__()
        self.register_buffer('r_pts', torch.linspace(0, r_max, r_points))

    def forward(self, dx):
        sigma = (self.r_pts[-1]-self.r_pts[0])/self.r_pts.shape[0]
        x = torch.exp(-0.5*(
            (self.r_pts - torch.norm(dx, dim=-1, keepdim=True))/sigma)**2)
        return x


class NeighborEdgeNN(nn.Module):

    def __init__(
            self, 
            edge_features: int, 
            in_features: int,
            out_features: int,
            layers: int = 2,
            activation: str = 'leaky_relu'):

        super(NeighborEdgeNN, self).__init__()

        matrix_features = in_features*out_features

        self.embedding_layer = nn.Linear(edge_features, matrix_features)
        self.dense_layers = nn.ModuleList([
            nn.Linear(matrix_features, matrix_features)
            for _ in range(layers-1)
        ])

        self.activation = get_activation_fn(activation)()

    def forward(self, x_edge):
        z_edge = self.embedding_layer(x_edge)

        for dense_layer in self.dense_layers:
            z_edge = dense_layer(self.activation(z_edge))

        return z_edge


class AtomwiseModel(nn.Module, ABC):

    def __init__(
            self, 
            out_features: int, 
            in_features: int | None = None,
            z_max: int = 118):

        super().__init__()

        self.out_features = out_features
        self.in_features = in_features if in_features is not None else 0
        self.z_max = z_max

    @abstractmethod
    def forward(self,
                atomic_numbers,
                positions,
                neighbors,
                features,
                **kwargs):
        raise NotImplementedError(
            "This method should be implemented in a "
            "subclass.")


class AtomwiseNNConv(AtomwiseModel):

    def __init__(
            self,
            out_features: int,
            in_features: int | None = None,
            layers: int = 1,
            aggr: str = 'add',
            activation: str = 'leakyrelu',
            neighbor_nn_layers: int = 2,
            neighbor_nn_activation: str = 'leakyrelu',
            r_max: float = 10.0,
            r_points: int = 32,
            z_max: int = 118):

        super().__init__(
            out_features=out_features,
            in_features=in_features,
            z_max=z_max)

        self.aggr = aggr
        self.activation = get_activation_fn(activation)()
        self.distance_embedding = RBFEmbedding(r_max=r_max, r_points=r_points)

        in_channels = (z_max+1) + self.in_features
        out_channels = self.out_features

        nnconv_layers = nn.ModuleList([
            NNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                nn=NeighborEdgeNN(
                    edge_features=r_points, 
                    in_features=in_channels, 
                    out_features=out_channels,
                    layers=neighbor_nn_layers,
                    activation=neighbor_nn_activation),
                aggr=self.aggr)
        ])

        for _ in range(layers - 1):
            nnconv_layers.append(
                NNConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    nn=NeighborEdgeNN(
                        edge_features=r_points, 
                        in_features=out_channels, 
                        out_features=out_channels, 
                        layers=neighbor_nn_layers,
                        activation=neighbor_nn_activation),
                    aggr=self.aggr)
            )

        self.nnconv_layers = nnconv_layers

    def forward(
            self,
            atomic_numbers: torch.IntTensor,
            neighbors: torch.IntTensor,
            neighbor_distances: torch.Tensor,
            features: torch.Tensor | None = None) -> torch.Tensor:

        # generate edge list between neighboring atoms (the "bond pairs"):
        bond_pairs, bond_distances = \
            _get_neighbor_distance_edgelist(
                neighbors=neighbors, distances=neighbor_distances)

        # get the "bond features" from the atomic distances:
        bond_features = self.distance_embedding(bond_distances)

        # build atomic "one-hot" encoded features:
        atom_features = torch.zeros(
            atomic_numbers.shape[0], self.z_max+1, 
            dtype=bond_features.dtype, device=atomic_numbers.device)
        atom_features[:, atomic_numbers] += 1
        
        # combine features into feature tensors:
        x = atom_features if features is None else \
            torch.cat((atom_features, features), dim=-1)

        for layer in self.nnconv_layers[:-1]:
            x = self.activation(
                layer(x, edge_index=bond_pairs, edge_attr=bond_features))

        y = self.nnconv_layers[-1](
            x, edge_index=bond_pairs, edge_attr=bond_features)

        return y
