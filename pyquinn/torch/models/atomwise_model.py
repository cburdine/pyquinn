from pyquinn.torch.atom_graph import TorchAtomGraph
from pyquinn.torch.atomwise import AtomwiseConv

import torch.nn as nn


class TorchAtomwiseModel(nn.Module):

    def __init__(
            self,
            atomwise_model: AtomwiseConv):

        self.atomwise_model = atomwise_model

    def forward(self, graph: TorchAtomGraph) -> TorchAtomGraph:
        return self.predict(graph=graph)

    def predict(self, graph: TorchAtomGraph) -> TorchAtomGraph:
        
        positions = graph.positions
        neighbor_distances = positions[graph.neighbors] + \
            (graph.unit_cell_images @ graph.unit_cell) - \
            positions.unsqueeze(-2)

        features_out = self.atomwise_model(
            atomic_numbers=graph.atomic_numbers,
            neighbors=graph.neighbors,
            neighbor_distances=neighbor_distances,
            features=graph.features
        )

        return TorchAtomGraph(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            masses=graph.masses,
            unit_cell=graph.unit_cell,
            periodic=graph.periodic,
            outputs=features_out,
            graph_outputs=graph.graph_outputs,
            cutoffs=graph.cutoffs,
            compute_neighbors=False)

        



    


