from pyquinn.torch.atom_graph import TorchAtomGraph
from pyquinn.torch.atomwise import AtomwiseConv
from pyquinn.torch.atomwise_pool import AtomwisePool
from pyquinn.torch.dataset import AtomGraphDataset
from pyquinn.torch.util import get_optimizer

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class TorchAtomwiseModel(nn.Module):

    def __init__(
            self,
            atomwise_model: AtomwiseConv,
            atomwise_pool: AtomwisePool | None = None):

        super().__init__()

        self.atomwise_model = atomwise_model
        self.atomwise_pool = atomwise_pool

    def forward(self, graph: TorchAtomGraph) -> TorchAtomGraph:
        return self.predict(graph=graph)

    def predict(self, graph: TorchAtomGraph) -> TorchAtomGraph:
        
        positions = graph.positions
        neighbor_distances = positions[graph.neighbors] + \
            (graph.unit_cell_images.to(torch.float32) @ graph.unit_cell) - \
            positions.unsqueeze(-2)

        features_out = self.atomwise_model(
            atomic_numbers=graph.atomic_numbers,
            neighbors=graph.neighbors,
            neighbor_distances=neighbor_distances,
            features=graph.features
        )

        if self.atomwise_pool is None:
            pooled_features_out = torch.zeros_like(graph.graph_outputs)
        else:
            pooled_features_out = self.atomwise_model(
                atomic_numbers=graph.atomic_numbers,
                neighbors=graph.neighbors,
                neighbor_distances=neighbor_distances,
                features=features_out
            )

        return TorchAtomGraph(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            masses=graph.masses,
            unit_cell=graph.unit_cell,
            periodic=graph.periodic,
            features=graph.features,
            outputs=features_out,
            graph_outputs=pooled_features_out,
            cutoffs=graph.cutoffs,
            compute_neighbors=False)

    def _outputs_loss_fn(
            self, 
            pred_outputs: torch.Tensor, 
            target_outputs: torch.Tensor):

        loss = torch.mean(
            torch.square(pred_outputs - target_outputs))
        
        return loss

    def _graph_outputs_loss_fn(
            self, 
            pred_outputs: torch.Tensor, 
            target_outputs: torch.Tensor):

        loss = torch.mean(
            torch.square(pred_outputs - target_outputs))
        
        return loss

    def _compute_loss(
            self,
            graph: TorchAtomGraph,
            use_outputs: bool = True,
            use_graph_outputs: bool = True,
            outputs_weight: float = 1.0,
            graph_outputs_weight: float = 1.0):

        assert use_outputs or use_graph_outputs, \
            "Loss function must either use outputs or graph outputs."
        
        pred_graph = self.predict(graph)

        loss = 0.0
        if use_graph_outputs:
            loss += graph_outputs_weight * self._graph_outputs_loss_fn(
                pred_graph.graph_outputs,
                graph.graph_outputs)
        if use_outputs:
            loss += outputs_weight * self._outputs_loss_fn(
                pred_graph.outputs,
                graph.outputs)

            return loss
    
    def fit(self,
            train_dataset: AtomGraphDataset | None = None,
            val_dataset: AtomGraphDataset | None = None,
            epochs: int = 100,
            lr: float = 1e-3,
            optimizer: str = 'adam',
            optimizer_kwargs: dict | None = None,
            verbosity: int = 2,
            use_outputs: bool = True,
            use_graph_outputs: bool = True,
            outputs_weight: float = 1.0,
            graph_outputs_weight: float = 1.0) -> dict:
        
        # initialize optimizer:
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = lr

        optim = get_optimizer(optimizer)(self.parameters(), **optimizer_kwargs)   

        train_dl = DataLoader(
            train_dataset, batch_size=None, shuffle=True, num_workers=1)
        val_dl = DataLoader(
            val_dataset, batch_size=None, shuffle=False)

        # create history dictionary
        history = {
            'train_loss': [],
            'validation_loss': [],
        }

        epoch_iter = tqdm if verbosity == 1 else (lambda x: x)
        dataset_iter = tqdm if verbosity >= 2 else (lambda x: x)

        for n in epoch_iter(range(epochs)):

            if verbosity >= 2:
                print(f'Epoch {n}:')
            for graph in dataset_iter(train_dl):

                # compute loss
                loss = self._compute_loss(
                    graph=graph,
                    use_outputs=use_outputs,
                    use_graph_outputs=use_graph_outputs,
                    outputs_weight=outputs_weight,
                    graph_outputs_weight=graph_outputs_weight)

                loss.backward()
                optim.step()
                self.zero_grad()

            if verbosity >= 2:
                print('Evaluating training and validation loss:')

            train_losses = []
            val_losses = []

            with torch.no_grad():
                for graph in dataset_iter(train_dl):

                    # compute training loss
                    loss = self._compute_loss(
                        graph=graph,
                        use_outputs=use_outputs,
                        use_graph_outputs=use_graph_outputs,
                        outputs_weight=outputs_weight,
                        graph_outputs_weight=graph_outputs_weight)

                    train_losses.append(loss.detach().numpy())

                if val_dataset is not None:
                    for graph in dataset_iter(val_dl):
                        
                        # compute validation loss
                        loss = self._compute_loss(
                            graph=graph,
                            use_outputs=use_outputs,
                            use_graph_outputs=use_graph_outputs,
                            outputs_weight=outputs_weight,
                            graph_outputs_weight=graph_outputs_weight)

                        val_losses.append(loss.detach().numpy())

            epoch_train_loss = np.nanmean(train_losses)
            epoch_validation_loss = None if val_dataset is None \
                else np.nanmean(val_losses)

            history['train_loss'].append(epoch_train_loss)
            history['validation_loss'].append(epoch_validation_loss)

            if verbosity >= 2:
                print(f'Train loss: {epoch_train_loss}; '
                      f'Validation loss: {epoch_validation_loss}.')

        for k, v in history.items():
            history[k] = np.array(v)

        return history
        



    


