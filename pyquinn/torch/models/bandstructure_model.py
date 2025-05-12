from pyquinn.torch.atom_graph import TorchAtomGraph
from pyquinn.torch.lcao.lcao_hamiltonian import TBModel, VariationalTBModel
from pyquinn.torch.bandstructure_dataset import TorchBandStructureDataset
from pyquinn.torch.bandstructure import TorchBandStructure
from pyquinn.torch.util import get_optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


class TorchBandStructureModel(nn.Module):

    def __init__(
            self, 
            bands_model: TBModel | None = None,
            **model_kwargs):

        super().__init__()

        self.bands_model = VariationalTBModel(**model_kwargs) \
            if bands_model is None else bands_model

    def forward(self, graph: TorchAtomGraph, k_pts: torch.Tensor):
        return self.bands(graph=graph, k_pts=k_pts)

    def predict(
            self, 
            graph: TorchAtomGraph, 
            k_pts: torch.Tensor) -> TorchBandStructure:
        
        bands, projections = self.band_projections(
            graph=graph,
            k_pts=k_pts,
        )

        chemical_potential = torch.tensor(0.0, device=bands.device)

        return TorchBandStructure(
            energies=bands,
            k_pts=k_pts,
            projections=projections,
            chemical_potential=chemical_potential,
        )

    def hamiltonian_matrix(self, graph: TorchAtomGraph, k_pts: torch.Tensor):
        return self.bands_model.hamiltonian_matrix(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images,
            k_pts=k_pts,
            sparse=False)

    def band_eigenstates(self, graph: TorchAtomGraph, k_pts: torch.Tensor):

        bands, band_eigenstates = self.bands_model.eigenstates(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images,
            k_pts=k_pts,
            sparse=False)

        return bands, band_eigenstates

    def _band_loss_fn(
            self, 
            pred_bands: torch.Tensor, 
            target_bands: torch.Tensor):

        n_bands = min(pred_bands.shape[-1], target_bands.shape[-1])
        loss = torch.mean(
            torch.square(pred_bands[..., :n_bands] -
                         target_bands[..., :n_bands]))
        
        return loss

    def _projection_loss_fn(
            self,
            pred_projections: torch.Tensor,
            target_projections: torch.Tensor):

        n_bands = min(
            pred_projections.shape[1], 
            target_projections.shape[1])
        n_orbitals = min(
            pred_projections.shape[-1],
            target_projections.shape[-1])

        loss = torch.mean(
            torch.sum(torch.square(
                pred_projections[:, :n_bands, :, :n_orbitals] - 
                target_projections[:, :n_bands, :, :n_orbitals],
            ), dim=(-2, -1)))

        return loss

    def _compute_loss(
            self,
            graph: TorchAtomGraph, 
            band_structure: TorchBandStructure,
            use_projections: bool = False,
            projection_weight: float = 1.0):

        if use_projections:

            pred_bands, pred_projections = self.band_projections(
                graph=graph,
                k_pts=band_structure.k_pts)

            band_loss = self._band_loss_fn(
                pred_bands=pred_bands, 
                target_bands=band_structure.energies)
            projection_loss = self._projection_loss_fn(
                pred_projections=pred_projections,
                target_projections=band_structure.projections)

            loss = band_loss + projection_weight * projection_loss
            
        else:
            pred_bands = self.bands(
                graph=graph,
                k_pts=band_structure.k_pts
            )

            loss = self._band_loss_fn(
                pred_bands=pred_bands,
                target_bands=band_structure.energies,
            )

        return loss

    def bands(self, graph: TorchAtomGraph, k_pts: torch.Tensor):

        return self.bands_model(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images,
            k_pts=k_pts,
            sparse=False)

    def band_projections(self, graph: TorchAtomGraph, k_pts: torch.Tensor):

        bands, bloch_eigenstates = self.bands_model.eigenstates(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images,
            k_pts=k_pts,
            sparse=False)

        site_indices, orbital_indices = self.bands_model.orbital_indices(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
        )

        bloch_projections = torch.square(
            torch.abs(bloch_eigenstates.transpose(-2, -1)))

        n_sites = graph.atomic_numbers.shape[0]
        n_orbital_indices = torch.max(orbital_indices) + 1

        band_projections = torch.zeros(
            bands.shape+(n_sites, n_orbital_indices))
        
        band_projections[..., site_indices, orbital_indices] += \
            bloch_projections

        return bands, band_projections

    @staticmethod
    def _collate_batch(batch):
        return batch
    
    def fit(self,
            train_dataset: TorchBandStructureDataset,
            val_dataset: TorchBandStructureDataset | None = None,
            epochs: int = 100,
            batch_size: int = 1,
            lr: float = 1e-3,
            optimizer: str = 'adam',
            optimizer_kwargs: dict | None = None,
            verbosity: int = 2,
            use_projections: bool = False,
            projection_weight: float = 0.1) -> dict:
        
        # initialize optimizer:
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = lr

        optim = get_optimizer(optimizer)(self.parameters(), **optimizer_kwargs)   

        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
            collate_fn=TorchBandStructureModel._collate_batch)
        val_dl = DataLoader(
            val_dataset, batch_size=None, shuffle=False)

        # create history dictionary
        history = {
            'train_loss': [],
            'validation_loss': [],
        }

        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))

        epoch_iter = tqdm if verbosity == 1 else (lambda x: x)
        dataset_iter = tqdm if verbosity >= 2 else (lambda x: x)

        for n in epoch_iter(range(epochs)):

            if verbosity >= 2:
                print(f'Epoch {n}:')
            
            for batch in dataset_iter(train_dl):

                # try:
                # compute loss
                loss = sum(
                    self._compute_loss(
                        graph=graph,
                        band_structure=band_structure,
                        use_projections=use_projections,
                        projection_weight=projection_weight)
                    for graph, band_structure in batch) / batch_size

                loss.backward()
                optim.step()
                
                # except RuntimeError as e:
                #     logging.error(
                #         'Warning: Possibly failed to diagonalize Hamiltonian due to ill-conditioning '
                #         ' or degeneracy. Try increasing H_noise or decreasing the learning rate:\n'
                #         '%s', e)

                # finally:
                self.zero_grad()

            if verbosity >= 2:
                print('Evaluating training and validation loss:')

            train_losses = []
            val_losses = []

            with torch.no_grad():
                for batch in dataset_iter(train_dl):

                    # compute training loss
                    loss = sum(
                        self._compute_loss(
                            graph=graph,
                            band_structure=band_structure,
                            use_projections=use_projections,
                            projection_weight=projection_weight)
                        for graph, band_structure in batch) / batch_size

                    train_losses.append(loss.detach().numpy())

                if val_dataset is not None:
                    for batch in dataset_iter(val_dl):
                        
                        # compute validation loss
                        loss = sum(
                            self._compute_loss(
                                graph=graph, 
                                band_structure=band_structure, 
                                use_projections=use_projections, 
                                projection_weight=projection_weight)
                            for graph, band_structure in batch) / batch_size

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
            

            

