from pyquinn.torch.types import TorchAtomGraph, TorchOperator
from pyquinn.torch.nn.lcao.lcao_hamiltonian import TBHamiltonian, VariationalTBHamiltonian
from pyquinn.torch.types.bandstructure_dataset import TorchBandStructureDataset
from pyquinn.torch.types.bandstructure import TorchBandStructure
from pyquinn.torch.util import get_optimizer, cuda_stream_map
from pyquinn.io.stdout import progress_bar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np


class TorchBandStructureModel(nn.Module):

    def __init__(
            self, 
            bands_model: TBHamiltonian | None = None,
            **model_kwargs):

        super().__init__()

        self.bands_model = VariationalTBHamiltonian(**model_kwargs) \
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

    def orbital_indices(
        self, 
        graph: TorchAtomGraph
    ) -> tuple[torch.IntTensor, torch.IntTensor]:
        return self.bands_model.orbital_indices(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images)

    def orbital_quantum_numbers(
        self, 
        graph: TorchAtomGraph
    ) -> torch.IntTensor:
        return self.bands_model.orbital_quantum_numbers(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images)

    def orbital_names(self, graph: TorchAtomGraph) -> list[str]:
        return self.bands_model.orbital_names(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images)

    def hamiltonian_matrix(self, graph: TorchAtomGraph, k_pts: torch.Tensor):
        return self.bands_model.hamiltonian_matrix(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images,
            k_pts=k_pts,
            sparse=False)

    def graph_hamiltonian_operator(self, graph: TorchAtomGraph):
        raise NotImplementedError()

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
            bands.shape+(n_sites, n_orbital_indices),
            device=bloch_projections.device)
        
        band_projections[..., site_indices, orbital_indices] += \
            bloch_projections

        return bands, band_projections

    def atomic_orbital_amplitudes(
            self,
            graph: TorchAtomGraph,
            x_pts: torch.Tensor,
            site_idxs: torch.IntTensor,
            orbital_idxs: torch.Tensor):

        if not isinstance(self.bands_model, TBHamiltonian):
            raise ValueError(
                f'Orbital amplitudes are only supported for instances of TBHamiltonian (got: {type(self.bands_model)})')

        return self.bands_model.atomic_orbital_amplitudes(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images,
            site_idxs=site_idxs,
            orbital_idxs=orbital_idxs,
            x_pts=x_pts,
        )

    def bloch_eigenstate_amplitudes(
            self,
            graph: TorchAtomGraph,
            x_pts: torch.Tensor,
            k_pts: torch.Tensor,
            eigenstate_idxs: torch.IntTensor | None = None,
            eigenstates: torch.IntTensor | None = None,
            k_idxs: torch.IntTensor | None = None,
            x_cutoff: torch.Tensor | None = None):

        if not isinstance(self.bands_model, TBHamiltonian):
            raise ValueError(
                f'Orbital amplitudes are only supported for instances of '
                f'TBHamiltonian (got: {type(self.bands_model)})')

        return self.bands_model.bloch_eigenstate_amplitudes(
            atomic_numbers=graph.atomic_numbers,
            positions=graph.positions,
            neighbors=graph.neighbors,
            unit_cell=graph.unit_cell,
            unit_cell_images=graph.unit_cell_images,
            x_pts=x_pts,
            k_pts=k_pts,
            eigenstate_idxs=eigenstate_idxs,
            eigenstates=eigenstates,
            k_idxs=k_idxs,
            x_cutoff=x_cutoff,
        )

    @staticmethod
    def _collate_fn(batch):
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
            projection_weight: float = 0.1,
            device: torch.device | None = None,
            data_workers: int = 0,
            cuda_streams: int = 0) -> dict:
        
        # move model to device
        self.to(device)

        # initialize optimizer:
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = lr

        optim = get_optimizer(optimizer)(self.parameters(), **optimizer_kwargs)   

        # initialize data loaders
        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=data_workers,
            collate_fn=TorchBandStructureModel._collate_fn)
        val_dl = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=data_workers,
            collate_fn=TorchBandStructureModel._collate_fn)

        # initialize batch streams
        batch_streams = [torch.cuda.Stream() for _ in range(cuda_streams)]

        # loss function evaluator
        def eval_loss(data):
            graph, band_structure = data
            return self._compute_loss(
                graph=graph.to(device),
                band_structure=band_structure.to(device),
                use_projections=use_projections,
                projection_weight=projection_weight)

        # create history dictionary
        history = {
            'train_loss': [],
            'validation_loss': [],
        }

        # for p in self.parameters():
        #     if p.requires_grad:
        #         p.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))

        # epoch and dataset iterators
        epoch_iter = progress_bar if verbosity == 1 else (lambda x: x)
        dataset_iter = progress_bar if verbosity >= 2 else (lambda x: x)

        for n in epoch_iter(range(epochs)):

            if verbosity >= 2:
                print(f'Epoch {n}:')
            
            for batch in dataset_iter(train_dl):

                batch_losses = cuda_stream_map(
                    eval_loss, batch, batch_streams)
                loss = sum(batch_losses) / len(batch_losses)

                loss.backward()
                optim.step()
                self.zero_grad()

            if verbosity >= 2:
                print('Evaluating training and validation loss:')

            train_losses = []
            val_losses = []

            with torch.no_grad():
                for batch in dataset_iter(train_dl):

                    # compute training loss
                    batch_losses = cuda_stream_map(
                        eval_loss, batch, batch_streams)
                    loss = sum(batch_losses) / len(batch_losses)

                    train_losses.append(loss.detach().cpu().numpy())

                if val_dataset is not None:
                    for batch in dataset_iter(val_dl):
                        
                        # compute validation loss
                        batch_losses = cuda_stream_map(
                            eval_loss, batch, batch_streams)
                        loss = sum(batch_losses) / len(batch_losses)

                        val_losses.append(loss.detach().cpu().numpy())

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


# add TorchBandStructureModel as safe to serialize
torch.serialization.add_safe_globals([TorchBandStructureModel])