from pyquinn.atoms.bandstructure import BandStructure
from pyquinn.atoms.dataset import AtomGraphDataset
from pyquinn.atoms.bandstructure_dataset import (
    BandStructureDataset, 
    BackendBandStructureDataset
)
from pyquinn.torch.dataset import TorchAtomGraphDataset
from pyquinn.torch.bandstructure import TorchBandStructure

import os
from glob import glob

from tqdm import tqdm
import torch


class TorchBandStructureDataset(BackendBandStructureDataset):

    @classmethod
    def _compile_bands(
            cls,
            bands: BandStructure | str,
            name: str,
            cache_dir: str,
            **compile_kwargs):
        
        # load bands (if necessary)
        bands = BandStructure.load(bands) \
            if isinstance(bands, str) else bands

        # compile bands
        torch_bands = bands.compile(
            backend='torch',
            **compile_kwargs)

        torch_bands_path = os.path.join(
            cache_dir, f'{name}.pt.bands')
        torch_bands.save(torch_bands_path)

        return torch_bands_path

    @classmethod 
    def _save_bands(
            cls,
            bands: BandStructure | str,
            name: str,
            cache_dir: str):
        
        # load bands (if necessary)
        bands = TorchBandStructure.load(bands) \
            if isinstance(bands, str) else bands

        # save bands to cache dir
        torch_

    @classmethod
    def compile(
            cls,
            dataset: BandStructureDataset,
            cache_dir: str,
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            max_k_pts: int | None = None,
            max_energies: int | None = None,
            device: torch.device = 'cpu',
            dtype: torch.dtype = torch.float32,
            idx_dtype: torch.dtype = torch.int64):
        
        torch_graphs = []
        torch_band_structures = []

        for name, (graph, bands) in tqdm(dataset.items(), total=len(dataset)):

            # compile and add graph
            torch_graphs.append(
                TorchAtomGraphDataset._compile_graph(
                    graph=graph,
                    name=name,
                    cache_dir=cache_dir,
                    cutoffs=cutoffs,
                    masses=masses,
                    symbols=symbols,
                    device=device,
                    dtype=dtype,
                    idx_dtype=idx_dtype))

            # compile and add band structure
            torch_band_structures.append(
                TorchBandStructureDataset._compile_bands(
                    bands=bands,
                    name=name,
                    cache_dir=cache_dir,
                    max_k_pts=max_k_pts,
                    max_energies=max_energies,
                    device=device,
                    dtype=dtype,
                )
            )

        return TorchBandStructureDataset(
            graphs=torch_graphs,
            band_structures=torch_band_structures,
            names=dataset.names,
            cache_dir=cache_dir,
            device=device
        )

    def __init__(
            self,
            graphs: list[str],
            band_structures: list[str],
            names: list[str],
            cache_dir: str,
            device: torch.device = 'cpu'):

        super().__init__(
            self,
            graphs=graphs,
            band_structures=band_structures,
            names=names,
            cache_dir=cache_dir)

        self.device = device

    @classmethod
    def load(
            cls,
            cache_dir: str,
            device: torch.device='cpu'):

        graphs = glob(os.path.join(cache_dir, '*.pt'))
        band_structures = glob(os.path.join(cache_dir, '*.pt.bands'))
        names = [
            AtomGraphDataset._parse_graph_name(file)
            for file in graphs
        ]

        return TorchBandStructureDataset(
            graphs=graphs,
            band_structures=band_structures,
            names=names,
            cache_dir=cache_dir,
            device=device)

    def save(self, cache_dir: str):

        _dataset_iter = tqdm(zip(
            self.names, self.graphs, self.band_structures
        ), total=len(self.names))

        for name, graph, bands in _dataset_iter:
            TorchAtomGraphDataset._save_graph(
                graph=graph,
                name=name,
                cache_dir=cache_dir,
                device=self.device)

            TorchBandStructureDataset._save_b
