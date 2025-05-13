from pyquinn.types import BandStructure, AtomGraphDataset
from pyquinn.types.bandstructure_dataset import (
    BandStructureDataset, 
    BackendBandStructureDataset
)
from pyquinn.torch.types import (
    TorchAtomGraphDataset,
    TorchBandStructure,
    TorchAtomGraph,
)

import os
from glob import glob

from tqdm import tqdm
import torch


@BandStructureDataset.register_backend('torch')
class TorchBandStructureDataset(
        BackendBandStructureDataset, 
        torch.utils.data.Dataset):

    @staticmethod
    def _compile_bands(
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

    @staticmethod 
    def _save_bands(
            bands: TorchBandStructure | str,
            name: str,
            cache_dir: str,
            device: torch.device = 'cpu'):
        
        # load bands (if necessary)
        bands = TorchBandStructure.load(bands, device=device) \
            if isinstance(bands, str) else bands

        # save bands to cache dir
        torch_bands_path = os.path.join(
            cache_dir, f'{name}.pt.bands')
        bands.save(torch_bands_path)

    @staticmethod
    def compile(
            dataset: BandStructureDataset,
            cache_dir: str,
            cutoffs: dict | list[float] | None = None,
            masses: dict | list[float] | None = None,
            symbols: dict | list[str] | None = None,
            max_k_pts: int | None = None,
            max_bands: int | None = None,
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
                    max_bands=max_bands,
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
            graphs=graphs,
            band_structures=band_structures,
            names=names,
            cache_dir=cache_dir)

        self.device = device

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = TorchAtomGraph.load(
            filename=self.graphs[index],
            device=self.device)
        bands = TorchBandStructure.load(
            filename=self.band_structures[index],
            device=self.device)

        return graph, bands

    def items(self):
        for i in range(len(self)):
            yield self.names[i], self[i] 

    @staticmethod
    def load(
            cache_dir: str,
            device: torch.device = 'cpu'):

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

            TorchBandStructureDataset._save_bands(
                bands=bands,
                name=name,
                cache_dir=cache_dir,
                device=self.device
            )
