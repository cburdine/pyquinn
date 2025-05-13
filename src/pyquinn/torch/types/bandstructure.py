from pyquinn.types.bandstructure import BandStructure, BackendBandStructure
import torch

@BandStructure.register_backend('torch')
class TorchBandStructure(BackendBandStructure):

    @staticmethod
    def compile(
            bands: BandStructure,
            max_k_pts: int | None = None,
            max_bands: int | None = None,
            device: torch.device = 'cpu',
            dtype: torch.dtype = torch.float32):

        bands = bands.downsample_bands(
            max_k_pts=max_k_pts, 
            max_bands=max_bands)

        energies = torch.tensor(
            bands.energies,
            device=device, dtype=dtype)

        k_pts = torch.tensor(
            bands.k_pts,
            device=device, dtype=dtype)

        projections = torch.tensor(
            bands.projections,
            device=device, dtype=dtype)

        chemical_potential = torch.tensor(
            bands.projections
        )

        return TorchBandStructure(
            energies=energies,
            k_pts=k_pts,
            projections=projections,
            chemical_potential=chemical_potential)

    def __init__(
            self,
            energies,
            k_pts,
            projections,
            chemical_potential):

        super().__init__(
            energies=energies,
            k_pts=k_pts,
            projections=projections,
            chemical_potential=chemical_potential)

    def detach(self):
        return TorchBandStructure(
            energies=self.energies.detach(),
            k_pts=self.k_pts.detach(),
            projections=self.projections.detach(),
            chemical_potential=self.chemical_potential.detach())

    def copy(self):
        return TorchBandStructure(
            energies=self.energies.copy(),
            k_pts=self.k_pts.copy(),
            projections=self.projections.copy(),
            chemical_potential=self.chemical_potential.copy())

    def to(self, **kwargs):
        return TorchBandStructure(
            energies=self.energies.to(**kwargs),
            k_pts=self.k_pts.to(**kwargs),
            projections=self.projections.to(**kwargs),
            chemical_potential=self.chemical_potential.to(**kwargs))

    def extract(self) -> BandStructure:
        return BandStructure(
            energies=self.energies.detach().numpy(),
            k_pts=self.k_pts.detach().numpy(),
            projections=self.projections.detach().numpy(),
            chemical_potential=self.chemical_potential.detach().numpy())

    @staticmethod
    def load(
            filename: str, 
            device: torch.device | None = None) -> 'TorchBandStructure':
        
        graph = torch.load(filename, weights_only=True, map_location=device)

        if not isinstance(graph, TorchBandStructure):
            raise IOError(
                f'Expected an instance of TorchBandStructure; '
                f'got: {type(graph)}')

        return graph

    def save(self, filename: str):
        torch.save(self, filename)

    
# add TorchBandStructure as safe to serialize
torch.serialization.add_safe_globals([TorchBandStructure])
