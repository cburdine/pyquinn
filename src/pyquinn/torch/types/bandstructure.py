from pyquinn.types.bandstructure import BandStructure, BackendBandStructure
import torch

@BandStructure.register_backend('torch')
class TorchBandStructure(BackendBandStructure):

    _dos_smearing = [
        'gaussian',
        'fermi-dirac'
    ]

    _idos_smearing = [
        'gaussian',
        'fermi-dirac'
    ]

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

    def clone(self):
        return TorchBandStructure(
            energies=self.energies.clone(),
            k_pts=self.k_pts.clone(),
            projections=self.projections.clone(),
            chemical_potential=self.chemical_potential.clone())

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

    def _gaussian_dos(
            self, 
            band_energies: torch.Tensor,
            energy: torch.Tensor, 
            sigma: float,
            projections: torch.Tensor,
            weight: float) -> torch.Tensor:
        
        # compute Gaussian dos
        unnorm_dos = torch.sum(
            projections * torch.exp(
                -((energy - band_energies)/sigma)**2), axis=1)
        dos = weight * unnorm_dos / ((torch.pi**0.5) * sigma)

        return dos

    def _gaussian_idos(
            self, 
            band_energies: torch.Tensor,
            energy: torch.Tensor, 
            sigma: float,
            projections: torch.Tensor,
            weight: float) -> torch.Tensor:
        
        # compute Gaussian dos
        unnorm_dos = torch.sum(
            projections * 0.5*(
                1.0 + torch.erf((energy - band_energies)/sigma)), 
            axis=1)
        dos = weight * unnorm_dos

        return dos

    def _fermi_dirac_dos(
            self,
            band_energies: torch.Tensor,
            energy: torch.Tensor, 
            sigma: float,
            projections: torch.Tensor,
            weight: float) -> torch.Tensor:

        # compute Fermi-Dirac dos
        unnorm_dos = torch.sum(
            projections / (1.0 + torch.cosh((energy - band_energies)/sigma)), 
            axis=1)

        dos = weight * unnorm_dos / (2*sigma)
        return dos

    def _fermi_dirac_idos(
            self,
            band_energies: torch.Tensor,
            energy: torch.Tensor, 
            sigma: float,
            projections: torch.Tensor,
            weight: float) -> torch.Tensor:

        # compute Fermi-Dirac dos
        unnorm_dos = torch.sum(
            projections * (
                0.5 + 0.5*torch.tanh((energy - band_energies)/(2*sigma))), 
            axis=1)

        dos = weight * unnorm_dos
        return dos

    def _get_dos_data(
            self,
            energy: torch.Tensor,  
            projections: torch.Tensor | None = None):

        if projections is not None and projections.ndim > 0:

            if projections.shape[:2] != self.energies.shape[:2]:
                raise ValueError(
                    'Expected first two dimensions of projections'
                    ' to match self.energies. (projections shape '
                    f'{projections.shape} incompatible with '
                    f'{self.energies.shape}.)')

            projections = projections.reshape(
                (1, self.energies.numel()) + projections.shape[2:])
            energy = energy.reshape((-1,) + (1,)*(projections.ndim-1))
            band_energies = self.energies.reshape(
                (1, -1)+(1,)*(projections.ndim-2))
        else:
            projections = torch.tensor(
                1.0, 
                dtype=self.energies.dtype,
                device=self.energies.device)
            
            energy = energy.reshape((-1, 1))
            band_energies = self.energies.reshape(1, -1)
        
        return band_energies, energy, projections

    def _default_sigma(self):
        E_min = self.energies.min()
        E_max = self.energies.max()
        N = self.energies.numel()
       
        sigma = (E_max - E_min)/(N**0.5)

        return sigma

    def dos(
            self,
            energy: torch.Tensor | float = 0.0,
            smearing: str = 'gaussian',
            sigma: float | None = None,
            spin_mult: float = 1.0,
            projections: torch.Tensor | None = None) -> torch.Tensor:
        
        if isinstance(energy, float):
            energy = torch.tensor(
                energy,
                dtype=self.energies.dtype,
                device=self.energies.device
            )

        weight = spin_mult / self.k_pts.shape[0]
        sigma = self._default_sigma() if sigma is None else sigma

        band_energies, energy, projections = \
            self._get_dos_data(energy, projections)
        
        if smearing == 'gaussian':
            return self._gaussian_dos(
                band_energies, energy, sigma, projections, weight)
        elif smearing == 'fermi-dirac':
            return self._fermi_dirac_dos(
                band_energies, energy, sigma, projections, weight)
        else:
            raise ValueError(
                f'Unknown smearing type {smearing} '
                f'(expected one of {self._idos_smearing}).')

    def idos(
            self,
            energy: torch.Tensor | float = 0.0,
            smearing: str = 'gaussian',
            sigma: float | None = None,
            spin_mult: float = 1.0,
            projections: torch.Tensor | None = None) -> torch.Tensor:

        if isinstance(energy, float):
            energy = torch.tensor(
                energy,
                dtype=self.energies.dtype,
                device=self.energies.device
            )

        weight = spin_mult / self.k_pts.shape[0]
        sigma = self._default_sigma() if sigma is None else sigma
        
        band_energies, energy, projections = \
            self._get_dos_data(energy, projections)

        if smearing == 'gaussian':
            return self._gaussian_idos(
                band_energies, energy, sigma, projections, weight)
        elif smearing == 'fermi-dirac':
            return self._fermi_dirac_idos(
                band_energies, energy, sigma, projections, weight)
        else:
            raise ValueError(
                f'Unknown smearing type {smearing} '
                f'(expected one of {self._dos_smearing}).')

    def idos_level(
            self,
            n: torch.Tensor | float,
            smearing: str = 'fermi-dirac',
            sigma: float = None,
            spin_mult: float = 1.0,
            projections: torch.Tensor | None = None,
            E_guess: torch.Tensor | None = None,
            max_evals: int = 16) -> torch.Tensor:
        
        if isinstance(n, float):
            n = torch.tensor(
                n,
                dtype=self.energies.dtype,
                device=self.energies.device
            )

        n_max = self.energies.shape[1]*spin_mult
        E_min, E_max = self.energies.min(), self.energies.max()
        E_range = E_max - E_min

        E_level = E_guess if E_guess is not None \
            else (n/n_max)*E_range + E_min

        E_level = torch.clamp(E_level, E_min, E_max)

        for i in range(max_evals):
            n_eval = self.idos(
                E_level, 
                smearing=smearing,
                sigma=sigma,
                spin_mult=spin_mult,
                projections=projections)
            delta_E = (E_range)/2**(i+1)
            delta_E = torch.where(n_eval > n, -delta_E, delta_E)

            E_level = E_level + delta_E

        E_level = torch.clamp(E_level, E_min, E_max)

        return E_level
    
    def band_gap(
            self, 
            energy: torch.Tensor | float = 0.0, 
            sigma_cutoff: float = 0.0) -> torch.Tensor:
        
        if isinstance(energy, float):
            energy = torch.tensor(
                energy,
                dtype=self.energies.dtype,
                device=self.energies.device
            )

        _band_energies, _ = torch.sort(self.energies.flatten())
        i = torch.searchsorted(_band_energies, energy, right=True) - 1
        i_above = torch.clamp(i + 1, max=_band_energies.shape[0]-1)
        i_below = i.clamp(min=0)

        gaps = _band_energies[i_above] - _band_energies[i_below]
        gaps = torch.where(gaps <= sigma_cutoff, 0, gaps)

        return gaps

    
# add TorchBandStructure as safe to serialize
torch.serialization.add_safe_globals([TorchBandStructure])
