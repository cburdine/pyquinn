import pickle
from abc import abstractmethod, ABC

from pyquinn.types.backend import _load_backend_object

import numpy as np
from scipy.special import erf


SUPPORTED_BANDSTRUCTURE_BACKENDS = {
    'torch': 'pyquinn.torch.types.TorchBandStructure'
}
""" The supported backend BandStructure objects"""


class BandStructureBase:
    """Base class for BandStructure instances
    """


class BandStructure(BandStructureBase):

    _backends = {}

    _dos_smearing = [
        'gaussian',
        'fermi-dirac'
    ]

    _idos_smearing = [
        'gaussian',
        'fermi-dirac'
    ]

    @staticmethod
    def register_backend(name: str):

        def decorator(subclass):
            BandStructure._backends[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def load(filename: str) -> 'BandStructure':
        """Helper function to load a BandStructure 
        from a pickle file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to load.

        Returns:
            BandStructure: the loaded BandStructure.
        """
        with open(filename, 'rb') as f:
            bs = pickle.load(f)

        if not isinstance(bs, BandStructure):
            raise IOError(
                f'Expected an instance of BandStructure; '
                f'got: {type(bs)}')

        return bs

    def __init__(
            self,
            energies: list[list[float]],
            k_pts: list[list[float]] | None = None,
            projections: list[list[list[float]]] | None = None,
            chemical_potential: float | None = None):

        self.energies = energies

        self.k_pts = k_pts if k_pts is not None \
            else np.zeros((self.energies.shape[0], 0))

        self.projections = projections if projections is not None \
            else np.zeros(self.energies.shape + (0,))

        self.chemical_potential = chemical_potential if \
            chemical_potential is not None else np.array(0.0)

    @property
    def energies(self):
        return self._energies

    @energies.setter
    def energies(self, value):
        self._energies = np.array(value)

    @property
    def k_pts(self):
        return self._k_pts

    @k_pts.setter
    def k_pts(self, value):
        self._k_pts = np.array(value)

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, value):
        self._projections = np.array(value)

    @property
    def chemical_potential(self):
        return self._chemical_potential

    @chemical_potential.setter
    def chemical_potential(self, value):
        self._chemical_potential = np.array(value)

    def _validate_energies(self):
        if self.energies.shape[0] != self.k_pts.shape[0]:
            raise ValueError(
                f'Expected energies.shape[0] = k_pts.shape[0], '
                f'got: {self.energies.shape[0]} != {self.k_pts.shape[0]}')

    def save(self, filename: str):
        """Helper function to save a BandStructure to a pickle
        file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to save.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def compile(
            self,
            backend: str = 'torch',
            max_k_pts: int | None = None,
            max_bands: int | None = None,
            **backend_kwargs):
        
        self._validate_energies()

        if backend not in BandStructure._backends:
            if backend in SUPPORTED_BANDSTRUCTURE_BACKENDS:
                path = SUPPORTED_BANDSTRUCTURE_BACKENDS[backend]
                BandStructure._backends[backend] = \
                    _load_backend_object(backend, path)
            else:
                raise ValueError(f'Unknown Backend Type: {backend}')

        return BandStructure._backends[backend].compile(
            bands=self,
            max_k_pts=max_k_pts,
            max_bands=max_bands,
            **backend_kwargs)

    def downsample_bands(
            self, 
            max_k_pts: int | None = None,
            max_bands: int | None = None) -> 'BandStructure':

        energies = self.energies
        k_pts = self.k_pts
        projections = self.projections
        chemical_potential = self.chemical_potential

        # downsample k points (if applicable)
        if max_k_pts is not None:
            n_k_pts = self.energies.shape[0]
            sample_rate = n_k_pts // max_k_pts
            sample_idx = np.arange(0, min(n_k_pts, max_k_pts))*sample_rate

            energies = energies[sample_idx]
            k_pts = k_pts[sample_idx]
            projections = projections[sample_idx]

        # downsample energies (if applicable)
        if max_bands is not None:
            energies = energies[:, :min(energies.shape[1], max_bands)]
            projections = projections[:, :min(projections.shape[1], max_bands)]

        return BandStructure(
            energies=energies,
            k_pts=k_pts,
            projections=projections,
            chemical_potential=chemical_potential
        )

    def _gaussian_dos(
            self, 
            band_energies: np.ndarray,
            energy: np.ndarray, 
            sigma: float,
            projections: np.ndarray,
            weight: float) -> np.ndarray:
        
        # compute Gaussian dos
        unnorm_dos = np.sum(
            projections * np.exp(-((energy - band_energies)/sigma)**2), axis=1)
        dos = weight * unnorm_dos / (np.sqrt(np.pi) * sigma)

        return dos

    def _gaussian_idos(
            self, 
            band_energies: np.ndarray,
            energy: np.ndarray, 
            sigma: float,
            projections: np.ndarray,
            weight: float) -> np.ndarray:
        
        # compute Gaussian dos
        unnorm_dos = np.sum(
            projections * 0.5*(1.0 + erf((energy - band_energies)/sigma)), 
            axis=1)
        dos = weight * unnorm_dos

        return dos

    def _fermi_dirac_dos(
            self,
            band_energies: np.ndarray,
            energy: np.ndarray, 
            sigma: float,
            projections: np.ndarray | float,
            weight: float) -> np.ndarray:

        # compute Fermi-Dirac dos
        unnorm_dos = np.sum(
            projections / (1.0 + np.cosh((energy - band_energies)/sigma)), 
            axis=1)

        dos = weight * unnorm_dos / (2*sigma)
        return dos

    def _fermi_dirac_idos(
            self,
            band_energies: np.ndarray,
            energy: np.ndarray, 
            sigma: float,
            projections: np.ndarray | float,
            weight: float) -> np.ndarray:

        # compute Fermi-Dirac dos
        unnorm_dos = np.sum(
            projections * (
                0.5 + 0.5*np.tanh((energy - band_energies)/(2*sigma))), 
            axis=1)

        dos = weight * unnorm_dos
        return dos

    def _get_dos_data(
            self,
            energy: np.ndarray,  
            projections: np.ndarray | None = None):

        if projections is not None and not np.isscalar(projections):

            if projections.shape[:2] != self.energies.shape[:2]:
                raise ValueError(
                    'Expected first two dimensions of projections'
                    ' to match self.energies. (projections shape '
                    f'{projections.shape} incompatible with '
                    f'{self.energies.shape}.)')

            projections = projections.reshape(
                (1, self.energies.size) + projections.shape[2:])
            energy = energy.reshape((-1,) + (1,)*(projections.ndim-1))
            band_energies = self.energies.reshape(
                (1, -1)+(1,)*(projections.ndim-2))
        else:
            projections = np.array(1.0)
            energy = energy.reshape((-1, 1))
            band_energies = self.energies.reshape(1, -1)
        
        return band_energies, energy, projections

    def _default_sigma(self):
        E_min = self.energies.min()
        E_max = self.energies.max()
        N = self.energies.size
       
        sigma = (E_max - E_min)/np.sqrt(N)

        return sigma

    def dos(
            self,
            energy: np.ndarray | float = 0.0,
            smearing: str = 'gaussian',
            sigma: float | None = None,
            spin_mult: float = 1.0,
            projections: np.ndarray | None = None) -> np.ndarray:
        
        energy = np.array(energy)
        weight = np.array(spin_mult / self.k_pts.shape[0])
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
            energy: np.ndarray | float = 0.0,
            smearing: str = 'gaussian',
            sigma: float | None = None,
            spin_mult: float = 1.0,
            projections: np.ndarray | None = None) -> np.ndarray:
        
        energy = np.array(energy)
        weight = np.array(spin_mult / self.k_pts.shape[0])
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
            n: np.ndarray,
            smearing: str = 'fermi-dirac',
            sigma: float = None,
            spin_mult: float = 1.0,
            E_guess: np.ndarray | None = None, 
            max_evals: int = 64) -> np.ndarray:

        n_max = self.energies.shape[1]*spin_mult
        E_min, E_max = self.energies.min(), self.energies.max()
        E_range = E_max - E_min

        E_level = E_guess if E_guess is not None \
            else (n/n_max)*E_range + E_min

        for i in range(max_evals):
            n_eval = self.idos(
                E_level, 
                smearing=smearing,
                sigma=sigma,
                spin_mult=spin_mult)
            delta_E = (E_range)/2**(i+1)
            delta_E[n_eval > n] *= -1

            E_level += delta_E

        return E_level

    def band_gap(self, energy: np.ndarray | float = 0.0) -> np.ndarray:
        _band_energies = np.sort(self.energies.flatten())
        i = np.searchsorted(_band_energies, energy, side='right') - 1
        i_above = np.minimum(i + 1, _band_energies.size-1)

        gaps = _band_energies[i_above] - _band_energies[i]

        return gaps

    def extract(self) -> 'BandStructure':
        return self


class BackendBandStructure(ABC, BandStructureBase):

    def __init__(
            self,
            energies,
            k_pts,
            projections,
            chemical_potential):

        self.energies = energies
        self.k_pts = k_pts
        self.projections = projections
        self.chemical_potential = chemical_potential

    @staticmethod
    @abstractmethod
    def compile(
            bands: BandStructure,
            max_k_pts: int | None = None,
            max_bands: int | None = None,
            **backend_kwargs):
        pass

    @abstractmethod
    def extract(self) -> BandStructure:
        """ Returns an instance of a concrete AtomGraph
        which is removed from any backend-specific
        implementations. """
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str) -> 'BackendBandStructure':
        pass

    @abstractmethod
    def save(self, filename: str):
        pass


# import supported BackendBandStructures
#import pyquinn.torch.types.bandstructure # noqa
