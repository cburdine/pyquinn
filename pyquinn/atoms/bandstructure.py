import pickle
from abc import abstractmethod, ABC

import numpy as np


SUPPORTED_BANDSTRUCTURE_BACKENDS = []
""" The full list of bandstructure backends 
    supported by default"""


class BandStructureBase:
    """Base class for BandStructure instances
    """


class BandStructure(BandStructureBase):

    _backends = {}

    @staticmethod
    def register_backend(name: str):

        def decorator(subclass):
            if name not in BandStructure._backends:
                SUPPORTED_BANDSTRUCTURE_BACKENDS.append(name)
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
            raise ValueError(f'Unknown Backend Type: {backend}')

        return BandStructure._backends[backend].compile(
            bands=self,
            max_k_pts=max_k_pts,
            max_bands=max_bands,
            **backend_kwargs)

    def resample_bands(
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
import pyquinn.torch.bandstructure # noqa
