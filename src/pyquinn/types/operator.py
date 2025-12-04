import pickle
from abc import abstractmethod, ABC
from typing import Any

from pyquinn.types.backend import _load_backend_object
from pyquinn.types import AtomGraph

import numpy as np


SUPPORTED_BACKENDS = {
    'torch': 'pyquinn.torch.types.TorchOperator'
}
""" The supported backend Operator objects"""

class OperatorBase:
    """ Base class for Operator instances"""

class Operator(OperatorBase):
    """Represents an Operator expanded in a basis of local orbitals at each AtomGraph site.
    Also supports periodic structures, with interactions between unit cell images.
    """

    _backends = {}

    @staticmethod
    def register_backend(name: str):
        """Registers a
            :class:`~pyquinn.types.operator.Operator`
            subclass.

        This should be invoked as a decorator of the subclass. For example:


        .. code-block:: python

            @Operator.register_backend('my_backend_name')
            class MyBackendOperator(BackendOperator):
                ...

        Args:
            name (str): BackendOperator class to be registered.
        """

        def decorator(subclass):
            Operator._backends[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def from_matrix(
            matrix: list[list[float]] | np.ndarray, 
            site_indices: list[float], 
            orbital_indices: list[float]):
        """Constructs an Operator from a matrix indexed site and orbital indices.

        Args:
            matrix (list[list[float]] | np.ndarray): A Matrix representation of an operator (shape: [N,N])
            site_indices (list[float]): Indices corresponding to AtomGraph sites (shape: [N])
            orbital_indices (list[float]): Orbital indices corresponding to each site's orbitals, 
                starting at 0 for each site. (shape: N)

        Returns:
            Operator: The created Operator
        """
        
        onsite_values = np.diagonal(matrix)
        i, j = np.triu(matrix, 1).nonzero()

        coupling_values = matrix[i][j]
        couplings = np.hstack((i, j))

        return Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
        )

    def load(filename: str) -> 'Operator':
        """Helper function to load an Operator
        from a pickle file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to load.

        Returns:
            Operator: The loaded Operator.
        """
        with open(filename, 'rb') as f:
            op = pickle.load(f)

        if not isinstance(op, Operator):
            raise IOError(
                f'Expected an instance of Operator; '
                f'got: {type(op)}')

    def __init__(
            self,
            onsite_values: list[float],
            couplings: list[tuple[int, int]],
            coupling_values: list[complex],
            site_indices: list[float], 
            orbital_indices: list[float],
            images: list[list[tuple[int, int, int]]] | None = None):

        self.onsite_values = np.array(onsite_values, dtype=float)
        self.couplings = np.array(couplings, dtype=int)
        self.coupling_values = np.array(coupling_values, dtype=complex)
        self.site_indices = np.array(site_indices, dtype=int)
        self.orbital_indices = np.array(orbital_indices, dtype=int)
        
        if images is not None:
            images = np.array(images, dtype=int)

        self.images = images
        
    def matrix(
            self,
            k_pts: list[list[float]] | None = None,
            frac_positions: list[list[float]] | None = None,
            graph: AtomGraph | None = None):
        
        np_k_pts = np.zeros((1, 3)) if k_pts is None \
            else np.array(k_pts)
        
        N = self.onsite_values.shape[0]
        N_k = np_k_pts.shape[0]

        H_k = np.zeros((N_k, N, N), dtype=complex)
        diag_idx = np.arange(N)
        H_k[:, diag_idx, diag_idx] += self.onsite_values
        
        np_images = np.zeros((1, 3)) if self.images is None \
            else self.images

        np_frac_distances = 0

        if graph is not None:
            frac_positions = graph.positions @ np.linalg.pinv(graph.unit_cell)
        
        if frac_positions is not None:
            frac_positions = np.array(frac_positions)
            np_frac_distances = frac_positions[self.site_indices[self.couplings[:, 1]]] \
                - frac_positions[self.site_indices[self.couplings[:, 0]]]

        for i, k in enumerate(np_k_pts):
            z_k = self.coupling_values * np.exp(2.j*np.pi*((np_frac_distances + np_images)@k))
            np.add.at(H_k[i], (self.couplings[:,0], self.couplings[:,1]), z_k)
    
        return H_k

    def save(self, filename: str):
        """Helper function to save an Operator to a pickle
        file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to save.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def compile(
            self,
            backend: str = 'torch',
            **backend_kwargs):

        if backend not in Operator._backends:
            if backend in SUPPORTED_BACKENDS:
                path = SUPPORTED_BACKENDS[backend]
                Operator._backends[backend] = \
                    _load_backend_object(backend, path)
            else:
                raise ValueError(f'Unknown Backend Type: {backend}')

        return Operator._backends[backend].compile(
            operator=self,
            **backend_kwargs
        )

    def extract(self) -> 'Operator':
        """Extracts an Operator from its backend, returning a new Operator.
        Since a regular Operator has no associated backend, this method 
        simply returns the calling Operator object.

        Returns:
            Operator: The extracted Operator.
        """
        return self


class BackendOperator(ABC, OperatorBase):

    def __init__(
            self,
            onsite_values,
            couplings,
            coupling_values,
            site_indices, 
            orbital_indices,
            images):
        
        self.onsite_values = onsite_values
        self.couplings = couplings
        self.coupling_values = coupling_values
        self.site_indices = site_indices
        self.orbital_indices = orbital_indices
        self.images = images


    @staticmethod
    @abstractmethod
    def compile(
            operator: Operator,
            **backend_kwargs) -> 'BackendOperator':
        pass

    @abstractmethod
    def extract(self) -> Operator:
        pass