
from abc import abstractmethod

import torch.nn as nn
import torch

from pyquinn.torch.eigensolver import Eigensolver


class HamiltonianModel(nn.Module):
    """Abstract base class for a Pytorch Hamiltonian Model.
    This class represents a Hamiltonian of a quantum system 
    that is parameterized by a neural network.

    All subclasses of this class must at a minimum provide an
    implementation of

    `get_hamiltonian_matrix()`

    and 

    `get_hamiltonian_key()`

    The default behavior of the `forward()` method is to return 
    the eigenvalues of the quantum system, obtained through
    the diagonalization of the Hamiltonian matrix via a
    concrete instance of the `pyquinn.torch.Eigensolver` class.
    """

    def __init__(self, eigensolver: str | Eigensolver | None = None):

        super().__init__()
        self.eigensolver = Eigensolver.from_name(eigensolver) \
            if isinstance(eigensolver, str) else eigensolver

    @abstractmethod
    def hamiltonian_matrix(self, 
                           atomic_numbers: torch.IntTensor,
                           positions: torch.Tensor,
                           neighbors: torch.IntTensor
                           ) -> torch.Tensor:
        """Returns the Hamiltonian of an atomic system in matrix form.
        This abstract method provides the recommended signature of
        all subclasses that provide an implementation of this function.
        Note that some subclasses may change or add arguments to this 
        signature.

        Args:
            atomic_numbers (torch.IntTensor): 
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring 
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            NotImplementedError: If this method is called directly 
                                 (must be reimplemented in a subclass)

        Returns:
            torch.Tensor: Hamiltonian matrix (shape: [N_orbitals, N_orbitals])
        """
        raise NotImplementedError()

    @abstractmethod
    def hamiltonian_key(self, 
                        atomic_numbers: torch.IntTensor, 
                        positions: torch.Tensor, 
                        neighbors: torch.IntTensor
                        ) -> list[str]:
        """Returns a list of string labels for the entries of the Hamiltonian
           matrix. Entries in the Hamiltonian matrix are expected to follow the
           ordering of `atomic_numbers` and `neighbors`, though each site may
           contain a variable number of orbitals.

        Args:
            atomic_numbers (torch.IntTensor): 
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring 
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            NotImplementedError: If this method is called directly 
                                 (must be reimplemented in a subclass)

        Returns:
            list[str]: A list of string labels for the entries of the 
                       Hamiltonian matrix. Must have the same dimension 
                       as the matrix.
        """
        raise NotImplementedError()

    def eigenstates(self, 
                    atomic_numbers: torch.IntTensor, 
                    positions: torch.Tensor, 
                    neighbors: torch.IntTensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the eigenenergies and associated eigenstates of the
        Hamiltonian matrix.

        Args:
            atomic_numbers (torch.IntTensor): 
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring 
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            ValueError: If an Eigensolver is not supplied.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: Eigenenergies and associated 
                                              matrix of eigenstates.
                                              (shape: [N_eigstates], 
                                              [N_eigstates, N_orbitals]) 
        """

        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers, 
            positions=positions,
            neighbors=neighbors)

        if self.eigensolver is None:
            raise ValueError('An eigensolver must be supplied.')

        return self.eigensolver.diagonalize(H)

    def eigenenergies(self,
                      atomic_numbers: torch.IntTensor, 
                      positions: torch.Tensor, 
                      neighbors: torch.IntTensor
                      ) -> torch.Tensor:
        """Returns the eigenenergies of an atomic system's Hamiltonian.

        Args:
            atomic_numbers (torch.IntTensor): 
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring 
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            ValueError: If an Eigensolver is not supplied.

        Returns:
            torch.Tensor: Eigenenergies of the system (shape: [N_eigstates])
        """
        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers, 
            positions=positions,
            neighbors=neighbors)

        if self.eigensolver is None:
            raise ValueError('An eigensolver must be supplied.')

        return self.eigensolver(H)

    def forward(self,
                atomic_numbers: torch.IntTensor, 
                positions: torch.Tensor, 
                neighbors: torch.IntTensor
                ) -> torch.Tensor:
        """Forward pass function for a HamiltonianModel. Default behavior is
        to return the eigenenergies of an atomic system's Hamiltonian. In any
        concrete subclasses, this function should have a signature that 
        matches `self.hamiltonian_matrix()`.

        Args:
            atomic_numbers (torch.IntTensor): 
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring 
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Returns:
            torch.Tensor: Eigenenergies of the system (shape: [N_eigstates])
        """

        return self.eigenenergies(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors)
