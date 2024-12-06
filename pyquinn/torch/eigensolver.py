import torch
import warnings
from abc import abstractmethod
from typing import TypeVar


import torch.nn as nn


Self = TypeVar('Self')

SUPPORTED_EIGENSOLVERS = []
""" The full list of supported eigensolvers"""


class Eigensolver(nn.Module):
    """ Abstract Base class for PyQuinn Eigensolvers  """

    _solvers = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def register(cls, name: str):
        """Registers an eigensolver subclass of this abstract class. You can
        add a custom eigensolver by using the python decorator syntax as
        follows:

        .. code-block:: python

            from pyquinn.torch.eigensolver import Eigensolver

            @Eigensolver.register('my_eigensolver_name')
            class MyEigensolver(Eigensolver):

                def diagonalize(...):
                    ...

        The registered eigensolver will be added to 
        :attr:`SUPPORTED_EIGENSOLVERS
        <pyquinn.torch.eigensolver.SUPPORTED_EIGENSOLVERS>`

        Args:
            name (str): Eigensolver class to be registered.
        """
        def decorator(subclass):
            if name not in cls._solvers:
                SUPPORTED_EIGENSOLVERS.append(name)
            cls._solvers[name] = subclass
            return subclass
        return decorator

    @classmethod
    def from_name(cls, solver_name: str, solver_kwargs: dict = {}) -> Self:
        """Loads an eigensolver with the given name from the set of 
        supported eigensolvers in :attr:`SUPPORTED_EIGENSOLVERS
        <pyquinn.torch.eigensolver.SUPPORTED_EIGENSOLVERS>`


        Args:
            solver_name (str): Name of eigensolver
            solver_kwargs (dict, optional): Keyword args passed to eigensolver. 
                                            Defaults to {}.

        Raises:
            ValueError: If eigensolver name is invalid

        Returns:
            Eigensolver: Eigensolver with given name
        """
        if solver_name in cls._solvers:
            return cls._solvers[solver_name](**solver_kwargs)
        else:
            raise ValueError(f'Unknown Eigensolver Type: {solver_name}')

    @abstractmethod
    def diagonalize(self, M: torch.Tensor,
                    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "This method should be implemented in a "
            "subclass. To create a subclass instance, "
            "use Eigensolver.from_name('<solver_name>').")

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        return self.diagonalize()[0]


@Eigensolver.register('torch_eigh')
class EighEigensolver(Eigensolver):
    """ Dense Matrix Eigensolver that uses `pytorch.linalg.eigh` """

    def __init__(self):
        """Creates an instance of an EighEigensolver"""
        super().__init__()

    def diagonalize(self, H: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagonalizes a dense Hermitian matrix (:math:`H = (H^*)^{T}`). 
           Supports backward passes for gradients with respect to 
           eigenvectors and eigenvalues.

        .. warning::
            Gradients with respect to eigenvectors are only uniquely defined
            up to a phase. It is recommended that you define loss functions
            using the modulus or modulus squared of eigenvalues.

            Also, gradients with respect to eigenvectors can be unstable for
            eigenvalues that are degenerate or nearly degenerate. Due to the
            non-uniqueness of degenerate eigenvectors, the eigenvectors
            returned may be different on different systems.

            See `torch.linalg.eigh` for more information.

        Args:
            H (torch.Tensor): Hermitian matrix (shape: [..., N, N])

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Eigenvalues and Eigenvectors
            of the matrix H.
            (shape: [..., N], [..., N, N])
        """

        return torch.linalg.eigh(H)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return torch.linalg.eigvalsh(H)


@Eigensolver.register('torch_lobpcg')
class LobpcgEigensolver(Eigensolver):

    def __init__(self, solver_kwargs: dict = {}):
        super().__init__()
        self.solver_kwargs = solver_kwargs

        if not hasattr(self.__class__, '_initialized'):
            warnings.warn("The Pytorch API for LOBPCG diagonalization is "
                          "still under development. Backward passes and "
                          "complex Hamiltonians are not yet fully supported.", 
                          FutureWarning, stacklevel=2)
            self.__class__._initialized = True

    def diagonalize(self, 
                    H: torch.Tensor, 
                    E_min: torch.Tensor = torch.tensor(0.0), 
                    n: torch.Tensor = torch.tensor(1),
                    S: torch.Tensor | None = None,
                    C: torch.Tensor | None = None):

        H_posdef = H - E_min*torch.eye(H.shape[-1])

        eigvals, eigvects = torch.lobpcg(A=H_posdef, k=n, B=S, X=C,
                                         largest=False,
                                         **self.solver_kwargs)
        eigvals += E_min

        return eigvals, eigvects 

    def forward(self, 
                H: torch.Tensor,
                E_min: torch.Tensor = torch.tensor(0.0),
                n: torch.Tensor = torch.tensor(1),
                S: torch.Tensor | None = None,
                C: torch.Tensor | None = None):

        eigvals, _ = self.diagonalize(H=H, E_min=E_min, n=n, S=S, C=C)

        return eigvals

