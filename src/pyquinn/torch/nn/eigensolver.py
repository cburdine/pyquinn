import torch
import warnings
from abc import abstractmethod, ABC


import torch.nn as nn


class _StableEigh(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, A, eps):
        eigvals, eigvecs = torch.linalg.eigh(A)
        ctx.save_for_backward(eigvals, eigvecs)
        ctx.eps = eps
        return eigvals, eigvecs

    @staticmethod
    def backward(ctx, grad_eigvals, grad_eigvecs):
        eigvals, eigvecs = ctx.saved_tensors
        eps = ctx.eps

        V = eigvecs  # (..., N, N)
        VT = V.conj().transpose(-2, -1)
        dLdLambda = grad_eigvals      # (..., N)
        dLdV = grad_eigvecs           # (..., N, N)

        # 1. Compute symmetric part: V @ diag(dL/dLambda) @ V^T
        grad_lambda_term = V @ torch.diag_embed(dLdLambda).to(V.dtype) @ VT  # (..., N, N)

        # 2. Compute eigenvector term
        #    Compute F_ij = 1 / (λ_i - λ_j) with 0 on diagonal
        lambda_i = eigvals.unsqueeze(-1)  # (..., N, 1)
        lambda_j = eigvals.unsqueeze(-2)  # (..., 1, N)
        delta = lambda_i - lambda_j       # (..., N, N)

        # Avoid divide-by-zero on diagonal
        F = torch.where(delta.abs() > eps, 1.0 / delta, torch.zeros_like(delta))

        # Project grad onto eigenbasis
        Vt_dLdV = VT @ dLdV               # (..., N, N)

        # Apply symmetric correction
        F_prod = F * Vt_dLdV
        grad_v_term = V @ (F_prod + F_prod.conj().transpose(-2, -1)) * 0.5 @ VT  # (..., N, N)

        # Combine
        grad_input = grad_lambda_term + grad_v_term
        return grad_input, None  # Second arg is for eps


SUPPORTED_EIGENSOLVERS = []
"""The full list of eigensolvers supported by default."""


class Eigensolver(nn.Module, ABC):
    """ Abstract Base class for PyTorch Eigensolvers  """

    _solvers = {}

    def __init__(self):
        super().__init__()

    @staticmethod
    def register(name: str):
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
            if name not in Eigensolver._solvers:
                SUPPORTED_EIGENSOLVERS.append(name)
            Eigensolver._solvers[name] = subclass
            return subclass
        return decorator

    @staticmethod
    def from_name(
            solver_name: str, 
            solver_kwargs: dict | None = None) -> 'Eigensolver':
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
        if solver_name not in Eigensolver._solvers:
            raise ValueError(f'Unknown Eigensolver Type: {solver_name}')

        solver_kwargs = {} if solver_kwargs is None else solver_kwargs

        return Eigensolver._solvers[solver_name](**solver_kwargs)

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
    """ Dense Matrix Eigensolver that uses `pytorch.linalg.eigh` with
        added support for stabilizing gradients of degenerate eigenvalues."""

    def __init__(self, eps=1e-5):
        """Creates an instance of an EighEigensolver"""
        super().__init__()
        self.eps = eps

    def diagonalize(self, H: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagonalizes a dense Hermitian matrix (:math:`H = (H^*)^{T}`). 
        Supports backward passes for gradients with respect to 
        eigenvectors and eigenvalues.

        .. warning::
            Gradients with respect to eigenvectors are only uniquely defined
            up to a phase. It is recommended that you define loss functions
            using the modulus or modulus squared of each eigenvector component.

            See `torch.linalg.eigh` for more information.

        Args:
            H (torch.Tensor): Hermitian matrix (shape: [..., N, N])

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Eigenvalues and Eigenvectors
            of the matrix H. (shape: [..., N], [..., N, N])
        """
        if not hasattr(self, 'eps'):
            self.eps = 1e-5

        return _StableEigh.apply(H, self.eps)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Computes the Eigenvalues of a dense Hermitian matrix 
        (:math:`H = (H^*)^{T}`). Supports backward passes for 
        stable gradients with respect to eigenvalues.

        Args:
            H (torch.Tensor): Hermitian matrix (shape: [..., N, N])

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Eigenvalues of 
            the matrix H. (shape: [..., N])
        """

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
                          UserWarning, stacklevel=2)
            self.__class__._initialized = True

    def diagonalize(self, 
                    H: torch.Tensor, 
                    E_min: torch.Tensor = torch.tensor(0.0), 
                    n: torch.Tensor = torch.tensor(1),
                    S: torch.Tensor | None = None,
                    C: torch.Tensor | None = None) -> torch.Tensor:
        """Diagonalizes a dense symmetric matrix by solving a generalized 
           eigenvalue problem for the `n` least eigenvectors of a 
           real symmetric Hamiltonian :math:`H` with eigenvalues bounded
           below by some constant `E_min`. The generalized eigenvalue problem
           takes the form

        .. math::
            H C = SCE

        where :math:`C` is a matrix of eigenvalue coefficients, `E` is a 
        diagonal matrix containing the `n` least eigenvalues, 
        and :math:`S` is a basis overlap matrix (equal to the 
        identity by default).

        .. warning::
            This eigensolver uses the `torch.lobpcg` function, which is still 
            under development. Certain feature like backward passes for
            non-trivial :math:`S` matrices and complex Hamiltonians 
            are not yet fully supported. Use at your own risk.

        Args:
            H (torch.Tensor): Hermitian matrix (shape: [..., N, N])
            E_min (torch.Tensor, optional): Lower bound on eigenvalues of the 
                Hamiltonian. Defaults to torch.tensor(0.0).
            n (torch.Tensor, optional): Number of lowest eigenvalues to 
                solve for. Defaults to 1.
            S (torch.Tensor | None, optional): Overlap matrix :math:`S`.
                (shape: [..., N, N]). Defaults to None.
            C (torch.Tensor | None, optional): Initial guess of eigenvectors 
                (shape: [...,N,n]). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Eigenvalues and Eigenvectors
            of the matrix `H`. (shape: [..., n], [..., N, n])
        """

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
        """Diagonalizes a dense symmetric matrix by solving a generalized 
           eigenvalue problem for the `n` least eigenvalues of a 
           real symmetric Hamiltonian :math:`H` with the eigenvalues bounded
           below by some constant `E_min`. The generalized eigenvalue problem
           takes the form

        .. math::
            H C = SCE

        where :math:`C` is a matrix of eigenvalue coefficients, `E` is a 
        diagonal matrix containing the `n` least eigenvalues, 
        and :math:`S` is a basis overlap matrix (equal to the 
        identity by default).

        .. warning::
            This eigensolver uses the `torch.lobpcg` function, which is still 
            under development. Certain feature like backward passes for
            non-trivial :math:`S` matrices and complex Hamiltonians 
            are not yet fully supported. Use at your own risk.

        Args:
            H (torch.Tensor): Hermitian matrix (shape: [..., N, N])
            E_min (torch.Tensor, optional): Lower bound on eigenvalues of the 
                Hamiltonian. Defaults to torch.tensor(0.0).
            n (torch.Tensor, optional): Number of lowest eigenvalues to 
                solve for. Defaults to 1.
            S (torch.Tensor | None, optional): Overlap matrix :math:`S`.
                (shape: [..., N, N]). Defaults to None.
            C (torch.Tensor | None, optional): Initial guess of eigenvectors 
                (shape: [...,N,n]). Defaults to None.

        Returns:
            torch.tensor: Eigenvalues of the matrix `H`. (shape: [..., n])
        """
        
        eigvals, _ = self.diagonalize(H=H, E_min=E_min, n=n, S=S, C=C)

        return eigvals

