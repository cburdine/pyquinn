import unittest
import warnings

import numpy as np
import torch

from pyquinn.torch.eigensolver import (
    Eigensolver, SUPPORTED_EIGENSOLVERS,
    EighEigensolver,
    LobpcgEigensolver
)


def _make_tridiag(diag, gamma=0.5):
    D = np.array(diag).astype(np.complex128)
    M_np = np.diag(D)
    M_np += np.diag(1.j*gamma*np.diff(diag), 1)
    M_np += np.diag(-1.j*gamma*np.diff(diag), -1)

    return torch.tensor(M_np, requires_grad=False)


class EigensolverTest(unittest.TestCase):

    def test_from_name(self):

        solvers = [
            Eigensolver.from_name(name)
            for name in SUPPORTED_EIGENSOLVERS
        ]

        self.assertTrue(len(solvers) > 0)

        self.assertRaises(
            ValueError, 
            lambda: Eigensolver.from_name('unknown_eigensolver')
        )

    def test_eigh_eigensolver(self):

        solver = EighEigensolver()

        H = _make_tridiag(np.linspace(0, 5, 10))

        np_eigvals, np_eigvects = np.linalg.eigh(H.detach().numpy())

        # check forward pass
        forward_eigvals = solver(H)
        self.assertTrue(np.allclose(forward_eigvals.detach().numpy(), 
                                    np_eigvals, atol=1e-5))

        # check diagonalize() function
        eigvals, eigvects = solver.diagonalize(H)
        self.assertTrue(np.allclose(eigvals.detach().numpy(),
                                    np_eigvals, atol=1e-5))

        self.assertTrue(np.allclose(np.abs(eigvects.detach().numpy()**2),
                                    np.abs(np_eigvects**2), atol=1e-5))

    def test_lobpcg_eigensolver(self):

        # note: lobpcg API is subject to change (throws warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = LobpcgEigensolver()

        H = torch.abs(_make_tridiag(np.linspace(3, 5, 20)))
        n_eigvals = 5

        np_eigvals, np_eigvects = np.linalg.eigh(H.detach().numpy())

        # check forward pass
        forward_eigvals = solver(H, n=n_eigvals)

        self.assertTrue(np.allclose(forward_eigvals.detach().numpy(), 
                                    np_eigvals[:n_eigvals], atol=1e-5))

        # check diagonalize() function
        eigvals, eigvects = solver.diagonalize(H, n=n_eigvals)
        self.assertTrue(np.allclose(eigvals.detach().numpy(), 
                                    np_eigvals[:n_eigvals], atol=1e-5))

        self.assertTrue(np.allclose(np.abs(eigvects.detach().numpy()**2), 
                                    np.abs(np_eigvects[:, :n_eigvals]**2), 
                                    atol=1e-5))

        



