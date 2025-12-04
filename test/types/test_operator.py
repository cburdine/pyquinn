import unittest
import numpy as np

from pyquinn.util.data import ATOMIC_SYMBOLS
from pyquinn.types.operator import Operator, SUPPORTED_BACKENDS

class OperatorTest(unittest.TestCase):

    def test_operator(self):

        N = 100
        onsite_values = np.linspace(1.0, 1.0, N)
        couplings = np.concatenate([
            np.hstack([
                np.arange(0, N) % N,
                np.arange(1, N+1) % N
            ]),
            np.hstack([
                np.arange(1, N+1) % N,
                np.arange(0, N) % N
            ])
        ])
        coupling_values = np.ones(couplings.shape[0])


        images = np.zeros((2*N, 3))
        images[N-1, 0] = 1
        images[2*N-1, 0] = -1

        site_indices = np.arange(N)
        orbital_indices = np.zeros(N)

        operator = Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
            images=images
        )

        self.assertTrue(np.allclose(operator.onsite_values, onsite_values))
        self.assertTrue(np.allclose(operator.coupling_values, coupling_values))
        self.assertTrue(np.allclose(operator.site_indices, site_indices))
        self.assertTrue(np.allclose(operator.orbital_indices, orbital_indices))
        self.assertTrue(np.allclose(operator.images, images))

    def test_compile(self):

        N = 100
        onsite_values = np.linspace(1.0, 1.0, N)
        couplings = np.concatenate([
            np.hstack([
                np.arange(0, N) % N,
                np.arange(1, N+1) % N
            ]),
            np.hstack([
                np.arange(1, N+1) % N,
                np.arange(0, N) % N
            ])
        ])
        coupling_values = np.ones(couplings.shape[0])

        images = np.zeros((2*N, 3))
        images[N-1, 0] = 1
        images[2*N-1, 0] = -1

        site_indices = np.arange(N)
        orbital_indices = np.zeros(N)

        operator = Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
            images=images
        )

        self.assertTrue(SUPPORTED_BACKENDS)

        backend_operators = [
            operator.compile(backend=backend)
            for backend in SUPPORTED_BACKENDS
        ]

        self.assertTrue(len(backend_operators) > 0)

        self.assertRaises(
            ValueError,
            lambda: operator.compile(backend='unknown_backend')
        )

        extracted_operators = [
            op.extract()
            for op in backend_operators
        ]

        for op in extracted_operators:
            self.assertTrue(np.allclose(op.onsite_values, onsite_values))
            self.assertTrue(np.allclose(op.coupling_values, coupling_values))
            self.assertTrue(np.allclose(op.site_indices, site_indices))
            self.assertTrue(np.allclose(op.orbital_indices, orbital_indices))
            self.assertTrue(np.allclose(op.images, images))

