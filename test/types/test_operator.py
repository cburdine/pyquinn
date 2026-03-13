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

    def test_operator_entry_shapes(self):

        N = 3
        value_shape = (2,)

        onsite_values = np.array([
            [1.0, 0.5],
            [2.0, -0.5],
            [3.0, 1.5],
        ])

        couplings = np.array([
            [0, 1],
            [1, 2],
            [2, 0],
        ])

        coupling_values = np.array([
            [0.1, -0.2],
            [0.3, 0.4],
            [0.5, -0.6],
        ])

        site_indices = np.arange(N)
        orbital_indices = np.zeros(N)

        operator = Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
        )

        self.assertTrue(np.allclose(operator.onsite_values, onsite_values))
        self.assertTrue(np.allclose(operator.coupling_values, coupling_values))
        self.assertTrue(operator.onsite_values.shape[1:] == value_shape)
        self.assertTrue(operator.coupling_values.shape[1:] == value_shape)

        # matrix shape and values for tensor-valued entries
        H_k = operator.matrix()
        self.assertEqual(H_k.shape, (1, N, N) + value_shape)

        expected = np.zeros((N, N) + value_shape, dtype=coupling_values.dtype)
        diag_idx = np.arange(N)
        expected[diag_idx, diag_idx] += onsite_values
        for (i, j), v in zip(couplings, coupling_values):
            expected[i, j] += v

        self.assertTrue(np.allclose(H_k[0], expected))

        # trim uses value norm for tensor entries
        coupling_values_trim = coupling_values.copy()
        coupling_values_trim[1] = np.array([1e-12, 0.0])
        coupling_values_trim[2] = np.array([1e-9, 0.0])

        operator_trim = Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values_trim,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
        )

        operator_trim.trim(tol=1e-10)
        self.assertEqual(operator_trim.couplings.shape[0], 2)

        # from_matrix with tensor-valued entries
        matrix = np.zeros((N, N) + value_shape, dtype=coupling_values.dtype)
        matrix[0, 0] = onsite_values[0]
        matrix[1, 1] = onsite_values[1]
        matrix[2, 2] = onsite_values[2]
        matrix[0, 1] = coupling_values[0]
        matrix[1, 2] = coupling_values[1]
        matrix[0, 2] = np.array([0.7, -0.8])

        op_from_matrix = Operator.from_matrix(
            matrix=matrix,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
        )

        self.assertTrue(np.allclose(op_from_matrix.onsite_values, onsite_values))
        self.assertEqual(op_from_matrix.couplings.shape[1], 2)
        self.assertTrue(op_from_matrix.coupling_values.shape[1:] == value_shape)

    def test_matrix_accumulates_duplicate_couplings(self):

        N = 3
        onsite_values = np.zeros((N, 2))
        site_indices = np.arange(N)
        orbital_indices = np.zeros(N)

        couplings = np.array([
            [0, 1],
            [0, 1],
            [1, 2],
            [1, 2],
        ])

        coupling_values = np.array([
            [0.5, -0.5],
            [1.5, 0.5],
            [2.0, 0.0],
            [-1.0, 1.0],
        ])

        operator = Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
        )

        H_k = operator.matrix()

        expected = np.zeros((N, N, 2), dtype=coupling_values.dtype)
        expected[0, 1] += coupling_values[0] + coupling_values[1]
        expected[1, 2] += coupling_values[2] + coupling_values[3]

        self.assertTrue(np.allclose(H_k[0], expected))
