import unittest

import numpy as np
import torch

from pyquinn.torch.types.operator import build_matrix, build_nonperiodic_matrix
from pyquinn.types.operator import Operator


class TorchOperatorTest(unittest.TestCase):
    def test_build_nonperiodic_matrix_leading_batch(self):
        onsite_values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
            ]
        )
        couplings = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.int64,
        )
        coupling_values = torch.tensor(
            [
                [0.5, 1.0, 1.5],
                [5.0, 10.0, 15.0],
            ]
        )

        H = build_nonperiodic_matrix(onsite_values, couplings, coupling_values)
        self.assertEqual(H.shape, (2, 3, 3))

        expected = torch.zeros((2, 3, 3), dtype=H.dtype)
        expected[:, torch.arange(3), torch.arange(3)] = onsite_values
        expected[:, 0, 1] += coupling_values[:, 0]
        expected[:, 1, 2] += coupling_values[:, 1]
        expected[:, 2, 0] += coupling_values[:, 2]

        self.assertTrue(torch.allclose(H, expected))

    def test_build_matrix_leading_batch(self):
        onsite_values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.complex64,
        )
        couplings = torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.int64,
        )
        coupling_values = torch.tensor(
            [
                [0.5, 0.25],
                [1.5, 0.75],
            ],
            dtype=torch.complex64,
        )

        site_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        images = torch.zeros((2, 3), dtype=torch.int64)
        k_pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        frac_positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        H_k = build_matrix(
            onsite_values,
            couplings,
            coupling_values,
            site_indices,
            images,
            k_pts,
            frac_positions,
        )

        self.assertEqual(H_k.shape, (3, 2, 3, 3))

        H0 = build_nonperiodic_matrix(onsite_values, couplings, coupling_values)
        self.assertTrue(torch.allclose(H_k[0], H0))

    def test_compile_extract_preserves_operator_layout(self):
        onsite_values = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        couplings = np.array(
            [
                [0, 1],
                [1, 2],
            ]
        )
        coupling_values = np.array(
            [
                [0.1, 1.0],
                [0.2, 2.0],
            ]
        )

        operator = Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=np.array([0, 1, 2]),
            orbital_indices=np.array([0, 0, 0]),
        )

        torch_operator = operator.compile(backend="torch", dtype=torch.float32)
        self.assertEqual(tuple(torch_operator.onsite_values.shape), (2, 3))
        self.assertEqual(tuple(torch_operator.coupling_values.shape), (2, 2))

        extracted = torch_operator.extract()
        self.assertTrue(np.allclose(extracted.onsite_values, onsite_values))
        self.assertTrue(np.allclose(extracted.coupling_values, coupling_values))

    def test_build_matrix_sparse_k_before_batch(self):
        onsite_values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.complex64,
        )
        couplings = torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.int64,
        )
        coupling_values = torch.tensor(
            [
                [0.5, 0.25],
                [1.5, 0.75],
            ],
            dtype=torch.complex64,
        )

        site_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        images = torch.zeros((2, 3), dtype=torch.int64)
        k_pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        frac_positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        H_dense = build_matrix(
            onsite_values,
            couplings,
            coupling_values,
            site_indices,
            images,
            k_pts,
            frac_positions,
            sparse=False,
        )

        H_sparse = build_matrix(
            onsite_values,
            couplings,
            coupling_values,
            site_indices,
            images,
            k_pts,
            frac_positions,
            sparse=True,
        )

        self.assertTrue(H_sparse.is_sparse)
        self.assertEqual(tuple(H_sparse.shape), (3, 3, 3, 2))

        H_sparse_dense = H_sparse.to_dense().permute(2, 3, 0, 1)
        self.assertTrue(torch.allclose(H_dense, H_sparse_dense))


if __name__ == "__main__":
    unittest.main()
