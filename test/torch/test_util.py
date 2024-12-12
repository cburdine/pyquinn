import unittest
import warnings

import numpy as np
import torch

from pyquinn.torch.util import (
    sparse_csr_block_diag,
    sparse_coo_block_diag
)


class TorchUtilTest(unittest.TestCase):

    def test_sparse_csr_block_diag(self):
        block_sizes = torch.tensor([ 
            (1, 2), (3, 5), (2, 0), (1, 2), (10, 7), 
            (0, 1), (4, 0)
        ])

        blocks = [
            torch.arange(1, b.prod(-1)+1).reshape((b[0], b[1]))
            for b in block_sizes
        ]

        sparse_blocks = [
            block.to_sparse_csr()
            for block in blocks
        ]

        true_block_tensor = torch.block_diag(*blocks)

        # check if sparse block tensor is correct
        sparse_block_tensor = sparse_csr_block_diag(*blocks)
        self.assertTrue(np.allclose(
            true_block_tensor.detach().numpy(),
            sparse_block_tensor.to_dense().detach().numpy()
        ))

        # check if sparse block tensor constructed from
        # sparse blocks is correct
        sparse_block_tensor_2 = sparse_csr_block_diag(*sparse_blocks)
        self.assertTrue(np.allclose(
            true_block_tensor.detach().numpy(),
            sparse_block_tensor_2.to_dense().detach().numpy()
        ))

    def test_sparse_coo_block_diag(self):
        block_sizes = torch.tensor([ 
            (1, 2), (3, 5), (2, 0), (1, 2), (10, 7), 
            (0, 1), (4, 0)
        ])

        blocks = [
            torch.arange(1, b.prod(-1)+1).reshape((b[0], b[1]))
            for b in block_sizes
        ]

        sparse_blocks = [
            block.to_sparse_csr()
            for block in blocks
        ]

        true_block_tensor = torch.block_diag(*blocks)

        # check if sparse block tensor is correct
        sparse_block_tensor = sparse_coo_block_diag(*blocks)
        self.assertTrue(np.allclose(
            true_block_tensor.detach().numpy(),
            sparse_block_tensor.to_dense().detach().numpy()
        ))

        # check if sparse block tensor constructed from
        # sparse blocks is correct
        sparse_block_tensor_2 = sparse_coo_block_diag(*sparse_blocks)
        self.assertTrue(np.allclose(
            true_block_tensor.detach().numpy(),
            sparse_block_tensor_2.to_dense().detach().numpy()
        ))
