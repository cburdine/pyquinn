import unittest

import numpy as np
import torch

from pyquinn.torch.nn import AtomwiseNNConv


class AtomwiseTest(unittest.TestCase):
    def test_atomwise_nnconv(self):

        atomic_numbers = torch.tensor([6, 8, 8])

        neighbors = torch.tensor(
            [
                [1, 2],
                [0, -1],
                [0, -1],
            ]
        )
        neighbor_distances = torch.tensor(
            [
                [[0, 0, 1], [0, 0, -1]],
                [[0, 0, -1], [0, 0, 0]],
                [[0, 0, 1], [0, 0, 0]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )

        features = torch.tensor([[0.0, 1.0, 0.0], [2.0, 3.0, 4.0], [2.0, 1.0, 0.0]])

        model = AtomwiseNNConv(out_features=10, in_features=3, layers=3, z_max=15)

        output = model(
            atomic_numbers=atomic_numbers,
            neighbors=neighbors,
            neighbor_distances=neighbor_distances,
            features=features,
        )

        self.assertEqual(output.shape, (3, 10))

        output.sum().backward(inputs=neighbor_distances)
        distances_grad = neighbor_distances.grad.detach().cpu().numpy()

        self.assertFalse(np.allclose(distances_grad, np.zeros_like(distances_grad)))
