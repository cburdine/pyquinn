import unittest
import numpy as np

from pyquinn.util.data import ATOMIC_SYMBOLS
from pyquinn.types.operator import Operator, SUPPORTED_BACKENDS

class OperatorTest(unittest.TestCase):

    def test_operator(self):

        N = 100
        onsite_values = np.linspace(1.0,10.0,N)
        couplings = np.concatenste([
            np.hstack([
                np.arange(0,N) % N,
                np.arange(1,N+1) % N
            ]),
            np.hstack([
                np.arange(1,N+1) % N,
                np.arange()
            ])

    # TODO


