import unittest

import torch

from pyquinn.torch.analysis.negf import impact_ionization_self_energy


def _explicit_ii_kernel_reference(g_min: torch.Tensor, g_maj: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the 3-particle kernel in `impact_ionization_self_energy`.

    Matches the code-path:
      - zero-pad by 2*n_E on the *right* along energy
      - FFT over the padded energy axis
      - frequency-domain product: conj(F_min) @ F_maj @ F_maj
      - IFFT and keep the first n_E energies

    Implemented as time-domain circular convolution/correlation on the padded grid
    to validate the index conventions without using FFT.
    """

    if g_min.shape != g_maj.shape:
        raise ValueError("Expected g_min and g_maj to have the same shape.")
    if g_min.ndim < 3:
        raise ValueError("Expected tensors shaped like [..., E, N, N].")

    n_E = g_min.shape[-3]
    L = 3 * n_E

    pad = (0, 0, 0, 0, 0, 2 * n_E)
    a = torch.nn.functional.pad(g_min, pad)
    b = torch.nn.functional.pad(g_maj, pad)

    # d[t] = sum_p b[p] @ b[t - p]  (circular convolution, length L)
    d = torch.zeros_like(b)
    for t in range(L):
        acc = torch.zeros_like(b[..., 0, :, :])
        for p in range(L):
            acc = acc + (b[..., p, :, :] @ b[..., (t - p) % L, :, :])
        d[..., t, :, :] = acc

    # c[n] = sum_m conj(a[m]) @ d[m + n]  (circular cross-correlation, length L)
    c = torch.zeros_like(a)
    for n in range(L):
        acc = torch.zeros_like(a[..., 0, :, :])
        for m in range(L):
            acc = acc + (a[..., m, :, :].conj() @ d[..., (m + n) % L, :, :])
        c[..., n, :, :] = acc

    return c[..., :n_E, :, :]


class ImpactIonizationKernelTest(unittest.TestCase):
    def test_kernel_matches_explicit_reference(self):
        torch.manual_seed(0)
        dtype = torch.complex64
        device = torch.device("cpu")

        E = 8
        N = 3
        g_min = torch.randn((E, N, N), dtype=torch.float32, device=device) + 1j * torch.randn(
            (E, N, N), dtype=torch.float32, device=device
        )
        g_maj = torch.randn((E, N, N), dtype=torch.float32, device=device) + 1j * torch.randn(
            (E, N, N), dtype=torch.float32, device=device
        )
        g_min = g_min.to(dtype)
        g_maj = g_maj.to(dtype)

        D = 0.7
        dE = 0.25

        got = impact_ionization_self_energy(g_min=g_min, g_maj=g_maj, D=D, dE=dE)
        ref = D * _explicit_ii_kernel_reference(g_min=g_min, g_maj=g_maj) * (dE**2)

        self.assertTrue(torch.allclose(got, ref, atol=2e-5, rtol=2e-5))

    def test_kernel_delta_function_energy_mapping(self):
        # Single-delta test to validate the energy-index convention in the kernel:
        # If g_maj is a delta at p0, then (g_maj * g_maj)[t] is a delta at 2*p0.
        # The final output is a circular correlation with conj(g_min), so a delta at m0
        # shifts it to n = 2*p0 - m0.
        dtype = torch.complex64
        device = torch.device("cpu")

        E = 8
        N = 2
        D = 1.0
        dE = 0.5

        m0 = 1
        p0 = 2
        expected_n = 2 * p0 - m0  # = 3 (chosen to avoid modular wrap)

        g_min = torch.zeros((E, N, N), dtype=dtype, device=device)
        g_maj = torch.zeros((E, N, N), dtype=dtype, device=device)

        A = (0.3 + 0.1j) * torch.eye(N, dtype=dtype, device=device)
        B = (0.2 - 0.4j) * torch.eye(N, dtype=dtype, device=device)

        g_min[m0] = A
        g_maj[p0] = B

        sigma = impact_ionization_self_energy(g_min=g_min, g_maj=g_maj, D=D, dE=dE)

        expected = torch.zeros_like(sigma)
        expected[expected_n] = (A.conj() @ B @ B) * (dE**2)

        self.assertTrue(torch.allclose(sigma, expected, atol=1e-6, rtol=1e-6))

