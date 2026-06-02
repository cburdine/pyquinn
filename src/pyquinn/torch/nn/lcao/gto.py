from math import comb, factorial, gamma, sqrt

import torch

from pyquinn.torch.util import sparse_coo_block_diag

GTO_L_MAX = 4
r"""Maximum angular momentum :math:`l` supported by default for Gaussian-type
orbital (GTO) representations."""


def _make_flat_tensors(tensor_list: list):

    list_shapes = torch.vstack([torch.tensor(lt.shape) for lt in tensor_list])
    list_sizes = list_shapes.prod(-1)

    list_idxs = torch.cumsum(list_sizes, dim=0) - list_sizes
    list_flat = torch.cat([lt.flatten() for lt in tensor_list])

    return (list_idxs, list_shapes, list_flat)


def _get_flat_tensor(data, i):

    idx_data, shape_data, flat_data = data
    idx = idx_data[i]
    shape = shape_data[i]

    return flat_data[idx : idx + shape.prod()].view(*shape.unbind())


def _make_gto_ijk_coeffs(l_max):
    """Generate factorial-based normalization coefficients for Cartesian powers.

    Args:
        l_max (int): Maximum angular momentum :math:`l` (inclusive).

    Returns:
        torch.Tensor: 1D tensor `c` where `c[i] = i!/(2i)!` for `0 <= i <= l_max`.
    """
    return torch.tensor(
        [factorial(i) / factorial(2 * i) for i in range(l_max + 1)],
        requires_grad=False,
    )


_GTO_IJK_COEFFS = _make_gto_ijk_coeffs(GTO_L_MAX)
r"""Normalizing constant factors for Cartesian GTOs up to `l = GTO_L_MAX`."""


def _make_potential_ijk_coeffs(l_max):
    """Generate normalization coefficients for Cartesian Gaussian potentials.

    Args:
        l_max (int): Maximum angular momentum :math:`l` (inclusive).

    Returns:
        torch.Tensor: 1D tensor `c` where `c[i] = 1/Gamma((i+1)/2)` for
        `0 <= i <= l_max`.
    """
    return torch.tensor([1 / gamma((i + 1) / 2) for i in range(l_max + 1)])


_POTENTIAL_IJK_COEFFS = _make_potential_ijk_coeffs(GTO_L_MAX)
r"""Normalizing constant factors for Cartesian l1-normalized Gaussian potentials
up to `l = GTO_L_MAX`."""


def _make_gto_ijk_values(l):
    """Generate the ordered list of all `(i, j, k)` with `i + j + k = l`.

    Args:
        l (int): Angular momentum (total Cartesian power).

    Returns:
        torch.Tensor: Integer tensor of shape `[n, 3]` listing all triples.
    """

    return torch.tensor(
        [[l - (j + k), j, k] for k in range(l + 1) for j in range(l + 1 - k)],
        requires_grad=False,
    )


_GTO_IJK_VALUES = [_make_gto_ijk_values(l) for l in range(GTO_L_MAX + 1)]
r"""IJK values for each angular momentum `l` up to `l = GTO_L_MAX`."""


def _make_gto_hermite_tensor(l_max):
    """Generate a matrix of Hermite polynomial coefficients.

    Args:
        l_max (int): Maximum angular momentum :math:`l` (inclusive).

    Returns:
        torch.Tensor: Coefficient matrix `H` of shape `[n, n]` where
        `n = 2 * (l_max + 1)`.
    """
    n = 2 * (l_max + 1)
    H = torch.zeros((n, n), requires_grad=False)
    for j in range(n):
        for p in range(j // 2 + 1):
            H[j, j - 2 * p] = factorial(j) * (-1) ** p * 2 ** (j - 2 * p) / (factorial(p) * factorial(j - 2 * p))

    return H


_GTO_HERMITE_TENSOR = _make_gto_hermite_tensor(GTO_L_MAX)
r"""Matrix of Hermite polynomial coefficients up to order
`n = 2 * (GTO_L_MAX + 1)`."""


def _make_gto_alpha_tensor(l_max):
    """Generate a matrix of coefficients used for products of primitives.

    Args:
        l_max (int): Maximum angular momentum :math:`l` (inclusive).

    Returns:
        torch.Tensor: Coefficient matrix `A` of shape `[l_max+1, l_max+1]`.
    """
    n = l_max + 1
    A = torch.zeros((n, n), requires_grad=False)
    for l in range(n):
        norm_coeff = ((2) ** (1 / 2) * 8**l * factorial(l) / factorial(2 * l)) ** (1 / 2)
        for i in range(l // 2 + 1):
            A[l, l - 2 * i] = norm_coeff * factorial(l) / (2**l * factorial(i) * factorial(l - 2 * i))

    return A


_GTO_ALPHA_TENSOR = _make_gto_alpha_tensor(GTO_L_MAX)
r"""Matrix of coefficients up to `l = GTO_L_MAX`."""


def _make_gto_cho_tensor(l, ijks=None) -> torch.Tensor:
    r"""Generate the Cartesian-to-cubic-harmonic transformation matrix.

    The returned matrix `X` maps the basis of l2-normalized Cartesian GTOs
    with `i + j + k = l` to the basis of real cubic harmonics with the same
    angular momentum:

    .. math::

        \chi_{l m}(\mathbf{x}) = \sum_{i+j+k=l} X_{l m,\, i j k}\,\phi^{i j k}(\mathbf{x})

    Args:
        l (int): Angular momentum (total Cartesian power).
        ijks (torch.Tensor, optional): Ordered `(i, j, k)` values of shape
            `[n, 3]`. If `None`, uses the default ordering for the given `l`.

    Returns:
        torch.Tensor: Transformation matrix `X` of shape `[2l+1, n]`.
    """

    if ijks is None:
        ijks = _make_gto_ijk_values(l).detach().cpu().numpy()
    ijk_coeffs = _make_gto_ijk_coeffs(l)

    n_ijks = len(ijks)
    n_chos = 2 * l + 1
    X = torch.zeros((n_chos, n_ijks))

    ijk_idxs = {tuple(ijk): i for i, ijk in enumerate(ijks)}

    ijk_norm_coeffs = {
        tuple(ijk): (ijk_coeffs[torch.tensor(ijk)].prod(-1) * factorial(2 * l) / factorial(l)) ** (-1 / 2)
        for ijk in ijks
    }

    # generate unnormalized cubic harmonic matrix:
    for i, m_signed in enumerate(range(-l, l + 1)):
        m = abs(m_signed)
        gamma_0 = 1.0 if m == 0 else sqrt(2 * (factorial(l - m) / factorial(l + m)))
        for k in range((l - m) // 2 + 1):
            N_lmk = (comb(l, k) * comb(2 * l - 2 * k, l) * factorial(l - 2 * k)) / factorial(l + (-2 * k) + (-m))
            gamma_lmk = gamma_0 * N_lmk * ((-1) ** k / 2**l)

            for p in range(m + 1):
                if m_signed == 0:
                    sigma_lmk = comb(m, p)
                elif m_signed < 0:
                    # sigma_lmk equals: sin(torch.pi/2 * (m-p))*comb(m,p)
                    sigma_lmk = comb(m, p) * ((-1) ** ((m - p) // 2) - (-1) ** (((m - p) + 1) // 2)) // 2
                else:
                    # sigma_lmk equals: cos(torch.pi/2 * (m-p))*comb(m,p)
                    sigma_lmk = comb(m, p) * ((-1) ** ((m - p) // 2) + (-1) ** (((m - p) + 1) // 2)) // 2

                for s in range(k + 1):
                    for t in range(s + 1):
                        ijk = (
                            p + 2 * (k - s),  # i
                            (m - p) + 2 * (s - t),  # j
                            (l - 2 * k - m) + 2 * t,
                        )  # k

                        assert ijk in ijks
                        N_ks = comb(k, s) * comb(s, t)
                        norm_coeff = ijk_norm_coeffs[ijk]
                        X[i, ijk_idxs[ijk]] += gamma_lmk * sigma_lmk * norm_coeff * N_ks

    return X


_GTO_CHO_TENSORS = [_make_gto_cho_tensor(l, _GTO_IJK_VALUES[l].detach().cpu().numpy()) for l in range(GTO_L_MAX + 1)]
r"""List of matrices mapping Cartesian GTO bases to cubic harmonics for each
`l` up to `l = GTO_L_MAX`."""


def _make_gto_e3_cho_tensor(l, cho_tensor=None, ijks=None):

    if ijks is None:
        ijks = _make_gto_ijk_values(l)
    if cho_tensor is None:
        cho_tensor = _make_gto_cho_tensor(l, ijks)

    e3_ijk_blocks = []
    e3_cho_blocks = []

    for cho_row in cho_tensor:
        ijk_idx = cho_row.nonzero(as_tuple=True)
        e3_ijk_blocks.append(ijks[ijk_idx])
        e3_cho_blocks.append(cho_row[ijk_idx])

    e3_ijks = torch.vstack(e3_ijk_blocks)
    e3_cho_tensor = torch.block_diag(*e3_cho_blocks)

    return e3_cho_tensor, e3_ijks


_GTO_E3_CHO_TENSORS = []
_GTO_E3_IJK_VALUES = []

for l in range(GTO_L_MAX + 1):
    e3_cho_tensor, e3_ijks = _make_gto_e3_cho_tensor(l, _GTO_CHO_TENSORS[l], _GTO_IJK_VALUES[l])
    _GTO_E3_CHO_TENSORS.append(e3_cho_tensor)
    _GTO_E3_IJK_VALUES.append(e3_ijks)


_GTO_CHO_FLAT_TENSORS = _make_flat_tensors(_GTO_CHO_TENSORS)
_GTO_IJK_FLAT_TENSORS = _make_flat_tensors(_GTO_IJK_VALUES)


def gaussian_3d(x: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""Evaluate a 3D unnormalized Gaussian primitive function.

    .. math::

        f(\mathbf{x}) = \exp(-\beta(\mathbf{x}-\boldsymbol{\mu})^2)

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Gaussian center (shape: `[..., 3]`, broadcastable to
            `x`).
        beta (torch.Tensor): Gaussian exponent (shape: `[...]`, broadcastable to
            `x[..., 0]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[..., 1]`).
    """
    return torch.exp(-beta * torch.sum((x - mu) ** 2, dim=-1, keepdim=True))


def l1_gaussian_3d(x: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    r"""Evaluate a 3D l1-normalized Gaussian primitive function.

    .. math::

        f(\mathbf{x}) = \left(\frac{\gamma}{\pi}\right)^{3/2}
        \exp(-\gamma(\mathbf{x}-\boldsymbol{\mu})^2)

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Gaussian center (shape: `[..., 3]`, broadcastable to
            `x`).
        gamma (torch.Tensor): Gaussian exponent (shape: `[...]`, broadcastable
            to `x[..., 0]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[..., 1]`).
    """
    return (gamma / torch.pi) ** (3 / 2) * torch.exp(-gamma * torch.sum((x - mu) ** 2, dim=-1, keepdim=True))


def l2_gaussian_3d(x: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    r"""Evaluate a 3D l2-normalized Gaussian primitive function.

    .. math::

        f(\mathbf{x}) = \left(\frac{2\alpha}{\pi}\right)^{3/4}
        \exp(-\alpha(\mathbf{x}-\boldsymbol{\mu})^2)

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Gaussian center (shape: `[..., 3]`, broadcastable to
            `x`).
        alpha (torch.Tensor): Gaussian exponent (shape: `[...]`, broadcastable
            to `x[..., 0]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[..., 1]`).
    """
    return (2 * alpha / torch.pi) ** (3 / 4) * torch.exp(-alpha * torch.sum((x - mu) ** 2, dim=-1, keepdim=True))


def gto(x: torch.Tensor, mu: torch.Tensor, ijk: torch.IntTensor, alpha: torch.Tensor) -> torch.Tensor:
    r"""Evaluate an l2-normalized Cartesian Gaussian-type orbital (GTO).

    .. math::

        \phi_{\alpha}^{i j k}(\mathbf{x}) =
        N\,(x-\mu_x)^i\,(y-\mu_y)^j\,(z-\mu_z)^k\,
        \exp\!\left(-\alpha\,\lVert\mathbf{x}-\boldsymbol{\mu}\rVert^2\right)

    where :math:`N` is a normalization coefficient.

    A linear combination of these functions can be used to
    represent a cartesian Slater-type orbital (STO).

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Orbital center (shape: `[3]` or `[..., 3]`,
            broadcastable to `x`).
        ijk (torch.IntTensor): Cartesian powers `(i, j, k)` (shape: `[3]` or
            `[..., 3]`, broadcastable to `x`).
        alpha (torch.Tensor): Gaussian exponent (shape: `[...]` or scalar,
            broadcastable to `x[..., 0]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[..., 1]`).
    """

    dx = x - mu
    psi = (
        torch.exp(-alpha * ((dx) ** 2).sum(-1, keepdim=True))
        * (dx ** (ijk)).prod(-1, keepdim=True)
        * (2 * alpha / torch.pi) ** (3 / 4)
        * (8 * alpha) ** (ijk.sum(-1, keepdim=True) / 2)
        * torch.sqrt((_GTO_IJK_COEFFS[ijk]).prod(-1, keepdim=True))
    )

    return psi


def potential_gto(x: torch.Tensor, mu: torch.Tensor, ijk: torch.IntTensor, gamma: torch.Tensor) -> torch.Tensor:
    r"""Evaluate an l1-normalized Cartesian Gaussian potential basis function.

    .. math::

        F_{\gamma}^{i j k}(\mathbf{x}) =
        N\,(x-\mu_x)^i\,(y-\mu_y)^j\,(z-\mu_z)^k\,
        \exp\!\left(-\gamma\,\lVert\mathbf{x}-\boldsymbol{\mu}\rVert^2\right)

    where :math:`N` is a normalization coefficient.

    A linear combination of these functions can be
    used to represent an effective core potential.

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Potential center (shape: `[3]` or `[..., 3]`,
            broadcastable to `x`).
        ijk (torch.IntTensor): Cartesian powers `(i, j, k)` (shape: `[3]` or
            `[..., 3]`, broadcastable to `x`).
        gamma (torch.Tensor): Gaussian exponent (shape: `[...]` or scalar,
            broadcastable to `x[..., 0]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[..., 1]`).
    """

    dx = x - mu
    F = (
        torch.exp(-gamma * ((dx) ** 2).sum(-1, keepdim=True))
        * (dx ** (ijk)).prod(-1, keepdim=True)
        * (gamma) ** ((ijk.sum(-1, keepdim=True) + 3) / 2)
        * (_POTENTIAL_IJK_COEFFS[ijk]).prod(-1, keepdim=True)
    )

    return F


def sto_ng(
    x: torch.Tensor,
    mu: torch.Tensor,
    ijk: torch.IntTensor,
    alphas: torch.Tensor,
    cs: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate a contracted Cartesian orbital as a Gaussian contraction.

    .. math::

        \phi^{i j k}(\mathbf{x}) = \sum_{n=1}^{N} c_n\,\phi_{\alpha_n}^{i j k}(\mathbf{x})

    The contraction coefficients `cs` are used as-is (they are not internally
    normalized).

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Orbital center (shape: `[3]` or `[..., 3]`,
            broadcastable to `x`).
        ijk (torch.IntTensor): Cartesian powers `(i, j, k)` (shape: `[3]` or
            `[..., 3]`, broadcastable to `x`).
        alphas (torch.Tensor): Primitive Gaussian exponents (shape: `[N]`).
        cs (torch.Tensor): Primitive contraction coefficients (shape: `[N]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[...]`).
    """

    gto_3d_component = torch.vmap(lambda a, c: c * gto(x, mu, ijk, a), in_dims=-1, out_dims=-1)

    psi = torch.sum(gto_3d_component(alphas, cs), axis=-1).squeeze(-1)

    return psi


def potential_ng(
    x: torch.Tensor,
    mu: torch.Tensor,
    ijk: torch.IntTensor,
    gammas: torch.Tensor,
    vs: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate a contracted Cartesian effective potential function.

    .. math::

        V^{i j k}(\mathbf{x}) = \sum_{n=1}^{N} v_n\,F_{\gamma_n}^{i j k}(\mathbf{x})

    The contraction coefficients `vs` are used as-is (they are not internally
    normalized).

    For a radially symmetric potential, the value of
    (i,j,k) = (0,0,0) should be used.

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Potential center (shape: `[3]` or `[..., 3]`,
            broadcastable to `x`).
        ijk (torch.IntTensor): Cartesian powers `(i, j, k)` (shape: `[3]` or
            `[..., 3]`, broadcastable to `x`).
        gammas (torch.Tensor): Primitive potential Gaussian exponents
            (shape: `[N]`).
        vs (torch.Tensor): Primitive potential contraction coefficients
            (shape: `[N]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[...]`).
    """

    v_3d_component = torch.vmap(lambda g, v: v * potential_gto(x, mu, ijk, g), in_dims=-1, out_dims=-1)

    V = torch.sum(v_3d_component(gammas, vs), axis=-1).squeeze(-1)

    return V


def sto_ng_primitive_overlap_matrix(
    mu_1: torch.Tensor,
    alphas_1: torch.Tensor,
    mu_2: torch.Tensor,
    alphas_2: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate primitive 1D overlap tensors for l2-normalized Gaussian primitives.

    This routine computes the per-dimension primitive overlap tensor used to
    build Cartesian overlaps between primitives centered at `mu_1` and `mu_2`.
    It is used internally by :func:`sto_ng_overlap_matrix` and
    :func:`sto_ng_kinetic_overlap_matrix`.

    .. math::

        T_{n_1,n_2,d,i_1,i_2}
        \;\propto\;
        \int_{-\infty}^{\infty} x^{i_1}\,e^{-\alpha_{1,n_1}(x-\mu_{1,d})^2}\;
        x^{i_2}\,e^{-\alpha_{2,n_2}(x-\mu_{2,d})^2}\;dx

    Args:
        mu_1 (torch.Tensor): Centers for primitive set 1 (shape: `[..., 3]` or
            `[..., B_1, 3]` depending on caller; broadcastable with `alphas_1`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for set 1
            (shape: `[..., B_1, N_1]`).
        mu_2 (torch.Tensor): Centers for primitive set 2 (shape: `[..., 3]` or
            `[..., B_2, 3]` depending on caller; broadcastable with `alphas_2`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for set 2
            (shape: `[..., B_2, N_2]`).

    Returns:
        torch.Tensor: Primitive overlap tensor `T` (shape:
        `[..., B_1, B_2, N_1, N_2, 3, L+1, L+1]`), where
        `L = GTO_L_MAX`.
    """

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4)

    sqrt_etas = torch.sqrt(_alphas_1 * _alphas_2 / (_alphas_1 + _alphas_2))

    _mu_1 = mu_1.unsqueeze(-2)
    _mu_2 = mu_2.unsqueeze(-3)
    dx = _mu_1 - _mu_2

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1], device=sqrt_etas.device)

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        "...abmnj, jp, ...abmnp, ...abdp -> ...abmndj",
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR,
        sqrt_eta_powers,
        r_powers,
    )

    alphas_1_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_1.device)
    alphas_2_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_2.device)
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    sqrt_alphas_1 = torch.sqrt(alphas_1).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    eta_gaussian_exponents = torch.einsum("...abmn, ...abd -> ...abdmn", sqrt_etas**2, dx**2)

    eta_gaussians = torch.einsum(
        "...abdmn, ...abmn -> ...abdmn",
        torch.exp(-eta_gaussian_exponents),
        sqrt_etas / ((_alphas_1 * _alphas_2) ** (1 / 4)),
    )

    A_1 = (sqrt_alphas_1 ** (-alphas_1_powers.unsqueeze(0))) * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0))
    J_12 = hermite_coeffs[..., j_powers]

    T = torch.einsum(
        "ij, ...amj, ...abmndjk, ...abdmn, ...bnk, kl -> ...abmndil",
        _GTO_ALPHA_TENSOR,
        A_1,
        J_12,
        eta_gaussians,
        A_2,
        _GTO_ALPHA_TENSOR.T,
    )

    return T


def sto_ng_block_primitive_overlap_matrix(
    mu_1: torch.Tensor,
    alphas_1: torch.Tensor,
    mu_2: torch.Tensor,
    alphas_2: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate primitive 1D overlap tensors for l2-normalized Gaussian primitives.

    The returned tensor provides per-dimension overlaps that can be combined
    (via a Hadamard product over `d = 1,2,3`) to obtain Cartesian primitive
    overlaps for any two Cartesian GTOs.

    The maximum supported angular momentum is
    :attr:`GTO_L_MAX <pyquinn.torch.lcao.gto.GTO_L_MAX>`.

    For any two sets of cartesian GTOs :math:`\phi_1, \phi_2` with powers
    :math:`(i_1, j_1, k_1)` and :math:`(i_2,j_2,k_2)`, the returned tensor
    :math:`T` satisfies:

    .. math::

        \langle \phi_1 | \phi_2 \rangle =
        \sum_{n_1=1}^{N_1}\sum_{n_2=1}^{N_2} c_{n_1} c_{n_2}
            (T_{n_1, n_2, 1, i_1, i_2}
            \odot T_{n_1, n_2, 2, j_1, j_2}
            \odot T_{n_1, n_2, 3, k_1, k_2})

    where the product :math:`\odot` (the Hadamard product) is computed
    elementwise.

    This function supports pairwise batching over sets of exponents via the
    `B_1` and `B_2` axes of `alphas_1` and `alphas_2`.

    Note that any leading dimensions to the tensors `mu_1`
    and `mu_2` are not evaluated in this batch-pairwise fashion.

    Args:
        mu_1 (torch.Tensor): Centers for set 1 (shape: `[..., 3]`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for set 1
            (shape: `[..., B_1, N_1]`).
        mu_2 (torch.Tensor): Centers for set 2 (shape: `[..., 3]`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for set 2
            (shape: `[..., B_2, N_2]`).

    Returns:
        torch.Tensor: Primitive overlap tensor `T` (shape:
        `[..., B_1, B_2, N_1, N_2, 3, L+1, L+1]`).
    """

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4)
    sqrt_etas = torch.sqrt(_alphas_1 * _alphas_2 / (_alphas_1 + _alphas_2))

    dx = mu_1 - mu_2

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1], device=sqrt_etas.device)

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        "...abmnj, jp, ...abmnp, ...dp -> ...abmndj",
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR.to(sqrt_eta_powers.device),
        sqrt_eta_powers,
        r_powers,
    )

    alphas_1_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_1.device)
    alphas_2_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_2.device)
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    sqrt_alphas_1 = torch.sqrt(alphas_1).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    eta_gaussian_exponents = torch.einsum("...abmn, ...d -> ...abdmn", sqrt_etas**2, dx**2)

    eta_gaussians = torch.einsum(
        "...abdmn, ...abmn -> ...abdmn",
        torch.exp(-eta_gaussian_exponents),
        sqrt_etas / ((_alphas_1 * _alphas_2) ** (1 / 4)),
    )

    A_1 = (sqrt_alphas_1 ** (-alphas_1_powers.unsqueeze(0))) * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0))
    J_12 = hermite_coeffs[..., j_powers]

    _ALPHA = _GTO_ALPHA_TENSOR.to(eta_gaussians.device)
    T = torch.einsum(
        "ij, ...amj, ...abmndjk, ...abdmn, ...bnk, kl -> ...abmndil",
        _ALPHA,
        A_1,
        J_12,
        eta_gaussians,
        A_2,
        _ALPHA.T,
    )

    return T


def sto_ng_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate pairwise contracted Cartesian orbital overlap matrix.

    .. math::

        M_{i,j} = \langle \phi_{1,i} \mid \phi_{2,j} \rangle

    Pairwise batching over exponent sets is supported through `alphas_1` and
    `alphas_2` (axes `B_1` and `B_2`). Each orbital selects which exponent set
    to use via `alpha_idxs_1` and `alpha_idxs_2`.

    Args:
        mu_1 (torch.Tensor): Centers for orbital set 1 (shape: `[..., B_1, 3]`).
        ijk_1 (torch.IntTensor): Cartesian powers for orbital set 1
            (shape: `[..., I_1, 3]`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for orbital set 1
            (shape: `[..., B_1, N_1]`).
        alpha_idxs_1 (torch.IntTensor): Exponent-set indices for orbital set 1
            (shape: `[..., I_1]`, values in `[0, B_1)`).
        cs_1 (torch.Tensor): Contraction coefficients for orbital set 1
            (shape: `[..., I_1, N_1]`).
        mu_2 (torch.Tensor): Centers for orbital set 2 (shape: `[..., B_2, 3]`).
        ijk_2 (torch.IntTensor): Cartesian powers for orbital set 2
            (shape: `[..., I_2, 3]`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for orbital set 2
            (shape: `[..., B_2, N_2]`).
        alpha_idxs_2 (torch.IntTensor): Exponent-set indices for orbital set 2
            (shape: `[..., I_2]`, values in `[0, B_2)`).
        cs_2 (torch.Tensor): Contraction coefficients for orbital set 2
            (shape: `[..., I_2, N_2]`).

    Returns:
        torch.Tensor: Overlap matrix `M` (shape: `[..., I_1, I_2]`).
    """

    mu_batch_shape = mu_1.shape[:-2]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    # determine dimensions of I_1,I_2,B_1, and B_2 tensors
    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    # flatten leading indices of mu and alpha tensors
    _mu_1 = mu_1.reshape(mu_batch_size, b1_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, b2_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)

    # calculate primitive overlap matrix
    T = sto_ng_primitive_overlap_matrix(_mu_1, _alphas_1, _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2].prod(-5)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum("...im, ...ijmn, ...jn -> ...ij", _cs_1, T_ijk, _cs_2).reshape(*mu_batch_shape, i1_size, i2_size)

    return M


def sto_ng_block_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate the Cartesian GTO pairwise orbital overlap matrix
    between all orbitals on adjacent sites.

    .. math::

        M_{i,j} = \langle \phi_{1,i} | \phi_{2,j} \rangle

    Since each site may have multiple orbitals (indexed by i and j) that share
    the same alpha values, pairwise batching of the overlap integrals between
    sites is supported. This allows for the passing of a smaller list of alphas
    (of size B_1 and B_2), which are then indexed in each orbital by
    `alpha_idxs_1` and `alpha_idxs_2`, which are arrays of length I_1 > B_1
    and I_2 > B_2.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., 3])
        ijk_1 (torch.IntTensor): Cartesian powers of x,y,z for orbital 1
                                 (shape: [..., I_1, 3])
        alphas_1 (torch.Tensor): Gaussian widths of orbital 1
                                 (shape: [..., B_1, N_1])
        alpha_idxs_1 (torch.IntTensor): Batch indexes i (0 <= i < B_1) of the
                                        evaluated set of alphas. If batching
                                        of alphas is not used, should be all
                                        zeros (shape: [...,I_1])
        cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                             (shape: [..., I_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [..., 3])
        ijk_2 (torch.IntTensor): Cartesian powers of x,y,z for orbital 2
                                 (shape: [..., I_2, 3])
        alphas_2 (torch.Tensor): Gaussian widths of orbital 2
                                 (shape: [..., B_2, N_2])
        alpha_idxs_2 (torch.IntTensor): Batch indexes i (0 < i < B_2) of the
                                        evaluated set of alphas. If batching
                                        of alphas is not used, should be all
                                        zeros (shape: [...,I_2])
        cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                                (shape: [..., I_2, N_2])

    Returns:
        torch.Tensor: Overlap matrix tensor M (shape: [...,I_1, I_2])
    """

    mu_batch_shape = mu_1.shape[:-1]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    # determine dimensions of I_1,I_2,B_1, and B_2 tensors
    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    # flatten leading indices of mu and alpha tensors
    _mu_1 = mu_1.reshape(mu_batch_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)

    # calculate primitive overlap matrix
    T = sto_ng_block_primitive_overlap_matrix(_mu_1, _alphas_1, _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2].prod(-5)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum("...im, ...ijmn, ...jn -> ...ij", _cs_1, T_ijk, _cs_2).reshape(*mu_batch_shape, i1_size, i2_size)

    return M


def sto_ng_kinetic_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate pairwise contracted Cartesian kinetic energy overlap matrix.

    .. math::

        M_{i,j} = -\frac{1}{2} \langle \phi_{1,i} | \nabla^2 | \phi_{2,j} \rangle

    Since each site may have multiple orbitals (indexed by i and j) that share
    the same alpha values, pairwise batching of the overlap integrals between
    sites is supported. This allows for the passing of a smaller list of alphas
    (of size B_1 and B_2), which are then indexed in each orbital by
    `alpha_idxs_1` and `alpha_idxs_2`, which are arrays of length I_1 > B_1
    and I_2 > B_2.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., B_1, 3])
        ijk_1 (torch.IntTensor): Cartesian powers of x,y,z for orbital 1
                                 (shape: [..., I_1, 3])
        alphas_1 (torch.Tensor): Gaussian widths of orbital 1
                                 (shape: [..., B_1, N_1])
        alpha_idxs_1 (torch.IntTensor): Batch indexes i (0 <= i < B_1) of the
                                        evaluated set of alphas. If batching
                                        of alphas is not used, should be all
                                        zeros (shape: [...,I_1])
        cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                             (shape: [..., I_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [..., B_2, 3])
        ijk_2 (torch.IntTensor): Cartesian powers of x,y,z for orbital 2
                                 (shape: [..., I_2, 3])
        alphas_2 (torch.Tensor): Gaussian widths of orbital 2
                                 (shape: [..., B_2, N_2])
        alpha_idxs_2 (torch.IntTensor): Batch indexes i (0 < i < B_2) of the
                                        evaluated set of alphas. If batching
                                        of alphas is not used, should be all
                                        zeros (shape: [...,I_2])
        cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                                (shape: [..., I_2, N_2])

    Returns:
        torch.Tensor: Kinetic matrix tensor M (shape: [...,I_1, I_2])
    """

    mu_batch_shape = mu_1.shape[:-2]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    _mu_1 = mu_1.reshape(mu_batch_size, b1_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, b2_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)

    # normalized primitive overlaps
    T = sto_ng_primitive_overlap_matrix(_mu_1, _alphas_1, _mu_2, _alphas_2)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)

    P_alpha_1 = _alphas_1.take_along_dim(alpha_idxs_1.reshape(mu_batch_size, -1, 1), dim=-2).view(
        mu_batch_size, i1_size, -1
    )
    P_alpha_2 = _alphas_2.take_along_dim(alpha_idxs_2.reshape(mu_batch_size, -1, 1), dim=-2).view(
        mu_batch_size, i2_size, -1
    )

    i1_d = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    i2_d = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    def _gather_dim(i1_idx, i2_idx):
        valid = (i1_idx >= 0) & (i2_idx >= 0)
        i1_clamped = i1_idx.clamp(min=0)
        i2_clamped = i2_idx.clamp(min=0)
        gathered = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, i1_clamped, i2_clamped]
        return gathered * valid.unsqueeze(-1).unsqueeze(-1)

    T_dim = _gather_dim(i1_d, i2_d)
    T_dim_pp = _gather_dim(i1_d + 1, i2_d + 1)
    T_dim_pm = _gather_dim(i1_d + 1, i2_d - 1)
    T_dim_mp = _gather_dim(i1_d - 1, i2_d + 1)
    T_dim_mm = _gather_dim(i1_d - 1, i2_d - 1)

    S_base = T_dim.prod(dim=1)

    def _replace_dim(t_dim_shifted):
        ratio = torch.where(T_dim != 0, t_dim_shifted / T_dim, torch.zeros_like(t_dim_shifted))
        return S_base.unsqueeze(1) * ratio

    S_pp = _replace_dim(T_dim_pp)
    S_pm = _replace_dim(T_dim_pm)
    S_mp = _replace_dim(T_dim_mp)
    S_mm = _replace_dim(T_dim_mm)

    alpha_1 = P_alpha_1[:, None, :, None, :, None].expand(
        mu_batch_size,
        3,
        i1_size,
        i2_size,
        P_alpha_1.shape[-1],
        P_alpha_2.shape[-1],
    )
    alpha_2 = P_alpha_2[:, None, None, :, None, :].expand(
        mu_batch_size,
        3,
        i1_size,
        i2_size,
        P_alpha_1.shape[-1],
        P_alpha_2.shape[-1],
    )

    # normalized coefficients: a_i = sqrt(alpha*(2i+1)), b_i = 2i*sqrt(alpha/(2i-1))
    A_1 = torch.sqrt(alpha_1 * (2 * i1_d[..., None, None] + 1))
    A_2 = torch.sqrt(alpha_2 * (2 * i2_d[..., None, None] + 1))
    B_1 = torch.where(
        i1_d[..., None, None] < 1,
        torch.zeros_like(A_1),
        2 * i1_d[..., None, None] * torch.sqrt(alpha_1 / (2 * i1_d[..., None, None] - 1)),
    )
    B_2 = torch.where(
        i2_d[..., None, None] < 1,
        torch.zeros_like(A_2),
        2 * i2_d[..., None, None] * torch.sqrt(alpha_2 / (2 * i2_d[..., None, None] - 1)),
    )

    P_ijk = (A_1 * A_2) * S_pp
    P_ijk -= (A_1 * B_2) * S_pm
    P_ijk -= (B_1 * A_2) * S_mp
    P_ijk += (B_1 * B_2) * S_mm

    M = 0.5 * torch.einsum("...im, ...dijmn, ...jn -> ...ij", _cs_1, P_ijk, _cs_2).reshape(
        *mu_batch_shape, i1_size, i2_size
    )

    return M


def sto_ng_block_kinetic_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate pairwise Cartesian kinetic energy overlap matrix (unbatched centers).

    .. math::

        M_{i,j} = -\frac{1}{2} \langle \phi_{1,i} | \nabla^2 | \phi_{2,j} \rangle

    This has the same batching conventions as
    :func:`sto_ng_block_overlap_matrix`, but evaluates the kinetic energy
    operator instead of the identity operator.

    Args:
        mu_1 (torch.Tensor): Center of orbital set 1 (shape: `[..., 3]`).
        ijk_1 (torch.IntTensor): Cartesian powers for orbital set 1
            (shape: `[..., I_1, 3]`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for orbital set 1
            (shape: `[..., B_1, N_1]`).
        alpha_idxs_1 (torch.IntTensor): Exponent-set indices for orbital set 1
            (shape: `[..., I_1]`, values in `[0, B_1)`).
        cs_1 (torch.Tensor): Contraction coefficients for orbital set 1
            (shape: `[..., I_1, N_1]`).
        mu_2 (torch.Tensor): Center of orbital set 2 (shape: `[..., 3]`).
        ijk_2 (torch.IntTensor): Cartesian powers for orbital set 2
            (shape: `[..., I_2, 3]`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for orbital set 2
            (shape: `[..., B_2, N_2]`).
        alpha_idxs_2 (torch.IntTensor): Exponent-set indices for orbital set 2
            (shape: `[..., I_2]`, values in `[0, B_2)`).
        cs_2 (torch.Tensor): Contraction coefficients for orbital set 2
            (shape: `[..., I_2, N_2]`).

    Returns:
        torch.Tensor: Kinetic energy matrix `M` (shape: `[..., I_1, I_2]`).
    """

    mu_batch_shape = mu_1.shape[:-1]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    _mu_1 = mu_1.reshape(mu_batch_size, b1_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, b2_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)

    # normalized primitive overlaps
    T = sto_ng_primitive_overlap_matrix(_mu_1, _alphas_1, _mu_2, _alphas_2)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)

    P_alpha_1 = _alphas_1.take_along_dim(alpha_idxs_1.reshape(mu_batch_size, -1, 1), dim=-2).view(
        mu_batch_size, i1_size, -1
    )
    P_alpha_2 = _alphas_2.take_along_dim(alpha_idxs_2.reshape(mu_batch_size, -1, 1), dim=-2).view(
        mu_batch_size, i2_size, -1
    )

    i1_d = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    i2_d = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    def _gather_dim(i1_idx, i2_idx):
        valid = (i1_idx >= 0) & (i2_idx >= 0)
        i1_clamped = i1_idx.clamp(min=0)
        i2_clamped = i2_idx.clamp(min=0)
        gathered = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, i1_clamped, i2_clamped]
        return gathered * valid.unsqueeze(-1).unsqueeze(-1)

    T_dim = _gather_dim(i1_d, i2_d)
    T_dim_pp = _gather_dim(i1_d + 1, i2_d + 1)
    T_dim_pm = _gather_dim(i1_d + 1, i2_d - 1)
    T_dim_mp = _gather_dim(i1_d - 1, i2_d + 1)
    T_dim_mm = _gather_dim(i1_d - 1, i2_d - 1)

    S_base = T_dim.prod(dim=1)

    def _replace_dim(t_dim_shifted):
        ratio = torch.where(T_dim != 0, t_dim_shifted / T_dim, torch.zeros_like(t_dim_shifted))
        return S_base.unsqueeze(1) * ratio

    S_pp = _replace_dim(T_dim_pp)
    S_pm = _replace_dim(T_dim_pm)
    S_mp = _replace_dim(T_dim_mp)
    S_mm = _replace_dim(T_dim_mm)

    alpha_1 = P_alpha_1[:, None, :, None, :, None].expand(
        mu_batch_size,
        3,
        i1_size,
        i2_size,
        P_alpha_1.shape[-1],
        P_alpha_2.shape[-1],
    )
    alpha_2 = P_alpha_2[:, None, None, :, None, :].expand(
        mu_batch_size,
        3,
        i1_size,
        i2_size,
        P_alpha_1.shape[-1],
        P_alpha_2.shape[-1],
    )

    # normalized coefficients: a_i = sqrt(alpha*(2i+1)), b_i = 2i*sqrt(alpha/(2i-1))
    A_1 = torch.sqrt(alpha_1 * (2 * i1_d[..., None, None] + 1))
    A_2 = torch.sqrt(alpha_2 * (2 * i2_d[..., None, None] + 1))
    B_1 = torch.where(
        i1_d[..., None, None] < 1,
        torch.zeros_like(A_1),
        2 * i1_d[..., None, None] * torch.sqrt(alpha_1 / (2 * i1_d[..., None, None] - 1)),
    )
    B_2 = torch.where(
        i2_d[..., None, None] < 1,
        torch.zeros_like(A_2),
        2 * i2_d[..., None, None] * torch.sqrt(alpha_2 / (2 * i2_d[..., None, None] - 1)),
    )

    P_ijk = (A_1 * A_2) * S_pp
    P_ijk -= (A_1 * B_2) * S_pm
    P_ijk -= (B_1 * A_2) * S_mp
    P_ijk += (B_1 * B_2) * S_mm

    M = 0.5 * torch.einsum("...im, ...dijmn, ...jn -> ...ij", _cs_1, P_ijk, _cs_2).reshape(
        *mu_batch_shape, i1_size, i2_size
    )

    return M


def sto_ng_momentum_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate pairwise contracted Cartesian momentum overlap matrix.

    .. math::

        \vec{M}_{i,j} = \langle \phi_{1,i} | -\nabla | \phi_{2,j} \rangle

    Since each site may have multiple orbitals (indexed by i and j) that share
    the same alpha values, pairwise batching of the overlap integrals between
    sites is supported. This allows for the passing of a smaller list of alphas
    (of size B_1 and B_2), which are then indexed in each orbital by
    `alpha_idxs+1` and `alpha_idxs_2`, which are arrays of length I_1 > B_1
    and I_2 > B_2.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., B_1, 3])
        ijk_1 (torch.IntTensor): Cartesian powers of x,y,z for orbital 1
                                 (shape: [..., I_1, 3])
        alphas_1 (torch.Tensor): Gaussian widths of orbital 1
                                 (shape: [..., B_1, N_1])
        alpha_idxs_1 (torch.IntTensor): Batch indexes i (0 <= i < B_1) of the
                                        evaluated set of alphas. If batching
                                        of alphas is not used, should be all
                                        zeros (shape: [...,I_1])
        cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                             (shape: [..., I_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [..., B_2, 3])
        ijk_2 (torch.IntTensor): Cartesian powers of x,y,z for orbital 2
                                 (shape: [..., I_2, 3])
        alphas_2 (torch.Tensor): Gaussian widths of orbital 2
                                 (shape: [..., B_2, N_2])
        alpha_idxs_2 (torch.IntTensor): Batch indexes i (0 < i < B_2) of the
                                        evaluated set of alphas. If batching
                                        of alphas is not used, should be all
                                        zeros (shape: [...,I_2])
        cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                                (shape: [..., I_2, N_2])

    Returns:
        torch.Tensor: Momentum matrix tensor M (shape: [...,3, I_1, I_2])
    """

    mu_batch_shape = mu_1.shape[:-2]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    _mu_1 = mu_1.reshape(mu_batch_size, b1_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, b2_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)

    # normalized primitive overlaps
    T = sto_ng_primitive_overlap_matrix(_mu_1, _alphas_1, _mu_2, _alphas_2)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)

    P_alpha_1 = _alphas_1.take_along_dim(alpha_idxs_1.reshape(mu_batch_size, -1, 1), dim=-2).view(
        mu_batch_size, i1_size, -1
    )
    P_alpha_2 = _alphas_2.take_along_dim(alpha_idxs_2.reshape(mu_batch_size, -1, 1), dim=-2).view(
        mu_batch_size, i2_size, -1
    )

    i1_d = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    i2_d = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    def _gather_dim(i1_idx, i2_idx):
        valid = (i1_idx >= 0) & (i2_idx >= 0)
        i1_clamped = i1_idx.clamp(min=0)
        i2_clamped = i2_idx.clamp(min=0)
        gathered = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, i1_clamped, i2_clamped]
        return gathered * valid.unsqueeze(-1).unsqueeze(-1)

    T_dim = _gather_dim(i1_d, i2_d)
    T_dim_p = _gather_dim(i1_d, i2_d + 1)
    T_dim_m = _gather_dim(i1_d, i2_d - 1)

    # build per-dimension products explicitly to avoid division-by-zero
    T_x = T_dim[:, 0]
    T_y = T_dim[:, 1]
    T_z = T_dim[:, 2]

    S_p = torch.stack(
        (
            T_dim_p[:, 0] * T_y * T_z,
            T_dim_p[:, 1] * T_x * T_z,
            T_dim_p[:, 2] * T_x * T_y,
        ),
        dim=1,
    )
    S_m = torch.stack(
        (
            T_dim_m[:, 0] * T_y * T_z,
            T_dim_m[:, 1] * T_x * T_z,
            T_dim_m[:, 2] * T_x * T_y,
        ),
        dim=1,
    )

    alpha_2 = P_alpha_2[:, None, None, :, None, :].expand(
        mu_batch_size,
        3,
        i1_size,
        i2_size,
        P_alpha_1.shape[-1],
        P_alpha_2.shape[-1],
    )

    # normalized coefficients for derivative of the second orbital
    A_2 = torch.sqrt(alpha_2 * (2 * i2_d[..., None, None] + 1))
    B_2 = torch.where(
        i2_d[..., None, None] < 1,
        torch.zeros_like(A_2),
        2 * i2_d[..., None, None] * torch.sqrt(alpha_2 / (2 * i2_d[..., None, None] - 1)),
    )

    # -d/dx : -(b_i |i-1> - a_i |i+1>) = -b_i|i-1> + a_i|i+1>
    P_ijk = (A_2 * S_p) - (B_2 * S_m)

    M = torch.einsum("...im, ...dijmn, ...jn -> ...dij", _cs_1, P_ijk, _cs_2).reshape(
        *mu_batch_shape, 3, i1_size, i2_size
    )

    return M


def sto_ng_position_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate pairwise contracted Cartesian position overlap matrix.

    .. math::

        \vec{R}_{i,j} = \langle \phi_{1,i} | \vec{r} | \phi_{2,j} \rangle

    This uses the same batching conventions as :func:`sto_ng_overlap_matrix`
    and assumes l2-normalized Cartesian Gaussian primitives.

    Args:
        mu_1 (torch.Tensor): Centers for orbital set 1 (shape: `[..., B_1, 3]`).
        ijk_1 (torch.IntTensor): Cartesian powers for orbital set 1
            (shape: `[..., I_1, 3]`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for orbital set 1
            (shape: `[..., B_1, N_1]`).
        alpha_idxs_1 (torch.IntTensor): Exponent-set indices for orbital set 1
            (shape: `[..., I_1]`, values in `[0, B_1)`).
        cs_1 (torch.Tensor): Contraction coefficients for orbital set 1
            (shape: `[..., I_1, N_1]`).
        mu_2 (torch.Tensor): Centers for orbital set 2 (shape: `[..., B_2, 3]`).
        ijk_2 (torch.IntTensor): Cartesian powers for orbital set 2
            (shape: `[..., I_2, 3]`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for orbital set 2
            (shape: `[..., B_2, N_2]`).
        alpha_idxs_2 (torch.IntTensor): Exponent-set indices for orbital set 2
            (shape: `[..., I_2]`, values in `[0, B_2)`).
        cs_2 (torch.Tensor): Contraction coefficients for orbital set 2
            (shape: `[..., I_2, N_2]`).

    Returns:
        torch.Tensor: Position overlap tensor `R` (shape: `[..., 3, I_1, I_2]`).
    """

    mu_batch_shape = mu_1.shape[:-2]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    _mu_1 = mu_1.reshape(mu_batch_size, b1_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, b2_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)

    T = sto_ng_primitive_overlap_matrix(_mu_1, _alphas_1, _mu_2, _alphas_2)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(mu_batch_size, 3, i1_size, i2_size)

    P_alpha_2 = _alphas_2.take_along_dim(alpha_idxs_2.reshape(mu_batch_size, -1, 1), dim=-2).view(
        mu_batch_size, i2_size, -1
    )

    i1_d = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    i2_d = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    max_l = T.shape[-1] - 1
    if torch.any(i2_d + 1 > max_l):
        raise ValueError(
            f"sto_ng_position_overlap_matrix: required l index exceeds max_l={max_l}. "
            "Increase GTO_L_MAX or ensure angular momentum is within range."
        )

    T_dim = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, i1_d, i2_d]
    T_dim_p = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, i1_d, i2_d + 1]

    T_x = T_dim[:, 0]
    T_y = T_dim[:, 1]
    T_z = T_dim[:, 2]
    S_base = T_x * T_y * T_z

    S_p = torch.empty_like(T_dim_p)
    S_p[:, 0] = T_dim_p[:, 0] * T_y * T_z
    S_p[:, 1] = T_dim_p[:, 1] * T_x * T_z
    S_p[:, 2] = T_dim_p[:, 2] * T_x * T_y

    mu_2_orb = _mu_2.take_along_dim(alpha_idxs_2.reshape(mu_batch_size, -1, 1), dim=-2).view(mu_batch_size, i2_size, 3)
    mu2 = mu_2_orb.to(dtype=S_base.dtype).transpose(-2, -1)[:, :, None, :, None, None]

    # normalized raising coefficient for x on the ket:
    # x|i> = mu|i> + sqrt(2i+1)/(2*sqrt(alpha)) |i+1>
    alpha_2 = P_alpha_2[:, None, None, :, None, :].to(dtype=S_base.dtype)
    sqrt_alpha_2 = torch.sqrt(alpha_2)
    c_2 = torch.sqrt(2 * i2_d.to(dtype=S_base.dtype) + 1.0)[..., None, None] / (2.0 * sqrt_alpha_2)

    R_ijk = mu2 * S_base.unsqueeze(1) + c_2 * S_p

    M = torch.einsum("...im, ...dijmn, ...jn -> ...dij", _cs_1, R_ijk, _cs_2).reshape(
        *mu_batch_shape, 3, i1_size, i2_size
    )

    return M


def sto_ng_primitive_potential_overlap_matrix(
    mu_1: torch.Tensor,
    alphas_1: torch.Tensor,
    gammas_1: torch.Tensor,
    mu_2: torch.Tensor,
    alphas_2: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate primitive overlap tensors with an l1-normalized Gaussian potential.

    This routine computes per-dimension primitive overlap tensors used to build
    Cartesian matrix elements of a Gaussian-expanded potential centered at the
    first site.

    .. math::

        T_{m,n_1,n_2,d,i_1,i_2}
        \;\propto\;
        \int_{-\infty}^{\infty} x^{i_1}\,e^{-\alpha_{1,n_1}(x-\mu_{1,d})^2}\;
        F_{\gamma_m}^{0 0 0}(x-\mu_{1,d})\;
        x^{i_2}\,e^{-\alpha_{2,n_2}(x-\mu_{2,d})^2}\;dx

    Args:
        mu_1 (torch.Tensor): Centers for orbital set 1 (shape: `[..., 3]` or
            `[..., B_1, 3]`, depending on caller).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for orbital set 1
            (shape: `[..., B_1, N_1]`).
        gammas_1 (torch.Tensor): Primitive potential Gaussian exponents
            (shape: `[..., M_1]`).
        mu_2 (torch.Tensor): Centers for orbital set 2 (shape: `[..., 3]` or
            `[..., B_2, 3]`, depending on caller).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for orbital set 2
            (shape: `[..., B_2, N_2]`).

    Returns:
        torch.Tensor: Primitive potential overlap tensor `T` (shape:
        `[..., M_1, B_1, B_2, N_1, N_2, 3, L+1, L+1]`), where
        `L = GTO_L_MAX`.
    """

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3).unsqueeze(-5)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4).unsqueeze(-5)
    _gammas_1 = gammas_1.view(*(gammas_1.shape), 1, 1, 1, 1)
    sqrt_etas = torch.sqrt((_alphas_1 + _gammas_1) * _alphas_2 / (_alphas_1 + _gammas_1 + _alphas_2))

    _mu_1 = mu_1.unsqueeze(-2)
    _mu_2 = mu_2.unsqueeze(-3)
    dx = _mu_1 - _mu_2

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1], device=sqrt_etas.device)

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        "...gabmnj, jp, ...gabmnp, ...gabdp -> ...gabmndj",
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR.to(sqrt_eta_powers.device),
        sqrt_eta_powers,
        r_powers,
    )

    alphas_1_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_1.device)
    alphas_2_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_2.device)
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    eta_gaussian_exponents = torch.einsum("...gabmn, ...gabd -> ...gabdmn", sqrt_etas**2, dx**2)

    N_12 = torch.sqrt(
        (torch.sqrt(_alphas_1 * _alphas_2) * _gammas_1) / (torch.pi * (_alphas_1 + _alphas_2 + _gammas_1))
    )

    eta_gaussians = torch.einsum(
        "...gabdmn, ...gabmn -> ...gabdmn",
        torch.exp(-eta_gaussian_exponents),
        N_12,
    )

    sqrt_alphas_gammas_1 = torch.sqrt(alphas_1.unsqueeze(-3) + gammas_1.unsqueeze(-1).unsqueeze(-2)).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    g_alphas_1 = alphas_1.unsqueeze(-3)
    g_gammas_1 = gammas_1.unsqueeze(-1).unsqueeze(-2)

    G_1 = torch.sqrt(g_alphas_1 / (g_alphas_1 + g_gammas_1)).unsqueeze(-1) ** alphas_1_powers

    A_1 = (sqrt_alphas_gammas_1 ** (-alphas_1_powers.unsqueeze(0))) * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0))
    J_12 = hermite_coeffs[..., j_powers]

    _ALPHA = _GTO_ALPHA_TENSOR.to(device=eta_gaussians.device)
    T = torch.einsum(
        "...gami, ij, ...gamj, ...gabmndjk, ...gabdmn, " "...bnk, kl -> ...gabmndil",
        G_1,
        _ALPHA,
        A_1,
        J_12,
        eta_gaussians,
        A_2,
        _ALPHA.T,
    )

    return T


def sto_ng_block_primitive_potential_overlap_matrix(
    mu_1: torch.Tensor,
    alphas_1: torch.Tensor,
    gammas_1: torch.Tensor,
    mu_2: torch.Tensor,
    alphas_2: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate primitive overlap tensors under a Gaussian-expanded potential.

    The potential at site 1 is expanded as a weighted sum of `M_1`
    l1-normalized Gaussians, and the returned tensor provides per-dimension
    primitive overlaps used to assemble
    :math:`\langle \phi_1 \mid V_1 \mid \phi_2 \rangle`.

    The maximum supported angular momentum is
    :attr:`GTO_L_MAX <pyquinn.torch.lcao.gto.GTO_L_MAX>`.

    The potential function :math:`V_1(\vec{r})` is expanded as the sum of
    :math:`M` primitive Gaussians.

    For any two sets of cartesian GTOs :math:`\phi_1, \phi_2` with powers
    :math:`(i_1, j_1, k_1)` and :math:`(i_2,j_2,k_2)`, the returned tensor
    :math:`T` satisfies:

    .. math::

        \langle \phi_1 \mid V_1 \mid \phi_2 \rangle =
        \sum_{m=1}^{M_1} v_m
        \sum_{n_1=1}^{N_1}\sum_{n_2=1}^{N_2} c_{n_1} c_{n_2}
        \left(T_{m,n_1,n_2,1,i_1,i_2}
        \odot T_{m,n_1,n_2,2,j_1,j_2}
        \odot T_{m,n_1,n_2,3,k_1,k_2}\right)

    where the product :math:`\odot` (the Hadamard product) is computed
    elementwise.

    Pairwise batching over exponent sets is supported through the `B_1` and
    `B_2` axes of `alphas_1` and `alphas_2`.

    Args:
        mu_1 (torch.Tensor): Centers for orbital set 1 (shape: `[..., 3]`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for orbital set 1
            (shape: `[..., B_1, N_1]`).
        gammas_1 (torch.Tensor): Primitive potential Gaussian exponents
            (shape: `[..., M_1]`).
        mu_2 (torch.Tensor): Centers for orbital set 2 (shape: `[..., 3]`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for orbital set 2
            (shape: `[..., B_2, N_2]`).

    Returns:
        torch.Tensor: Primitive potential overlap tensor `T` (shape:
        `[..., M_1, B_1, B_2, N_1, N_2, 3, L+1, L+1]`).
    """

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3).unsqueeze(-5)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4).unsqueeze(-5)
    _gammas_1 = gammas_1.view(*(gammas_1.shape), 1, 1, 1, 1)
    sqrt_etas = torch.sqrt((_alphas_1 + _gammas_1) * _alphas_2 / (_alphas_1 + _gammas_1 + _alphas_2))

    dx = mu_1 - mu_2

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1], device=sqrt_etas.device)

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        "...gabmnj, jp, ...gabmnp, ...dp -> ...gabmndj",
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR.to(device=sqrt_eta_powers.device),
        sqrt_eta_powers,
        r_powers,
    )

    alphas_1_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_1.device)
    alphas_2_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1], device=alphas_1.device)
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    eta_gaussian_exponents = torch.einsum("...gabmn, ...d -> ...gabdmn", sqrt_etas**2, dx**2)

    N_12 = torch.sqrt(
        (torch.sqrt(_alphas_1 * _alphas_2) * _gammas_1) / (torch.pi * (_alphas_1 + _alphas_2 + _gammas_1))
    )

    eta_gaussians = torch.einsum(
        "...gabdmn, ...gabmn -> ...gabdmn",
        torch.exp(-eta_gaussian_exponents),
        N_12,
    )

    sqrt_alphas_gammas_1 = torch.sqrt(alphas_1.unsqueeze(-3) + gammas_1.unsqueeze(-1).unsqueeze(-2)).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    g_alphas_1 = alphas_1.unsqueeze(-3)
    g_gammas_1 = gammas_1.unsqueeze(-1).unsqueeze(-2)

    G_1 = torch.sqrt(g_alphas_1 / (g_alphas_1 + g_gammas_1)).unsqueeze(-1) ** alphas_1_powers

    A_1 = (sqrt_alphas_gammas_1 ** (-alphas_1_powers.unsqueeze(0))) * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0))
    J_12 = hermite_coeffs[..., j_powers]

    _ALPHA = _GTO_ALPHA_TENSOR.to(eta_gaussians.device)
    T = torch.einsum(
        "...gami, ij, ...gamj, ...gabmndjk, ...gabdmn, " "...bnk, kl -> ...gabmndil",
        G_1,
        _ALPHA,
        A_1,
        J_12,
        eta_gaussians,
        A_2,
        _ALPHA.T,
    )

    return T


def sto_ng_potential_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate pairwise contracted Cartesian potential overlap matrix.

    The potential at site 1 is expanded as a contraction of l1-normalized
    Gaussian potentials:

    .. math::

        V_1(\mathbf{r}) = \sum_{m=1}^{M_1} v_m\,F_{\gamma_m}^{0 0 0}(\mathbf{r}-\boldsymbol{\mu}_1)

    and the matrix elements are

    .. math::

        M_{i,j} = \langle \phi_{1,i} \mid V_1 \mid \phi_{2,j} \rangle

    Pairwise batching over exponent sets is supported through `alphas_1` and
    `alphas_2` (axes `B_1` and `B_2`). Each orbital selects which exponent set
    to use via `alpha_idxs_1` and `alpha_idxs_2`.

    Args:
        mu_1 (torch.Tensor): Centers for orbital set 1 (shape: `[..., B_1, 3]`).
        ijk_1 (torch.IntTensor): Cartesian powers for orbital set 1
            (shape: `[..., I_1, 3]`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for orbital set 1
            (shape: `[..., B_1, N_1]`).
        alpha_idxs_1 (torch.IntTensor): Exponent-set indices for orbital set 1
            (shape: `[..., I_1]`, values in `[0, B_1)`).
        cs_1 (torch.Tensor): Contraction coefficients for orbital set 1
            (shape: `[..., I_1, N_1]`).
        gammas_1 (torch.Tensor): Primitive potential Gaussian exponents
            (shape: `[..., M_1]`).
        vs_1 (torch.Tensor): Potential contraction coefficients
            (shape: `[..., M_1]`).
        mu_2 (torch.Tensor): Centers for orbital set 2 (shape: `[..., B_2, 3]`).
        ijk_2 (torch.IntTensor): Cartesian powers for orbital set 2
            (shape: `[..., I_2, 3]`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for orbital set 2
            (shape: `[..., B_2, N_2]`).
        alpha_idxs_2 (torch.IntTensor): Exponent-set indices for orbital set 2
            (shape: `[..., I_2]`, values in `[0, B_2)`).
        cs_2 (torch.Tensor): Contraction coefficients for orbital set 2
            (shape: `[..., I_2, N_2]`).

    Returns:
        torch.Tensor: Potential overlap matrix `M` (shape: `[..., I_1, I_2]`).
    """

    mu_batch_shape = mu_1.shape[:-2]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    # determine dimensions of I_1,I_2,B_1, and B_2 tensors
    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    # flatten leading indices of mu and alpha tensors
    _mu_1 = mu_1.reshape(mu_batch_size, b1_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, b2_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)
    _gammas_1 = gammas_1.reshape(mu_batch_size, -1)

    # calculate primitive overlap matrix
    T = sto_ng_primitive_potential_overlap_matrix(_mu_1, _alphas_1, _gammas_1, _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)
    _vs_1 = vs_1.reshape(mu_batch_size, -1)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, :, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2].prod(-6)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum("...im, ...g, ...ijgmn, ...jn -> ...ij", _cs_1, _vs_1, T_ijk, _cs_2).reshape(
        *mu_batch_shape, i1_size, i2_size
    )

    return M


def sto_ng_block_potential_overlap_matrix(
    mu_1: torch.Tensor,
    ijk_1: torch.IntTensor,
    alphas_1: torch.Tensor,
    alpha_idxs_1: torch.IntTensor,
    cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    ijk_2: torch.IntTensor,
    alphas_2: torch.Tensor,
    alpha_idxs_2: torch.IntTensor,
    cs_2: torch.IntTensor,
) -> torch.Tensor:
    r"""Evaluate pairwise Cartesian potential overlap matrix (unbatched centers).

    .. math::

        M_{i,j} = \langle \phi_{1,i} | V_1 | \phi_{2,j} \rangle

    Since each site may have multiple orbitals (indexed by i and j) that share
    the same alpha values, pairwise batching of the overlap integrals between
    sites is supported. This allows for the passing of a smaller list of alphas
    (of size B_1 and B_2), which are then indexed in each orbital by
    `alpha_idxs_1` and `alpha_idxs_2`, which are arrays of length I_1 > B_1
    and I_2 > B_2.

    Args:
        mu_1 (torch.Tensor): Center of orbital set 1 (shape: `[..., 3]`).
        ijk_1 (torch.IntTensor): Cartesian powers for orbital set 1
            (shape: `[..., I_1, 3]`).
        alphas_1 (torch.Tensor): Primitive Gaussian exponents for orbital set 1
            (shape: `[..., B_1, N_1]`).
        alpha_idxs_1 (torch.IntTensor): Exponent-set indices for orbital set 1
            (shape: `[..., I_1]`, values in `[0, B_1)`).
        cs_1 (torch.Tensor): Contraction coefficients for orbital set 1
            (shape: `[..., I_1, N_1]`).
        gammas_1 (torch.Tensor): Primitive potential Gaussian exponents
            (shape: `[..., M_1]`).
        vs_1 (torch.Tensor): Potential contraction coefficients
            (shape: `[..., M_1]`).
        mu_2 (torch.Tensor): Center of orbital set 2 (shape: `[..., 3]`).
        ijk_2 (torch.IntTensor): Cartesian powers for orbital set 2
            (shape: `[..., I_2, 3]`).
        alphas_2 (torch.Tensor): Primitive Gaussian exponents for orbital set 2
            (shape: `[..., B_2, N_2]`).
        alpha_idxs_2 (torch.IntTensor): Exponent-set indices for orbital set 2
            (shape: `[..., I_2]`, values in `[0, B_2)`).
        cs_2 (torch.Tensor): Contraction coefficients for orbital set 2
            (shape: `[..., I_2, N_2]`).

    Returns:
        torch.Tensor: Potential overlap matrix `M` (shape: `[..., I_1, I_2]`).
    """

    mu_batch_shape = mu_1.shape[:-1]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))

    # determine dimensions of I_1,I_2,B_1, and B_2 tensors
    i1_size = ijk_1.shape[-2]
    i2_size = ijk_2.shape[-2]
    b1_size = alphas_1.shape[-2]
    b2_size = alphas_2.shape[-2]

    # flatten leading indices of mu and alpha tensors
    _mu_1 = mu_1.reshape(mu_batch_size, 3)
    _mu_2 = mu_2.reshape(mu_batch_size, 3)
    _alphas_1 = alphas_1.reshape(mu_batch_size, b1_size, -1)
    _alphas_2 = alphas_2.reshape(mu_batch_size, b2_size, -1)
    _gammas_1 = gammas_1.reshape(mu_batch_size, -1)

    # calculate primitive overlap matrix
    T = sto_ng_block_primitive_potential_overlap_matrix(_mu_1, _alphas_1, _gammas_1, _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1).expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1).expand(mu_batch_size, 3, i1_size, i2_size)

    _dim = torch.arange(3, device=ijk_1.device).view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device).view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)
    _vs_1 = vs_1.reshape(mu_batch_size, -1)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, :, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2].prod(-6)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum("...im, ...g, ...ijgmn, ...jn -> ...ij", _cs_1, _vs_1, T_ijk, _cs_2).reshape(
        *mu_batch_shape, i1_size, i2_size
    )

    return M


def sto_ng_norm_coeff(
    ijk: torch.IntTensor,
    alphas: torch.Tensor,
    alpha_idxs: torch.IntTensor,
    cs: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate normalization coefficients for contracted Cartesian orbitals.

    For a contracted orbital

    .. math::

        \phi_i(\mathbf{r}) = \sum_{n=1}^{N} c_{i n}\,\phi_{\alpha_{b_i n}}^{i j k}(\mathbf{r}),

    this returns

    .. math::

        C_i = \sqrt{\langle \phi_i \mid \phi_i \rangle},

    so that `c_{i n} / C_i` produces an l2-normalized contracted orbital.

    Args:
        ijk (torch.IntTensor): Cartesian powers of x,y,z for the orbitals
            (shape: `[..., I, 3]`).
        alphas (torch.Tensor): Primitive Gaussian exponents for the orbitals
            (shape: `[..., B, N]`).
        alpha_idxs (torch.IntTensor): Exponent-set indices for the orbitals
            (shape: `[..., I]`, values in `[0, B)`).
        cs (torch.Tensor): Contraction coefficients (shape: `[..., I, N]`).

    Returns:
        torch.Tensor: Normalization coefficients `C` (shape: `[..., I]`).
    """

    mu_batch_shape = ijk.shape[:-2]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))
    i_size = ijk.shape[-2]

    mu = torch.zeros((mu_batch_size, 3), device=alphas.device)
    T = sto_ng_block_primitive_overlap_matrix(mu, alphas, mu, alphas)

    _ijk = ijk.transpose(-2, -1).reshape(mu_batch_size, 3, -1).expand(mu_batch_size, 3, i_size)
    _alpha_idxs = alpha_idxs.reshape(mu_batch_size, 1, -1).expand(*_ijk.shape)
    _dim = torch.arange(3, device=ijk.device).view(1, 3, 1).expand(*_ijk.shape)

    _i = torch.arange(mu_batch_size, device=ijk.device).view(-1, 1, 1).expand(*_ijk.shape)

    _cs = cs.reshape(mu_batch_size, i_size, -1)

    T_ijk = T[_i, _alpha_idxs, _alpha_idxs, :, :, _dim, _ijk, _ijk].prod(-4)

    C_norm = torch.sqrt(torch.einsum("...im, ...imn, ...in -> ...i", _cs, T_ijk, _cs))

    return C_norm.reshape(*mu_batch_shape, i_size)


def cho_ng(
    x: torch.Tensor,
    mu: torch.Tensor,
    l: torch.IntTensor,
    m: torch.IntTensor,
    alphas: torch.Tensor,
    cs: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate a cubic-harmonic orbital expanded in contracted Cartesian GTOs.

    The cubic harmonic orbital :math:`\chi_{l m}` is represented as a linear
    combination of contracted Cartesian orbitals (see :func:`sto_ng`):

    .. math::

        \chi_{l m}(\mathbf{x}) = \sum_{i+j+k=l} X_{l m,\, i j k}\,\phi^{i j k}(\mathbf{x})

    The contraction coefficients `cs` are used as-is (they are not internally
    normalized).

    Args:
        x (torch.Tensor): Cartesian points (shape: `[..., 3]`).
        mu (torch.Tensor): Orbital center (shape: `[3]` or `[..., 3]`,
            broadcastable to `x`).
        l (torch.IntTensor): Angular momentum quantum number (scalar tensor).
        m (torch.IntTensor): Magnetic quantum number (scalar tensor).
        alphas (torch.Tensor): Primitive Gaussian exponents (shape: `[N]`).
        cs (torch.Tensor): Primitive contraction coefficients (shape: `[N]`).

    Returns:
        torch.Tensor: Values at `x` (shape: `[...]`).
    """

    m_idx = l + m
    X_l = _GTO_CHO_TENSORS[l]  # _get_flat_tensor(_GTO_CHO_FLAT_TENSORS, l)
    X_lm = X_l[m_idx].view(-1)

    (X_lm_idx,) = torch.nonzero(X_lm, as_tuple=True)
    X_ijks = _GTO_IJK_VALUES[l]  # _get_flat_tensor(_GTO_IJK_FLAT_TENSORS, l)
    X_lm_ijks = X_ijks[X_lm_idx].view(-1, 3)
    X_lm_coeffs = X_lm[X_lm_idx].unsqueeze(-1)

    sto_3d_component = torch.vmap(
        lambda h, ijk,: h * sto_ng(x, mu, ijk, alphas, cs),
        in_dims=-2,
        out_dims=-1,
    )

    psi = torch.sum(sto_3d_component(X_lm_coeffs, X_lm_ijks), axis=-1).squeeze(-1)

    return psi


def cho_ng_norm_coeff(
    l: torch.IntTensor,
    alphas: torch.Tensor,
    alpha_idxs: torch.IntTensor,
    cs: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate normalization coefficients for cubic-harmonic contracted orbitals.

    This returns `C_i = sqrt(<chi_i|chi_i>)`, analogous to
    :func:`sto_ng_norm_coeff`. Internally it uses the Cartesian proxy
    `(i, j, k) = (l, 0, 0)` for each `l`.

    .. math::

        C_i = \sqrt{\langle \chi_i \mid \chi_i \rangle}

    Args:
        l (torch.IntTensor): Angular momentum for the orbitals (shape: `[..., I]`).
        alphas (torch.Tensor): Primitive Gaussian exponents (shape: `[..., B, N]`).
        alpha_idxs (torch.IntTensor): Exponent-set indices (shape: `[..., I]`,
            values in `[0, B)`).
        cs (torch.Tensor): Contraction coefficients (shape: `[..., I, N]`).

    Returns:
        torch.Tensor: Normalization coefficients `C` (shape: `[..., I]`).
    """
    l_zeros = torch.zeros_like(l)
    ijk = torch.stack((l, l_zeros, l_zeros), dim=-1)

    return sto_ng_norm_coeff(ijk, alphas, alpha_idxs, cs)


def cho_overlap_matrix_metadata(l_i: torch.Tensor, sparse: bool = False) -> tuple:
    r"""Compute metadata for building cubic-harmonic (CHO) overlap matrices.

    For each angular momentum value `l`, let `X_l` be the Cartesian-to-CHO
    transformation returned by :func:`_make_gto_cho_tensor`. This routine builds
    the block-diagonal transformation for a sequence `l_i`:

    .. math::

        X = \mathrm{blockdiag}\left(X_{l_1}, X_{l_2}, \ldots\right),

    so that a Cartesian overlap matrix `S` can be converted to the CHO basis as

    .. math::

        S_{\mathrm{CHO}} = X\,S\,X^\mathsf{T}.

    Args:
        l_i (torch.IntTensor): Angular momenta for blocks (shape: `[I]`).
        sparse (bool, optional): If `True`, returns a sparse block-diagonal
            transformation matrix. Defaults to `False`.

    Returns:
        tuple: A tuple containing the following:

        - cho_tensor (torch.Tensor): Cartesian-to-CHO transformation matrix
          `X` (shape: `[H, I]`, where `H = sum(2l+1)` and `I = sum(n_{ijk}(l))`).
        - l_ijk (torch.IntTensor): Concatenated Cartesian `(i, j, k)` values for
          all blocks (shape: `[I, 3]`).
        - l_alpha_idxs (torch.IntTensor): Block index for each Cartesian row
          (shape: `[I]`, values in `[0, len(l_i))`).
        - l_block_idxs (torch.IntTensor): Starting row indices for each CHO block
          in the transformed basis (shape: `[len(l_i)]`).
    """

    # dynamically construct block cho transformation matrix
    cho_tensor_blocks = tuple(_GTO_CHO_TENSORS[l.item()].to(l_i.device) for l in l_i)

    if sparse:
        cho_tensor = sparse_coo_block_diag(*cho_tensor_blocks)
    else:
        cho_tensor = torch.block_diag(*cho_tensor_blocks)

    # dynamically construct ijk indices
    l_ijk = torch.cat([_GTO_IJK_VALUES[l.item()] for l in l_i], dim=-2)

    # dynamically construct block alpha indices and coefficients
    l_alpha_idxs = torch.cat(
        [torch.tensor([i] * _GTO_IJK_VALUES[l.item()].shape[0]) for i, l in enumerate(l_i)],
        dim=-1,
    )

    # dynamically construct overlap matrix block indices
    l_block_sizes = torch.tensor([_GTO_CHO_TENSORS[l.item()].shape[0] for l in l_i])
    l_block_idxs = torch.cumsum(l_block_sizes, dim=-1) - l_block_sizes

    return cho_tensor, l_ijk, l_alpha_idxs, l_block_idxs


def e3_cho_overlap_matrix_metadata(l_i: torch.Tensor, sparse: bool = True) -> tuple:
    r"""Compute metadata for CHO overlaps with per-orbital (E3) coefficients.

    This is analogous to :func:`cho_overlap_matrix_metadata`, but also returns
    an index array `l_c_idxs` that expands per-block coefficient tensors into
    per-Cartesian-row coefficients (for cases where coefficients vary within a
    `(2l+1)` block).

    .. math::

        X = \mathrm{blockdiag}\left(X_{l_1}, X_{l_2}, \ldots\right), \qquad
        S_{\mathrm{CHO}} = X\,S\,X^\mathsf{T}

    Args:
        l_i (torch.IntTensor): Angular momenta for blocks (shape: `[I]`).
        sparse (bool, optional): If `True`, returns a sparse block-diagonal
            transformation matrix. Defaults to `True`.

    Returns:
        tuple: A tuple containing the following:

        - cho_tensor (torch.Tensor): Cartesian-to-CHO transformation matrix `X`.
        - l_ijk (torch.IntTensor): Concatenated Cartesian `(i, j, k)` values.
        - l_alpha_idxs (torch.IntTensor): Block index per Cartesian row.
        - l_c_idxs (torch.IntTensor): Indices selecting per-CHO-row coefficients
          for expanding per-block coefficient tensors.
    """

    # dynamically construct block cho transformation matrix
    cho_tensor_blocks = tuple(_GTO_E3_CHO_TENSORS[l.item()].to(l_i.device) for l in l_i)

    if sparse:
        cho_tensor = sparse_coo_block_diag(*cho_tensor_blocks)
    else:
        cho_tensor = torch.block_diag(*cho_tensor_blocks)

    # dynamically construct ijk indices
    l_ijk = torch.cat([_GTO_E3_IJK_VALUES[l.item()] for l in l_i], dim=-2)

    # dynamically construct block alpha indices and coefficients
    l_alpha_idxs = torch.cat(
        [torch.tensor([i] * _GTO_E3_IJK_VALUES[l.item()].shape[0]) for i, l in enumerate(l_i)],
        dim=-1,
    )

    # dynamically construct overlap matrix block indices
    c_block_sizes = torch.tensor([_GTO_E3_CHO_TENSORS[l.item()].shape[0] for l in l_i])
    c_idxs = torch.cumsum(c_block_sizes, dim=0) - c_block_sizes

    # dynamically construct block c indices:
    l_c_idxs = torch.cat(
        [c_idx + _GTO_E3_CHO_TENSORS[l.item()].nonzero()[:, 0] for l, c_idx in zip(l_i, c_idxs, strict=False)]
    )

    return cho_tensor, l_ijk, l_alpha_idxs, l_c_idxs


def _cho_block_data(l_i: torch.Tensor, l_cs_i: torch.Tensor, cho_data: tuple = None) -> tuple:
    """Build block metadata and expanded coefficients for CHO block overlap calls.

    Args:
        l_i (torch.Tensor): Angular momenta for blocks (shape: `[..., I]`).
        l_cs_i (torch.Tensor): Per-block contraction coefficients
            (shape: `[..., I, N]`).
        cho_data (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata`. Defaults to `None`.

    Returns:
        tuple: `(cho_tensor, l_ijk, l_alpha_idxs, l_block_cs)` suitable for the
        corresponding CHO block overlap function.
    """

    batch_shape = l_cs_i.shape[:-2]
    batch_ndim = l_cs_i.ndim - 2

    cho_tensor, l_ijk, l_alpha_idxs, _ = cho_data if cho_data is not None else cho_overlap_matrix_metadata(l_i)

    l_block_cs = l_cs_i[..., l_alpha_idxs, :]

    # pad alpha indices and coefficients to batch size
    l_alpha_idxs = l_alpha_idxs.view(*([1] * batch_ndim), -1).expand(*batch_shape, -1)

    l_ijk = l_ijk.view(*([1] * batch_ndim), -1, 3).expand(*batch_shape, -1, 3)

    return cho_tensor, l_ijk, l_alpha_idxs, l_block_cs


def cho_ng_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate pairwise cubic-harmonic (CHO) orbital overlap matrix.

    .. math::

        M_{i,j} = \langle \chi_{1,i} \mid \chi_{2,j} \rangle

    This is computed by transforming the contracted Cartesian overlap matrix
    `S` into the CHO basis:

    .. math::

        M = X_1\,S\,X_2^\mathsf{T}.

    Args:
        mu_1 (torch.Tensor): Centers for orbital blocks on site 1
            (shape: `[I_1, 3]`).
        l_1 (torch.IntTensor): Angular momenta for blocks on site 1
            (shape: `[I_1]`).
        l_alphas_1 (torch.Tensor): Primitive Gaussian exponents per block on
            site 1 (shape: `[I_1, N_1]`).
        l_cs_1 (torch.Tensor): Contraction coefficients per block on site 1
            (shape: `[I_1, N_1]`).
        mu_2 (torch.Tensor): Centers for orbital blocks on site 2
            (shape: `[I_2, 3]`).
        l_2 (torch.IntTensor): Angular momenta for blocks on site 2
            (shape: `[I_2]`).
        l_alphas_2 (torch.Tensor): Primitive Gaussian exponents per block on
            site 2 (shape: `[I_2, N_2]`).
        l_cs_2 (torch.Tensor): Contraction coefficients per block on site 2
            (shape: `[I_2, N_2]`).
        cho_data_1 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_1`. Defaults to `None`.
        cho_data_2 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_2`. Defaults to `None`.

    Returns:
        torch.Tensor: CHO overlap matrix `M` (shape: `[H_1, H_2]`), where
        `H_s = sum_i (2 l_{s,i} + 1)`.
    """
    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = (
        cho_data_1 if cho_data_1 is not None else cho_overlap_matrix_metadata(l_1)
    )

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = (
        cho_data_2 if cho_data_2 is not None else cho_overlap_matrix_metadata(l_2)
    )

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]

    M = sto_ng_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_X = cho_tensor_1 @ M @ cho_tensor_2.T

    return M_X


def cho_ng_kinetic_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate pairwise cubic-harmonic (CHO) kinetic energy overlap matrix.

    .. math::

        M_{i,j} = -\frac{1}{2}\langle \chi_{1,i} \mid \nabla^2 \mid \chi_{2,j} \rangle

    This is computed by transforming the contracted Cartesian kinetic energy
    matrix into the CHO basis:

    .. math::

        M = X_1\,T\,X_2^\mathsf{T}.

    Args:
        mu_1 (torch.Tensor): Centers for orbital blocks on site 1
            (shape: `[I_1, 3]`).
        l_1 (torch.IntTensor): Angular momenta for blocks on site 1
            (shape: `[I_1]`).
        l_alphas_1 (torch.Tensor): Primitive Gaussian exponents per block on
            site 1 (shape: `[I_1, N_1]`).
        l_cs_1 (torch.Tensor): Contraction coefficients per block on site 1
            (shape: `[I_1, N_1]`).
        mu_2 (torch.Tensor): Centers for orbital blocks on site 2
            (shape: `[I_2, 3]`).
        l_2 (torch.IntTensor): Angular momenta for blocks on site 2
            (shape: `[I_2]`).
        l_alphas_2 (torch.Tensor): Primitive Gaussian exponents per block on
            site 2 (shape: `[I_2, N_2]`).
        l_cs_2 (torch.Tensor): Contraction coefficients per block on site 2
            (shape: `[I_2, N_2]`).
        cho_data_1 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_1`. Defaults to `None`.
        cho_data_2 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_2`. Defaults to `None`.

    Returns:
        torch.Tensor: CHO kinetic energy matrix `M` (shape: `[H_1, H_2]`).
    """
    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = (
        cho_data_1 if cho_data_1 is not None else cho_overlap_matrix_metadata(l_1)
    )

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = (
        cho_data_2 if cho_data_2 is not None else cho_overlap_matrix_metadata(l_2)
    )

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]

    M = sto_ng_kinetic_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_X = cho_tensor_1 @ M @ cho_tensor_2.T

    return M_X


def cho_ng_momentum_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate pairwise cubic-harmonic (CHO) momentum overlap matrix.

    .. math::

        \vec{M}_{i,j} = \langle \chi_{1,i} \mid -\nabla \mid \chi_{2,j} \rangle

    This is computed by transforming the contracted Cartesian momentum tensor
    into the CHO basis for each component.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [I_1, 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [I_1, N_1])
        l_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [I_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [I_2, 3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [I_2, N_2])
        l_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [I_2, N_2])
        cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.

    Returns:
        torch.Tensor: Pairwise momentum overlap matrices with x, y, and z
                components. (shape: [3, H_1, H_2]).
    """
    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = (
        cho_data_1 if cho_data_1 is not None else cho_overlap_matrix_metadata(l_1)
    )

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = (
        cho_data_2 if cho_data_2 is not None else cho_overlap_matrix_metadata(l_2)
    )

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]

    M = sto_ng_momentum_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    # Apply temorary workaround to avoid pytorch unsupported expand()
    # for sparse matrix operations
    Mx, My, Mz = M.unbind(dim=-3)  # each [I1, I2]

    # Each component matmul stays 2D → sparse-safe
    Hx = cho_tensor_1 @ Mx @ cho_tensor_2.T
    Hy = cho_tensor_1 @ My @ cho_tensor_2.T
    Hz = cho_tensor_1 @ Mz @ cho_tensor_2.T

    return torch.stack([Hx, Hy, Hz], dim=-3)


def cho_ng_position_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate pairwise cubic-harmonic (CHO) position overlap matrix.

    .. math::

        \vec{R}_{i,j} = \langle \chi_{1,i} \mid \vec{r} \mid \chi_{2,j} \rangle

    This is computed by transforming the contracted Cartesian position tensor
    into the CHO basis for each component.

    Args:
        mu_1 (torch.Tensor): Centers for orbital blocks on site 1
            (shape: `[I_1, 3]`).
        l_1 (torch.IntTensor): Angular momenta for blocks on site 1
            (shape: `[I_1]`).
        l_alphas_1 (torch.Tensor): Primitive Gaussian exponents per block on
            site 1 (shape: `[I_1, N_1]`).
        l_cs_1 (torch.Tensor): Contraction coefficients per block on site 1
            (shape: `[I_1, N_1]`).
        mu_2 (torch.Tensor): Centers for orbital blocks on site 2
            (shape: `[I_2, 3]`).
        l_2 (torch.IntTensor): Angular momenta for blocks on site 2
            (shape: `[I_2]`).
        l_alphas_2 (torch.Tensor): Primitive Gaussian exponents per block on
            site 2 (shape: `[I_2, N_2]`).
        l_cs_2 (torch.Tensor): Contraction coefficients per block on site 2
            (shape: `[I_2, N_2]`).
        cho_data_1 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_1`. Defaults to `None`.
        cho_data_2 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_2`. Defaults to `None`.

    Returns:
        torch.Tensor: CHO position overlap tensor `R` (shape: `[3, H_1, H_2]`).
    """
    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = (
        cho_data_1 if cho_data_1 is not None else cho_overlap_matrix_metadata(l_1)
    )

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = (
        cho_data_2 if cho_data_2 is not None else cho_overlap_matrix_metadata(l_2)
    )

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]

    M = sto_ng_position_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    # Apply temorary workaround to avoid pytorch unsupported expand()
    # for sparse matrix operations
    Mx, My, Mz = M.unbind(dim=-3)  # each [I1, I2]

    # Each component matmul stays 2D → sparse-safe
    Hx = cho_tensor_1 @ Mx @ cho_tensor_2.T
    Hy = cho_tensor_1 @ My @ cho_tensor_2.T
    Hz = cho_tensor_1 @ Mz @ cho_tensor_2.T

    return torch.stack([Hx, Hy, Hz], dim=-3)


def cho_ng_hamiltonian_potential_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
    t0: float | torch.Tensor = 1.0,
    v0: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    r"""Evaluate the CHO Hamiltonian potential-energy matrix component.

    .. math::

        H^{(V)}_{i,j} = v_0\,\langle \chi_{1,i} \mid V_1 \mid \chi_{2,j} \rangle

    Note:
        `t0` is accepted for API consistency but is not used by this function.

    Args:
        mu_1 (torch.Tensor): Centers for orbital blocks on site 1
            (shape: `[I_1, 3]`).
        l_1 (torch.IntTensor): Angular momenta for blocks on site 1
            (shape: `[I_1]`).
        l_alphas_1 (torch.Tensor): Primitive Gaussian exponents per block on
            site 1 (shape: `[I_1, N_1]`).
        l_cs_1 (torch.Tensor): Contraction coefficients per block on site 1
            (shape: `[I_1, N_1]`).
        gammas_1 (torch.Tensor): Primitive potential Gaussian exponents
            (shape: `[M_1]`).
        vs_1 (torch.Tensor): Potential contraction coefficients (shape: `[M_1]`).
        mu_2 (torch.Tensor): Centers for orbital blocks on site 2
            (shape: `[I_2, 3]`).
        l_2 (torch.IntTensor): Angular momenta for blocks on site 2
            (shape: `[I_2]`).
        l_alphas_2 (torch.Tensor): Primitive Gaussian exponents per block on
            site 2 (shape: `[I_2, N_2]`).
        l_cs_2 (torch.Tensor): Contraction coefficients per block on site 2
            (shape: `[I_2, N_2]`).
        cho_data_1 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_1`. Defaults to `None`.
        cho_data_2 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_2`. Defaults to `None`.
        t0 (torch.Tensor | float, optional): Unused. Defaults to `1.0`.
        v0 (torch.Tensor | float, optional): Potential energy scalar prefactor.
            Defaults to `1.0`.

    Returns:
        torch.Tensor: Potential-energy matrix `H^(V)` (shape: `[H_1, H_2]`).
    """
    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = (
        cho_data_1 if cho_data_1 is not None else cho_overlap_matrix_metadata(l_1)
    )

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = (
        cho_data_2 if cho_data_2 is not None else cho_overlap_matrix_metadata(l_2)
    )

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]

    M_v = v0 * sto_ng_potential_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        gammas_1,
        vs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_X = cho_tensor_1 @ (M_v) @ cho_tensor_2.T

    return M_X


def cho_ng_hamiltonian_kinetic_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
    t0: float | torch.Tensor = 1.0,
    v0: float | torch.Tensor = 1.0,
):
    r"""Evaluate the CHO Hamiltonian kinetic-energy matrix component.

    .. math::

        H^{(T)} = t_0\,M^{(T)}, \qquad
        M^{(T)}_{i,j} = -\frac{1}{2}\langle \chi_{1,i} \mid \nabla^2 \mid \chi_{2,j} \rangle

    Parameters are forwarded to :func:`cho_ng_kinetic_overlap_matrix`.

    .. seealso::
        :func:`cho_ng_kinetic_overlap_matrix`
    """

    return t0 * cho_ng_kinetic_overlap_matrix(
        mu_1=mu_1,
        l_1=l_1,
        l_alphas_1=l_alphas_1,
        l_cs_1=l_cs_1,
        mu_2=mu_2,
        l_2=l_2,
        l_alphas_2=l_alphas_2,
        l_cs_2=l_cs_2,
    )


def cho_ng_hamiltonian_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
    t0: float | torch.Tensor = 1.0,
    v0: float | torch.Tensor = 1.0,
):
    r"""Evaluate the full CHO Hamiltonian matrix (potential + kinetic).

    .. math::

        H_{i,j} = v_0\,\langle \chi_{1,i} \mid V_1 \mid \chi_{2,j} \rangle
        \;-\; \frac{t_0}{2}\,\langle \chi_{1,i} \mid \nabla^2 \mid \chi_{2,j} \rangle

    Args:
        mu_1 (torch.Tensor): Centers for orbital blocks on site 1
            (shape: `[I_1, 3]`).
        l_1 (torch.IntTensor): Angular momenta for blocks on site 1
            (shape: `[I_1]`).
        l_alphas_1 (torch.Tensor): Primitive Gaussian exponents per block on
            site 1 (shape: `[I_1, N_1]`).
        l_cs_1 (torch.Tensor): Contraction coefficients per block on site 1
            (shape: `[I_1, N_1]`).
        gammas_1 (torch.Tensor): Primitive potential Gaussian exponents
            (shape: `[M_1]`).
        vs_1 (torch.Tensor): Potential contraction coefficients (shape: `[M_1]`).
        mu_2 (torch.Tensor): Centers for orbital blocks on site 2
            (shape: `[I_2, 3]`).
        l_2 (torch.IntTensor): Angular momenta for blocks on site 2
            (shape: `[I_2]`).
        l_alphas_2 (torch.Tensor): Primitive Gaussian exponents per block on
            site 2 (shape: `[I_2, N_2]`).
        l_cs_2 (torch.Tensor): Contraction coefficients per block on site 2
            (shape: `[I_2, N_2]`).
        cho_data_1 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_1`. Defaults to `None`.
        cho_data_2 (tuple, optional): Precomputed output from
            :func:`cho_overlap_matrix_metadata` for `l_2`. Defaults to `None`.
        t0 (torch.Tensor | float, optional): Kinetic energy scalar prefactor.
            Defaults to `1.0`.
        v0 (torch.Tensor | float, optional): Potential energy scalar prefactor.
            Defaults to `1.0`.

    Returns:
        torch.Tensor: Hamiltonian matrix `H` (shape: `[H_1, H_2]`).
    """

    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = (
        cho_data_1 if cho_data_1 is not None else cho_overlap_matrix_metadata(l_1)
    )

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = (
        cho_data_2 if cho_data_2 is not None else cho_overlap_matrix_metadata(l_2)
    )

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]

    M_v = v0 * sto_ng_potential_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        gammas_1,
        vs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_t = t0 * sto_ng_kinetic_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_X = cho_tensor_1 @ (M_v + M_t) @ cho_tensor_2.T

    return M_X


def e3_cho_ng_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    e3_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    e3_cs_2: torch.Tensor,
    e3_cho_data_1: tuple = None,
    e3_cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate pairwise CHO orbital overlap matrix with
    per-orbital coefficients.

    .. math::

        M_{i,j} = \langle \chi_{1,i} \mid \chi_{2,j} \rangle

    This uses the same basis transforms as :func:`cho_ng_overlap_matrix`, but
    allows the contraction coefficients to vary within each `(2l+1)` block
    (e.g. from an E(3)-equivariant model).

    The size of axis 0 for `e3_cs_1` and `e3_cs_2` (`C_1` and `C_2`)
    must satisfy `C = sum(2*l + 1)` where the sum is taken over all `l`
    in `l_1` and `l_2` respectively.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [I_1, 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [I_1, N_1])
        e3_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [C_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [I_2, 3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [I_2, N_2])
        e3_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [C_2, N_2])
        e3_cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        e3_cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.

    Returns:
        torch.Tensor: Pairwise overlap matrix of orbitals
                      (shape: [H_1, H_2]).
    """
    e3_cho_tensor_1, e3_ijk_1, e3_alpha_idxs_1, e3_c_idxs_1 = (
        e3_cho_data_1 if e3_cho_data_1 is not None else e3_cho_overlap_matrix_metadata(l_1)
    )

    e3_cho_tensor_2, e3_ijk_2, e3_alpha_idxs_2, e3_c_idxs_2 = (
        e3_cho_data_2 if e3_cho_data_2 is not None else e3_cho_overlap_matrix_metadata(l_2)
    )

    cs_1 = e3_cs_1[e3_c_idxs_1]
    cs_2 = e3_cs_2[e3_c_idxs_2]

    M = sto_ng_overlap_matrix(
        mu_1,
        e3_ijk_1,
        l_alphas_1,
        e3_alpha_idxs_1,
        cs_1,
        mu_2,
        e3_ijk_2,
        l_alphas_2,
        e3_alpha_idxs_2,
        cs_2,
    )

    M_X = e3_cho_tensor_1 @ M @ e3_cho_tensor_2.T

    return M_X


def e3_cho_ng_kinetic_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    e3_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    e3_cs_2: torch.Tensor,
    e3_cho_data_1: tuple = None,
    e3_cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate pairwise CHO kinetic energy overlap matrix with
    per-orbital coefficients.

    .. math::

        M_{i,j} = -\frac{1}{2}\langle \chi_{1,i} \mid \nabla^2 \mid \chi_{2,j} \rangle

    The size of axis 0 for `e3_cs_1` and `e3_cs_2` (`C_1` and `C_2`)
    must satisfy `C = sum(2*l + 1)` where the sum is taken over all `l`
    in `l_1` and `l_2` respectively. It is recommended that the
    coefficients are determined through some e3-equivariant function
    of the coordination
    geometry.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [I_1, 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [I_1, N_1])
        e3_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [C_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [I_2, 3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [I_2, N_2])
        e3_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [C_2, N_2])
        e3_cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        e3_cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.

    Returns:
        torch.Tensor: Pairwise overlap matrix of orbitals
                      (shape: [H_1, H_2]).
    """
    e3_cho_tensor_1, e3_ijk_1, e3_alpha_idxs_1, e3_c_idxs_1 = (
        e3_cho_data_1 if e3_cho_data_1 is not None else e3_cho_overlap_matrix_metadata(l_1)
    )

    e3_cho_tensor_2, e3_ijk_2, e3_alpha_idxs_2, e3_c_idxs_2 = (
        e3_cho_data_2 if e3_cho_data_2 is not None else e3_cho_overlap_matrix_metadata(l_2)
    )

    cs_1 = e3_cs_1[e3_c_idxs_1]
    cs_2 = e3_cs_2[e3_c_idxs_2]

    M = sto_ng_kinetic_overlap_matrix(
        mu_1,
        e3_ijk_1,
        l_alphas_1,
        e3_alpha_idxs_1,
        cs_1,
        mu_2,
        e3_ijk_2,
        l_alphas_2,
        e3_alpha_idxs_2,
        cs_2,
    )

    M_X = e3_cho_tensor_1 @ M @ e3_cho_tensor_2.T

    return M_X


def e3_cho_ng_hamiltonian_potential_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    e3_cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    e3_cs_2: torch.Tensor,
    e3_cho_data_1: tuple = None,
    e3_cho_data_2: tuple = None,
    t0: float | torch.Tensor = 1.0,
    v0: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    r"""Evaluate the CHO Hamiltonian potential-energy component (per-orbital coefficients).

    .. math::

        H^{(V)}_{i,j} = v_0\,\langle \chi_{1,i} \mid V_1 \mid \chi_{2,j} \rangle

    Note:
        `t0` is accepted for API consistency but is not used by this function.

    The size of axis 0 for `e3_cs_1` and `e3_cs_2` (`C_1` and `C_2`)
    must satisfy `C = sum(2*l + 1)` where the sum is taken over all `l`
    in `l_1` and `l_2` respectively. It is recommended that the
    coefficients are determined through some e3-equivariant function
    of the coordination
    geometry.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [I_1, 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [I_1, N_1])
        e3_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [C_1, N_1])
        gammas_1: (torch.Tensor): Gaussian widths of potential 1
                                  (shape: [M_1])
        vs_1: (torch.Tensor): Gaussian coefficients of potential 1
                                  (shape: [M_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [I_2, 3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [I_2, N_2])
        e3_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [C_2, N_2])
        e3_cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        e3_cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.
        t0 (torch.Tensor | float, optional): The kinetic energy scalar prefactor. Defaults to 1.0.
        v0 (torch.Tensor | float, optional): The potential energy scalar prefactor. Defaults to 1.0.

    Returns:
        torch.Tensor: Pairwise overlap matrix
                      (shape: [H_1, H_2]).
    """

    e3_cho_tensor_1, e3_ijk_1, e3_alpha_idxs_1, e3_c_idxs_1 = (
        e3_cho_data_1 if e3_cho_data_1 is not None else e3_cho_overlap_matrix_metadata(l_1)
    )

    e3_cho_tensor_2, e3_ijk_2, e3_alpha_idxs_2, e3_c_idxs_2 = (
        e3_cho_data_2 if e3_cho_data_2 is not None else e3_cho_overlap_matrix_metadata(l_2)
    )

    cs_1 = e3_cs_1[e3_c_idxs_1]
    cs_2 = e3_cs_2[e3_c_idxs_2]

    M = v0 * sto_ng_potential_overlap_matrix(
        mu_1,
        e3_ijk_1,
        l_alphas_1,
        e3_alpha_idxs_1,
        cs_1,
        gammas_1,
        vs_1,
        mu_2,
        e3_ijk_2,
        l_alphas_2,
        e3_alpha_idxs_2,
        cs_2,
    )

    M_X = e3_cho_tensor_1 @ M @ e3_cho_tensor_2.T

    return M_X


def e3_cho_ng_hamiltonian_kinetic_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    e3_cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    e3_cs_2: torch.Tensor,
    e3_cho_data_1: tuple = None,
    e3_cho_data_2: tuple = None,
    t0: float | torch.Tensor = 1.0,
    v0: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    r"""Evaluate the CHO Hamiltonian kinetic-energy component (per-orbital coefficients).

    .. math::

        H^{(T)} = t_0\,M^{(T)}, \qquad
        M^{(T)}_{i,j} = -\frac{1}{2}\langle \chi_{1,i} \mid \nabla^2 \mid \chi_{2,j} \rangle

    Parameters are forwarded to :func:`e3_cho_ng_kinetic_overlap_matrix`.

    .. seealso::
        :func:`e3_cho_ng_kinetic_overlap_matrix`
    """

    return t0 * e3_cho_ng_kinetic_overlap_matrix(
        mu_1=mu_1,
        l_1=l_1,
        l_alphas_1=l_alphas_1,
        e3_cs_1=e3_cs_1,
        mu_2=mu_2,
        l_2=l_2,
        l_alphas_2=l_alphas_2,
        e3_cs_2=e3_cs_2,
    )


def e3_cho_ng_hamiltonian_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    e3_cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    e3_cs_2: torch.Tensor,
    e3_cho_data_1: tuple = None,
    e3_cho_data_2: tuple = None,
    t0: float | torch.Tensor = 1.0,
    v0: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    r"""Evaluate the full CHO Hamiltonian matrix (per-orbital coefficients).

    .. math::

        H_{i,j} = v_0\,\langle \chi_{1,i} \mid V_1 \mid \chi_{2,j} \rangle
        \;-\; \frac{t_0}{2}\,\langle \chi_{1,i} \mid \nabla^2 \mid \chi_{2,j} \rangle

    The size of axis 0 for `e3_cs_1` and `e3_cs_2` (`C_1` and `C_2`)
    must satisfy `C = sum(2*l + 1)` where the sum is taken over all `l`
    in `l_1` and `l_2` respectively. It is recommended that the
    coefficients are determined through some e3-equivariant function
    of the coordination
    geometry.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [I_1, 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [I_1, N_1])
        e3_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [C_1, N_1])
        gammas_1: (torch.Tensor): Gaussian widths of potential 1
                                  (shape: [M_1])
        vs_1: (torch.Tensor): Gaussian coefficients of potential 1
                                  (shape: [M_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [I_2, 3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [I_2, N_2])
        e3_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [C_2, N_2])
        e3_cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        e3_cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `e3_cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.
        t0 (torch.Tensor | float, optional): The kinetic energy scalar prefactor. Defaults to 1.0.
        v0 (torch.Tensor | float, optional): The potential energy scalar prefactor. Defaults to 1.0.

    Returns:
        torch.Tensor: Pairwise overlap matrix
                      (shape: [H_1, H_2]).
    """

    e3_cho_tensor_1, e3_ijk_1, e3_alpha_idxs_1, e3_c_idxs_1 = (
        e3_cho_data_1 if e3_cho_data_1 is not None else e3_cho_overlap_matrix_metadata(l_1)
    )

    e3_cho_tensor_2, e3_ijk_2, e3_alpha_idxs_2, e3_c_idxs_2 = (
        e3_cho_data_2 if e3_cho_data_2 is not None else e3_cho_overlap_matrix_metadata(l_2)
    )

    cs_1 = e3_cs_1[e3_c_idxs_1]
    cs_2 = e3_cs_2[e3_c_idxs_2]

    M_t = t0 * sto_ng_kinetic_overlap_matrix(
        mu_1,
        e3_ijk_1,
        l_alphas_1,
        e3_alpha_idxs_1,
        cs_1,
        gammas_1,
        vs_1,
        mu_2,
        e3_ijk_2,
        l_alphas_2,
        e3_alpha_idxs_2,
        cs_2,
    )

    M_v = v0 * sto_ng_potential_overlap_matrix(
        mu_1,
        e3_ijk_1,
        l_alphas_1,
        e3_alpha_idxs_1,
        cs_1,
        gammas_1,
        vs_1,
        mu_2,
        e3_ijk_2,
        l_alphas_2,
        e3_alpha_idxs_2,
        cs_2,
    )

    M_X = e3_cho_tensor_1 @ (M_t + M_v) @ e3_cho_tensor_2.T

    return M_X


def cho_ng_block_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate batched CHO orbital overlap matrices for fixed-size blocks.

    .. math::

        M_{i,j} = \langle \chi_{1,i} \mid \chi_{2,j} \rangle

    (This function is currently unused, but may be helpful in the future
     if a large fixed basis is used for all sites.)

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [..., I_1, N_1])
        l_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [..., I_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [...,3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [..., I_2, N_2])
        l_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [..., I_2, N_2])
        cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.

    Returns:
        torch.Tensor: Pairwise overlap matrix of orbitals
                      (shape: [..., H_1, H_2]).
    """

    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, l_block_cs_1 = _cho_block_data(l_1, l_cs_1, cho_data=cho_data_1)

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, l_block_cs_2 = _cho_block_data(l_2, l_cs_2, cho_data_2)

    M = sto_ng_block_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_X = torch.einsum("mi, ...ij, jn -> ...mn", cho_tensor_1, M, cho_tensor_2.mT)

    return M_X


def cho_ng_block_kinetic_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate batched CHO kinetic energy overlap matrices for fixed-size blocks.

    .. math::

        M_{i,j} = -\frac{1}{2}\langle \chi_{1,i} \mid \nabla^2 \mid \chi_{2,j} \rangle

    (This function is currently unused, but may be helpful in the future
     if a large fixed basis is used for all sites.)

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [..., I_1, N_1])
        l_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [..., I_1, N_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [...,3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [..., I_2, N_2])
        l_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [..., I_2, N_2])
        cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.

    Returns:
        torch.Tensor: Pairwise overlap matrix of orbitals
                      (shape: [..., H_1, H_2]).
    """

    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, l_block_cs_1 = _cho_block_data(l_1, l_cs_1, cho_data=cho_data_1)

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, l_block_cs_2 = _cho_block_data(l_2, l_cs_2, cho_data=cho_data_2)

    M = sto_ng_block_kinetic_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_X = torch.einsum("mi, ...ij, jn -> ...mn", cho_tensor_1, M, cho_tensor_2.mT)

    return M_X


def cho_ng_block_potential_overlap_matrix(
    mu_1: torch.Tensor,
    l_1: torch.IntTensor,
    l_alphas_1: torch.Tensor,
    l_cs_1: torch.Tensor,
    gammas_1: torch.Tensor,
    vs_1: torch.Tensor,
    mu_2: torch.Tensor,
    l_2: torch.IntTensor,
    l_alphas_2: torch.Tensor,
    l_cs_2: torch.Tensor,
    cho_data_1: tuple = None,
    cho_data_2: tuple = None,
) -> torch.Tensor:
    r"""Evaluate batched CHO potential overlap matrices for fixed-size blocks.

    .. math::

        M_{i,j} = \langle \chi_{1,i} \mid V_1 \mid \chi_{2,j} \rangle

    (This function is currently unused, but may be helpful in the future
     if a large fixed basis is used for all sites.)

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [..., I_1, N_1])
        l_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [..., I_1, N_1])
        gammas_1: (torch.Tensor): Gaussian widths of potential 1
                                  (shape: [..., M_1])
        vs_1: (torch.Tensor): Gaussian coefficients of potential 1
                                  (shape: [..., M_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [...,3])
        l_2 (torch.IntTensor): Angular momenta of blocks for orbital 2
                               (shape: [I_2])
        l_alphas_2 (torch.Tensor): Gaussian widths of blocks for orbital 2
                                   (shape: [..., I_2, N_2])
        l_cs_2 (torch.Tensor): Gaussian coefficients of orbital 2
                               (shape: [..., I_2, N_2])
        cho_data_1 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_1)`.
                                  Defaults to None.
        cho_data_2 (tuple, optional): Precomputed data tuple returned from a
                                  call to
                                  `cho_overlap_matrix_metadata(l_2)`.
                                  Defaults to None.

    Returns:
        torch.Tensor: Pairwise overlap matrix
                      (shape: [..., H_1, H_2]).
    """

    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, l_block_cs_1 = _cho_block_data(l_1, l_cs_1, cho_data=cho_data_1)

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, l_block_cs_2 = _cho_block_data(l_2, l_cs_2, cho_data=cho_data_2)

    V_X = sto_ng_block_potential_overlap_matrix(
        mu_1,
        l_ijk_1,
        l_alphas_1,
        l_alpha_idxs_1,
        l_block_cs_1,
        gammas_1,
        vs_1,
        mu_2,
        l_ijk_2,
        l_alphas_2,
        l_alpha_idxs_2,
        l_block_cs_2,
    )

    M_X = torch.einsum("mi, ...ij, jn -> ...mn", cho_tensor_1, V_X, cho_tensor_2.mT)

    return M_X


_CHO_OCCUPATION_ORDER = torch.tensor(
    [(nl_sum - l, l) for nl_sum in range(1, 9) for l in reversed(range((nl_sum + 1) // 2))]
)
""" Order of cubic harmonic orbital occupations for
atoms expanded in a GTO basis"""

_CHO_BASIS_SIZES = torch.tensor([2 * l + 1 for l in range(GTO_L_MAX + 1)])
""" Cubic harmonic basis sizes for each
angular momentum l"""


def _get_valence_shells(Z: int) -> list:
    """Returns the valence shell angular momenta of an
    element with atomic number Z."""
    if Z <= 2:
        l_values = [0]
    else:
        for block in _CHO_OCCUPATION_ORDER:
            _, l_block = block
            Z -= 2 * _CHO_BASIS_SIZES[l_block].item()
            if Z <= 0:
                break

        if l_block < 2:
            l_values = [0, 1]
        elif l_block == 2:
            l_values = [0, 1, 2]
        else:
            l_values = [0, 1, 2, 3]

    return l_values


_ELEMENT_CHO_VALENCE_SHELLS = {z: _get_valence_shells(z) for z in range(1, 119)}

_CANONICAL_CHO_BASIS_NAMES = {
    0: [r"s"],
    1: [r"p_{y}", r"p_{z}", r"p_{x}"],
    2: [r"d_{xy}", r"d_{yz}", r"d_{z^2}", r"d_{xz}", r"d_{x^2 - y^2}"],
    3: [
        r"f_{y(3x^2 - y^2)}",
        r"f_{xyz}",
        r"f_{yz^2}",
        r"f_{z^3}",
        r"f_{xz^2}",
        r"f_{z(x^2 - y^2)}",
        r"f_{x(x^2-3y^2)}",
    ],
}
""" Map of l (angular momentum) -> names of orbitals
in cubic harmonic basis"""


def _get_cho_orbital_name(l: int, i: int):
    """Returns name of orbital with angular momentum l
    and index i (0 <= i < 2l+1)."""
    assert 0 <= l
    assert 0 <= i < 2 * l + 1

    l = int(l)
    if l in _CANONICAL_CHO_BASIS_NAMES:
        orbital = _CANONICAL_CHO_BASIS_NAMES[l][i]
    else:
        orb_symbol = chr(ord("f") + (l - 3)) if l <= 23 else f"({l})"
        orbital = orb_symbol + "_{" + str(i) + "}"

    return orbital
