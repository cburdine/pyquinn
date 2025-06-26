import torch

from math import factorial, sqrt, comb, gamma
from pyquinn.torch.util import sparse_coo_block_diag

GTO_L_MAX = 4
""" Maximum angular momentum :math:`l` supported by default for 
representations of gaussian atomic orbitals."""


def _make_flat_tensors(tensor_list: list):

    list_shapes = torch.vstack([
        torch.tensor(lt.shape)
        for lt in tensor_list
    ])
    list_sizes = list_shapes.prod(-1)

    list_idxs = torch.cumsum(list_sizes, dim=0) - list_sizes
    list_flat = torch.cat([lt.flatten() for lt in tensor_list])

    return (list_idxs, list_shapes, list_flat)


def _get_flat_tensor(data, i):

    idx_data, shape_data, flat_data = data
    idx = idx_data[i]
    shape = shape_data[i]

    return flat_data[idx:idx+shape.prod()].view(*shape.unbind())


def _make_gto_ijk_coeffs(l_max):
    """ Generates a tensor of coefficients corresponding to IJK values up
    to l=l_max"""
    return torch.tensor([
        factorial(i) / factorial(2*i)
        for i in range(l_max+1)
    ], requires_grad=False)


_GTO_IJK_COEFFS = _make_gto_ijk_coeffs(GTO_L_MAX)
""" Normalizing constant factors for ijk gaussian-type orbitals up to
l=GTO_L_MAX"""


def _make_potential_ijk_coeffs(l_max):
    """ Generates a tensor of coefficients corresponding to IJK values up
    to l=l_max"""
    return torch.tensor([
        1/gamma((i + 1)/2)
        for i in range(l_max+1)
    ])


_POTENTIAL_IJK_COEFFS = _make_potential_ijk_coeffs(GTO_L_MAX)
""" Normalizing constant factors for ijk l1 gaussian-type 
potentials up to l=GTO_L_MAX"""


def _make_gto_ijk_values(l):
    """ Generates a tensor of all IJK values wiht angular momentum l"""

    return torch.tensor([
        [l-(j+k), j, k]
        for k in range(l+1)
        for j in range(l+1-k)
    ], requires_grad=False)


_GTO_IJK_VALUES = [
    _make_gto_ijk_values(l)
    for l in range(GTO_L_MAX+1)
]
""" IJK values for each angular momentum  l up to l = GTO_L_MAX"""


def _make_gto_hermite_tensor(l_max):
    """ Generates a matrix of Hermite polynomial coefficients up to
    order n = 2*l+1"""
    n = 2*(l_max + 1)
    H = torch.zeros((n, n), requires_grad=False)
    for j in range(n):
        for p in range(j//2 + 1):
            H[j, j-2*p] = factorial(j) * (-1)**p * 2**(j-2*p) \
                / (factorial(p)*factorial(j-2*p))

    return H


_GTO_HERMITE_TENSOR = _make_gto_hermite_tensor(GTO_L_MAX)
""" Matrix of Hermite polynomial coefficients up to order n = 2*GTO_MAX + 1"""


def _make_gto_alpha_tensor(l_max):
    """ Generates a matrix of alpha coefficients for multiplying the product
    of Gaussian primitives up to l = GTO_L_MAX"""
    n = l_max+1
    A = torch.zeros((n, n), requires_grad=False)
    for l in range(n):
        norm_coeff = ((2)**(1/2) * 8**l * factorial(l) / factorial(2*l))**(1/2)
        for i in range(l//2 + 1):
            A[l, l-2*i] = norm_coeff * factorial(l) \
                / (2**l * factorial(i) * factorial(l-2*i))

    return A


_GTO_ALPHA_TENSOR = _make_gto_alpha_tensor(GTO_L_MAX)
""" Matrix of alphas coefficients up to l = GTO_L_MAX"""


def _make_gto_cho_tensor(l, ijks=None) -> torch.Tensor:
    """Generates a matrix mapping the basis of l2-normalized gaussian
    functions with i+j+k=l to the basis of orthonormal 2l+1 cubic
    yharmonics with angular momentum l

    Args:
        l (int): Angular momentum of gaussian basis functions (l = i+j+k)
        ijks (torch.Tensor, optional): Matrix of ordered ijk values
                                       (shape: [n,3]). If None, uses
                                       the default ordering of ijk values
                                       for the given l. Defaults to None.
        padded (bool): Whether or not to  return the tensor padded 

    Returns:
        torch.Tensor: Matrix of coefficients (shape: [2l+1, len(ijks)])
    """

    if ijks is None:
        ijks = _make_gto_ijk_values(l).detach().cpu().numpy()
    ijk_coeffs = _make_gto_ijk_coeffs(l)

    n_ijks = len(ijks)
    n_chos = 2*l+1
    X = torch.zeros((n_chos, n_ijks))

    ijk_idxs = {tuple(ijk): i for i, ijk in enumerate(ijks)}

    ijk_norm_coeffs = {
        tuple(ijk): (ijk_coeffs[torch.tensor(ijk)].prod(-1) *
                     factorial(2*l) / factorial(l))**(-1/2)
        for ijk in ijks
    }

    # generate unnormalized cubic harmonic matrix:
    for i, m_signed in enumerate(range(-l, l+1)):
        m = abs(m_signed)
        gamma_0 = 1.0 if m == 0 else sqrt(2*(factorial(l-m) / factorial(l+m)))
        for k in range((l-m)//2 + 1):
            N_lmk = (comb(l, k)*comb(2*l - 2*k, l) *
                     factorial(l-2*k)) / factorial(l + (-2*k) + (-m))
            gamma_lmk = gamma_0 * N_lmk * ((-1)**k / 2**l)

            for p in range(m+1):
                if m_signed == 0:
                    sigma_lmk = comb(m, p)
                elif (m_signed < 0):
                    # sigma_lmk equals: sin(torch.pi/2 * (m-p))*comb(m,p)
                    sigma_lmk = comb(m, p) * \
                        ((-1)**((m-p)//2) - (-1)**(((m-p)+1)//2)) // 2
                else:
                    # sigma_lmk equals: cos(torch.pi/2 * (m-p))*comb(m,p)
                    sigma_lmk = comb(m, p) * \
                        ((-1)**((m-p)//2) + (-1)**(((m-p)+1)//2)) // 2

                for s in range(k+1):
                    for t in range(s+1):
                        ijk = (p + 2*(k-s),      # i
                               (m-p)+2*(s-t),    # j
                               (l-2*k-m) + 2*t)  # k

                        assert ijk in ijks
                        N_ks = comb(k, s) * comb(s, t)
                        norm_coeff = ijk_norm_coeffs[ijk]
                        X[i, ijk_idxs[ijk]] += \
                            gamma_lmk*sigma_lmk*norm_coeff * N_ks

    return X


_GTO_CHO_TENSORS = [
    _make_gto_cho_tensor(l, _GTO_IJK_VALUES[l].detach().cpu().numpy())
    for l in range(GTO_L_MAX+1)
]
""" List of matrices mapping gaussian bases to cubic harmonics for
each l up to l=GTO_L_MAX."""


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

for l in range(GTO_L_MAX+1):
    e3_cho_tensor, e3_ijks = _make_gto_e3_cho_tensor(
        l, _GTO_CHO_TENSORS[l], 
        _GTO_IJK_VALUES[l])
    _GTO_E3_CHO_TENSORS.append(e3_cho_tensor)
    _GTO_E3_IJK_VALUES.append(e3_ijks)


_GTO_CHO_FLAT_TENSORS = _make_flat_tensors(_GTO_CHO_TENSORS)
_GTO_IJK_FLAT_TENSORS = _make_flat_tensors(_GTO_IJK_VALUES)


def gaussian_3d(
        x: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor) -> torch.Tensor:
    r"""Evaluates a 3D unnormalized gaussian primitive function.

    .. math::

        f(\mathbf{x}) = \exp(-\beta(\mathbf{x}-\boldsymbol{\mu})^2)

    Args:
        x (torch.Tensor): Points x in space (shap2e: [...,3])
        mu (torch.Tensor): Gaussian center (shape: [...,3])
        beta (torch.Tensor): Gaussian widths (shape: [...])

    Returns:
        torch.Tensor: Value of gaussians at points x (shape: [...,1])
    """
    return torch.exp(-beta*torch.sum((x-mu)**2, dim=-1, keepdim=True))


def l1_gaussian_3d(
        x: torch.Tensor,
        mu: torch.Tensor,
        gamma: torch.Tensor) -> torch.Tensor:
    r"""Evaluates a 3D l1-normalized gaussian primitive function.

    .. math::

        f(\mathbf{x}) = \left(\frac{\gamma}{\pi}\right)^{3/2}
        \exp(-\gamma(\mathbf{x}-\boldsymbol{\mu})^2)

    Args:
        x (torch.Tensor): Points x in space (shape: [...,3])
        mu (torch.Tensor): Gaussian center (shape: [...,3])
        gamma (torch.Tensor): Normalized gaussian widths 
                              (shape: [...])

    Returns:
        torch.Tensor: Value of gaussians at points x (shape: [...,1])
    """
    return (gamma/torch.pi)**(3/2) * \
        torch.exp(-gamma*torch.sum((x-mu)**2, dim=-1, keepdim=True))


def l2_gaussian_3d(x: torch.Tensor,
                   mu: torch.Tensor,
                   alpha: torch.Tensor) -> torch.Tensor:
    r"""Evaluates a 3D l2-normalized gaussian primitive function.

    .. math::

        f(\mathbf{x}) = \left(\frac{2\alpha}{\pi}\right)^{3/4}
        \exp(-\alpha(\mathbf{x}-\boldsymbol{\mu})^2)

    Args:
        x (torch.Tensor): Points x in space (shape: [...,3])
        mu (torch.Tensor): Gaussian center (shape: [...,3])
        alpha (torch.Tensor): Normalized gaussian widths 
                              (shape: [...])

    Returns:
        torch.Tensor: Value of gaussians at points x (shape: [...,1])
    """
    return (2*alpha/torch.pi)**(3/4) * \
        torch.exp(-alpha*torch.sum((x-mu)**2, dim=-1, keepdim=True))


def gto(x: torch.Tensor, 
        mu: torch.Tensor, 
        ijk: torch.IntTensor, 
        alpha: torch.Tensor) -> torch.Tensor:
    r"""Evaluates an l2-normalized cartesian gaussian-type Orbital (GTO) 
    function.

    .. math::

        \phi_\alpha^{ijk}(\mathbf{x}) = N(x-\mu_x)^i(y-\mu_y)^j(z-\mu_z)^k
        \exp(-\alpha(\mathbf{x}-\boldsymbol{\mu})^2)

    where :math:`N` is a normalization coefficient.

    A linear combination of these functions can be used to
    represent a cartesian Slater-type orbital (STO).

    Args:
        x (torch.Tensor): Points x in space (shape: [...,3])
        mu (torch.Tensor): Orbital center (shape: [3])
        ijk (torch.IntTensor): Cartesian powers of x,y,z (shape: [3])
        alpha (torch.Tensor): Normalized gaussian widths (shape: [3])

    Returns:
        torch.Tensor: Value of cartesian GTO at points x (shape: [...])
    """

    dx = x - mu
    psi = torch.exp(-alpha * ((dx)**2).sum(-1, keepdim=True)) \
        * (dx**(ijk)).prod(-1, keepdim=True) \
        * (2*alpha/torch.pi)**(3/4) \
        * (8*alpha)**(ijk.sum(-1, keepdim=True)/2) \
        * torch.sqrt((_GTO_IJK_COEFFS[ijk]).prod(-1, keepdim=True))

    return psi


def potential_gto(
        x: torch.Tensor, 
        mu: torch.Tensor, 
        ijk: torch.IntTensor, 
        gamma: torch.Tensor) -> torch.Tensor:
    r"""Evaluates an l1-normalized cartesian gaussian-type orbital
    function.

    .. math::

        F_\gamma^{ijk}(\mathbf{x}) = N(x-\mu_x)^i(y-\mu_y)^j(z-\mu_z)^k
        \exp(-\gamma(\mathbf{x}-\boldsymbol{\mu})^2)

    where :math:`N` is a normalization coefficient.

    A linear combination of these functions can be
    used to represent an effective core potential.

    Args:
        x (torch.Tensor): Points x in space (shape: [...,3])
        mu (torch.Tensor): Orbital center (shape: [3])
        ijk (torch.IntTensor): Cartesian powers of x,y,z (shape: [3])
        gamma (torch.Tensor): Normalized gaussian widths (shape: [3])

    Returns:
        torch.Tensor: Value of cartesian potential at points x (shape: [...])
    """

    dx = x - mu
    F = torch.exp(-gamma * ((dx)**2).sum(-1, keepdim=True)) \
        * (dx**(ijk)).prod(-1, keepdim=True) \
        * (gamma)**((ijk.sum(-1, keepdim=True)+3)/2) \
        * (_POTENTIAL_IJK_COEFFS[ijk]).prod(-1, keepdim=True)

    return F


def sto_ng(
        x: torch.Tensor, 
        mu: torch.Tensor, 
        ijk: torch.IntTensor, 
        alphas: torch.Tensor, 
        cs: torch.Tensor) -> torch.Tensor:
    r"""Evaluates a cartesian Slater-type orbital (STO) represented as the 
    contraction of :math:`N` primitive gaussians.

    .. math::

        \phi^{ijk}(\mathbf{x}) = \sum_{n=1}^N c_n 
        \phi_{\alpha_n}^{ijk}(\mathbf{x})

    The contraction coefficients `cs` are not normalized when :math:`\phi`
    is evaluated.

    Args:
        x (torch.Tensor): Points x in space (shape: [...,3])
        mu (torch.Tensor): Orbital center (shape: [3])
        ijk (torch.IntTensor): Cartesian powers of x,y,z (shape: [3])
        alphas (torch.Tensor): Gaussian widths (shape: [N])
        cs (torch.Tensor): Gaussian coeffficients (shape: [N])

    Returns:
        torch.Tensor: Value of the unnormalized orbital at points x 
                      (shape: [...])
    """

    gto_3d_component = torch.vmap(lambda a, c: c*gto(x, mu, ijk, a),
                                  in_dims=-1, out_dims=-1)

    psi = torch.sum(gto_3d_component(alphas, cs), axis=-1).squeeze(-1)

    return psi


def potential_ng(
        x: torch.Tensor,
        mu: torch.Tensor, 
        ijk: torch.IntTensor, 
        gammas: torch.Tensor, 
        vs: torch.Tensor) -> torch.Tensor:
    r"""Evaluates a cartesian effective potential function expanded as an
    weighted contraction of :math:`N` l1-normalized gaussians. 

    .. math::

        V^{ijk}(\mathbf{x}) = \sum_{n=1}^N v_n 
        F{\gamma_n}^{ijk}(\mathbf{x})

    The contraction coefficients `cs` are not normalized when :math:`\phi`
    is evaluated.

    For a radially symmetric potential, the value of
    (i,j,k) = (0,0,0) should be used.

    Args:
        x (torch.Tensor): Points x in space (shape: [...,3])
        mu (torch.Tensor): Potential center (shape: [3])
        ijk (torch.IntTensor): Cartesian powers of x,y,z (shape: [3])
        gammas (torch.Tensor): Potential Gaussian widths (shape: [N])
        vs (torch.Tensor): Potential Gaussian coeffficients (shape: [N])

    Returns:
        torch.Tensor: Value of the potential at points x.
    """

    v_3d_component = torch.vmap(lambda g, v: v*potential_gto(x, mu, ijk, g),
                                in_dims=-1, out_dims=-1)

    V = torch.sum(v_3d_component(gammas, vs), axis=-1).squeeze(-1)

    return V


def sto_ng_primitive_overlap_matrix(
        mu_1: torch.Tensor, alphas_1: torch.Tensor, 
        mu_2: torch.Tensor, alphas_2: torch.Tensor) -> torch.Tensor:

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4)

    sqrt_etas = torch.sqrt(_alphas_1 * _alphas_2 / (_alphas_1 + _alphas_2))

    _mu_1 = mu_1.unsqueeze(-2)
    _mu_2 = mu_2.unsqueeze(-3)
    dx = (_mu_1 - _mu_2)

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1])

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        '...abmnj, jp, ...abmnp, ...abdp -> ...abmndj',
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR,
        sqrt_eta_powers, r_powers)

    alphas_1_powers = torch.arange(
        _GTO_ALPHA_TENSOR.shape[-1],
        device=alphas_1.device)
    alphas_2_powers = torch.arange(
        _GTO_ALPHA_TENSOR.shape[-1],
        device=alphas_2.device)
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    sqrt_alphas_1 = torch.sqrt(alphas_1).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    eta_gaussian_exponents = torch.einsum(
        '...abmn, ...abd -> ...abdmn',
        sqrt_etas**2, dx**2)

    eta_gaussians = torch.einsum(
        '...abdmn, ...abmn -> ...abdmn',
        torch.exp(-eta_gaussian_exponents),
        sqrt_etas / ((_alphas_1 * _alphas_2)**(1/4)))

    A_1 = (sqrt_alphas_1 ** (-alphas_1_powers.unsqueeze(0))) \
        * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = (sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0)))
    J_12 = hermite_coeffs[..., j_powers]

    T = torch.einsum(
        'ij, ...amj, ...abmndjk, ...abdmn, ...bnk, kl -> ...abmndil',
        _GTO_ALPHA_TENSOR,
        A_1, J_12, eta_gaussians, A_2,
        _GTO_ALPHA_TENSOR.T)

    return T


def sto_ng_block_primitive_overlap_matrix(
        mu_1: torch.Tensor, alphas_1: torch.Tensor, 
        mu_2: torch.Tensor, alphas_2: torch.Tensor) -> torch.Tensor:
    r"""Evaluates the primitive overlap matrices for contracted slater-type
    orbitals expanded as the sum of :math:`N` gaussians. This overlap matrix 
    has the shape [...,N_1,N_2,3,L+1,L+1], where L is is the maximum allowed 
    angular momentum 
    (equal to :attr:`GTO_L_MAX <pyquinn.torch.lcao.gto.GTO_L_MAX>` by default).

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

    This function also supports the batch pairwise evaluation of overlap
    integrals between sets of contracted gaussians, with set sizes of
    B_1, and B_2 respectively. If batch pairwise evaluation is not 
    desired, the tensors `alphas_1` and `alphas_2` should have a first 
    dimension of B_1 = B_2 = 1.

    Note that any leading dimensions to the tensors `mu_1`
    and `mu_2` are not evaluated in this batch-pairwise fashion.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., 3])
        alphas_1 (torch.Tensor): Gaussian widths of orbital 1 
                                 (shape: [..., B_1, N_1])
        mu_2 (torch.Tensor): Center of Noteorbital 2 (shape: [..., 3])
        alphas_2 (torch.Tensor): Gaussian widths of orbital 2 
                                 (shape: [..., B_2, N_2])

    Returns:
        torch.Tensor: Overlap matrix tensor T.
                      (shape: [..., B_1, B_2, N_1, N_2, 3, L+1, L+1])
    """

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4)
    sqrt_etas = torch.sqrt(_alphas_1 * _alphas_2 / (_alphas_1 + _alphas_2))

    dx = (mu_1 - mu_2)

    hermite_powers = torch.arange(
        _GTO_HERMITE_TENSOR.shape[-1],
        device=sqrt_etas.device)

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        '...abmnj, jp, ...abmnp, ...dp -> ...abmndj',
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR.to(sqrt_eta_powers.device),
        sqrt_eta_powers, r_powers)

    alphas_1_powers = torch.arange(
        _GTO_ALPHA_TENSOR.shape[-1],
        device=alphas_1.device)
    alphas_2_powers = torch.arange(
        _GTO_ALPHA_TENSOR.shape[-1],
        device=alphas_2.device)
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    sqrt_alphas_1 = torch.sqrt(alphas_1).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    eta_gaussian_exponents = torch.einsum(
        '...abmn, ...d -> ...abdmn',
        sqrt_etas**2, dx**2)

    eta_gaussians = torch.einsum(
        '...abdmn, ...abmn -> ...abdmn',
        torch.exp(-eta_gaussian_exponents),
        sqrt_etas / ((_alphas_1 * _alphas_2)**(1/4)))

    A_1 = (sqrt_alphas_1 ** (-alphas_1_powers.unsqueeze(0))) \
        * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = (sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0)))
    J_12 = hermite_coeffs[..., j_powers]

    _ALPHA = _GTO_ALPHA_TENSOR.to(eta_gaussians.device)
    T = torch.einsum(
        'ij, ...amj, ...abmndjk, ...abdmn, ...bnk, kl -> ...abmndil',
        _ALPHA,
        A_1, J_12, eta_gaussians, A_2,
        _ALPHA.T)

    return T


def sto_ng_overlap_matrix(
        mu_1: torch.Tensor, ijk_1: torch.IntTensor,
        alphas_1: torch.Tensor, 
        alpha_idxs_1: torch.IntTensor,
        cs_1: torch.Tensor,
        mu_2: torch.Tensor, ijk_2: torch.IntTensor,
        alphas_2: torch.Tensor, 
        alpha_idxs_2: torch.IntTensor,
        cs_2: torch.IntTensor) -> torch.Tensor:

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
    T = sto_ng_primitive_overlap_matrix(
        _mu_1, _alphas_1, _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device)\
        .view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device)\
        .view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2]\
        .prod(-5)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum('...im, ...ijmn, ...jn -> ...ij', _cs_1, T_ijk, _cs_2)\
        .reshape(*mu_batch_shape, i1_size, i2_size)

    return M


def sto_ng_block_overlap_matrix(
        mu_1: torch.Tensor, ijk_1: torch.IntTensor,
        alphas_1: torch.Tensor, 
        alpha_idxs_1: torch.IntTensor,
        cs_1: torch.Tensor,
        mu_2: torch.Tensor, ijk_2: torch.IntTensor,
        alphas_2: torch.Tensor, 
        alpha_idxs_2: torch.IntTensor,
        cs_2: torch.IntTensor) -> torch.Tensor:
    r"""Evaluates the cartesian GTO pairwise orbital overlap matrix 
    between all orbitals on adjacent sites.

    .. math::

        M_{i,j} = \langle \phi_{1,i} | \phi_{2,j} \rangle

    Since each site may have multiple orbitals (indexed by i and j) that share
    the same alpha values, pairwise batching of the overlap integrals between
    sites is supported. This allows for the passing of a smaller list of alphas 
    (of size B_1 and B_2), which are then indexed in each orbital by 
    `alpha_idxs+1` and `alpha_idxs_2`, which are arrays of length I_1 > B_1 
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
        cs_1 (torch.Tensor): Gaussian coeffficients of orbital 1 
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
        cs_2 (torch.IntTensor): Gaussian coeffficients of orbital 2 
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
    T = sto_ng_block_primitive_overlap_matrix(
        _mu_1, _alphas_1, _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)

    _dim = torch.arange(3, device=ijk_1.device)\
        .view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device)\
        .view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2]\
        .prod(-5)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum('...im, ...ijmn, ...jn -> ...ij', _cs_1, T_ijk, _cs_2)\
        .reshape(*mu_batch_shape, i1_size, i2_size)

    return M


def sto_ng_primitive_potential_overlap_matrix(
        mu_1: torch.Tensor, alphas_1: torch.Tensor,
        gammas_1: torch.Tensor,
        mu_2: torch.Tensor, alphas_2: torch.Tensor) -> torch.Tensor:

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3).unsqueeze(-5)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4).unsqueeze(-5)
    _gammas_1 = gammas_1.view(*(gammas_1.shape), 1, 1, 1, 1)
    sqrt_etas = torch.sqrt(
        (_alphas_1 + _gammas_1) * _alphas_2 / 
        (_alphas_1 + _gammas_1 + _alphas_2))

    _mu_1 = mu_1.unsqueeze(-2)
    _mu_2 = mu_2.unsqueeze(-3)
    dx = (_mu_1 - _mu_2)

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1])

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        '...gabmnj, jp, ...gabmnp, ...gabdp -> ...gabmndj',
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR.to(sqrt_eta_powers.device),
        sqrt_eta_powers, r_powers)

    alphas_1_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1])
    alphas_2_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1])
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    eta_gaussian_exponents = torch.einsum(
        '...gabmn, ...gabd -> ...gabdmn',
        sqrt_etas**2, dx**2)

    N_12 = torch.sqrt(
        (torch.sqrt(_alphas_1*_alphas_2) * _gammas_1) /
        (torch.pi * (_alphas_1 + _alphas_2 + _gammas_1)))

    eta_gaussians = torch.einsum(
        '...gabdmn, ...gabmn -> ...gabdmn',
        torch.exp(-eta_gaussian_exponents),
        N_12,
    )

    sqrt_alphas_gammas_1 = torch.sqrt(
        alphas_1.unsqueeze(-3) + gammas_1.unsqueeze(-1).unsqueeze(-2)
    ).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    g_alphas_1 = alphas_1.unsqueeze(-3)
    g_gammas_1 = gammas_1.unsqueeze(-1).unsqueeze(-2)

    G_1 = torch.sqrt(g_alphas_1 / (g_alphas_1 + g_gammas_1)).unsqueeze(-1)\
        ** alphas_1_powers

    A_1 = (sqrt_alphas_gammas_1 ** (-alphas_1_powers.unsqueeze(0))) \
        * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = (sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0)))
    J_12 = hermite_coeffs[..., j_powers]

    _ALPHA = _GTO_ALPHA_TENSOR.to(device=eta_gaussians.device)
    T = torch.einsum(
        '...gami, ij, ...gamj, ...gabmndjk, ...gabdmn, '
        '...bnk, kl -> ...gabmndil',
        G_1,
        _ALPHA,
        A_1, J_12, eta_gaussians, A_2,
        _ALPHA.T)

    return T


def sto_ng_block_primitive_potential_overlap_matrix(
        mu_1: torch.Tensor, alphas_1: torch.Tensor,
        gammas_1: torch.Tensor,
        mu_2: torch.Tensor, alphas_2: torch.Tensor) -> torch.Tensor:
    r"""Evaluates the primitive overlap matrices for contracted slater-type
    orbitals expanded as the sum of :math:`N` gaussians under the influence of
    an effective cartesian potential centered at the first site. This overlap 
    matrix has the shape [..., M_1, N_1,N_2,3,L+1,L+1], where L is is the 
    maximum allowed angular momentum. 
    (equal to :attr:`GTO_L_MAX <pyquinn.torch.lcao.gto.GTO_L_MAX>` by default).

    The potential function :math:`V_1(\vec{r})` is expanded as the sum of 
    :math:`M` primitive Gaussians.

    For any two sets of cartesian GTOs :math:`\phi_1, \phi_2` with powers 
    :math:`(i_1, j_1, k_1)` and :math:`(i_2,j_2,k_2)`, the returned tensor
    :math:`T` satisfies:

    .. math::

        \langle \phi_1 | V_1 | \phi_2 \rangle = 
        \sum_{n_1=1}^{N_1}\sum_{n_2=1}^{N_2} v_m c_{n_1} c_{n_2} 
        (T_{m, n_1, n_2, 1, i_1, i_2} 
            \odot T_{m, n_1, n_2, 2, j_1, j_2} 
            \odot T_{m, n_2, n_2, 3, k_1, k_2})

    where the product :math:`\odot` (the Hadamard product) is computed 
    elementwise.

    This function also supports the batch pairwise evaluation of overlap
    integrals between sets of contracted gaussians, with set sizes of
    B_1, and B_2 respectively. If batch pairwise evaluation is not 
    desired, the tensors `alphas_1` and `alphas_2` should have a first 
    dimension of B_1 = B_2 = 1.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [..., 3])
        alphas_1 (torch.Tensor): Gaussian widths of orbital 1 
                                 (shape: [..., B_1, N_1])
        gammas_1 (torch.Tensor): Gaussian widths of potential 1
                                 (shape: [..., M_1])
        mu_2 (torch.Tensor): Center of orbital 2 (shape: [..., 3])
        alphas_2 (torch.Tensor): Gaussian widths of orbital 2 
                                 (shape: [..., B_2, N_2])

    Returns:
        torch.Tensor: Overlap matrix tensor T.
                      (shape: [..., M_1, B_1, B_2, N_1, N_2, 3, L+1, L+1])
    """

    _alphas_1 = alphas_1.unsqueeze(-1).unsqueeze(-3).unsqueeze(-5)
    _alphas_2 = alphas_2.unsqueeze(-2).unsqueeze(-4).unsqueeze(-5)
    _gammas_1 = gammas_1.view(*(gammas_1.shape), 1, 1, 1, 1)
    sqrt_etas = torch.sqrt(
        (_alphas_1 + _gammas_1) * _alphas_2 / 
        (_alphas_1 + _gammas_1 + _alphas_2))

    dx = (mu_1 - mu_2)

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1])

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = torch.einsum(
        '...gabmnj, jp, ...gabmnp, ...dp -> ...gabmndj',
        sqrt_eta_powers,
        _GTO_HERMITE_TENSOR.to(device=sqrt_eta_powers.device),
        sqrt_eta_powers, r_powers)

    alphas_1_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1])
    alphas_2_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1])
    j_powers = alphas_1_powers.view(-1, 1) + alphas_2_powers.view(1, -1)

    eta_gaussian_exponents = torch.einsum(
        '...gabmn, ...d -> ...gabdmn',
        sqrt_etas**2, dx**2)

    N_12 = torch.sqrt(
        (torch.sqrt(_alphas_1*_alphas_2) * _gammas_1) /
        (torch.pi * (_alphas_1 + _alphas_2 + _gammas_1)))

    eta_gaussians = torch.einsum(
        '...gabdmn, ...gabmn -> ...gabdmn',
        torch.exp(-eta_gaussian_exponents),
        N_12,
    )

    sqrt_alphas_gammas_1 = torch.sqrt(
        alphas_1.unsqueeze(-3) + gammas_1.unsqueeze(-1).unsqueeze(-2)
    ).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    g_alphas_1 = alphas_1.unsqueeze(-3)
    g_gammas_1 = gammas_1.unsqueeze(-1).unsqueeze(-2)

    G_1 = torch.sqrt(g_alphas_1 / (g_alphas_1 + g_gammas_1)).unsqueeze(-1)\
        ** alphas_1_powers

    A_1 = (sqrt_alphas_gammas_1 ** (-alphas_1_powers.unsqueeze(0))) \
        * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = (sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0)))
    J_12 = hermite_coeffs[..., j_powers]

    _ALPHA = _GTO_ALPHA_TENSOR.to(eta_gaussians.device)
    T = torch.einsum(
        '...gami, ij, ...gamj, ...gabmndjk, ...gabdmn, '
        '...bnk, kl -> ...gabmndil',
        G_1,
        _ALPHA,
        A_1, J_12, eta_gaussians, A_2,
        _ALPHA.T)

    return T


def sto_ng_potential_overlap_matrix(
        mu_1: torch.Tensor, ijk_1: torch.IntTensor,
        alphas_1: torch.Tensor, 
        alpha_idxs_1: torch.IntTensor,
        cs_1: torch.Tensor,
        gammas_1: torch.Tensor, vs_1: torch.Tensor,
        mu_2: torch.Tensor, ijk_2: torch.IntTensor,
        alphas_2: torch.Tensor,
        alpha_idxs_2: torch.IntTensor, 
        cs_2: torch.IntTensor) -> torch.Tensor:
    
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
    T = sto_ng_primitive_potential_overlap_matrix(
        _mu_1, _alphas_1, 
        _gammas_1,
        _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _dim = torch.arange(3, device=ijk_1.device)\
        .view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device)\
        .view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)
    _vs_1 = vs_1.reshape(mu_batch_size, -1)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, :, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2]\
        .prod(-6)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum(
        '...im, ...g, ...ijgmn, ...jn -> ...ij', 
        _cs_1, _vs_1, T_ijk, _cs_2)\
        .reshape(*mu_batch_shape, i1_size, i2_size)

    return M


def sto_ng_block_potential_overlap_matrix(
        mu_1: torch.Tensor, ijk_1: torch.IntTensor,
        alphas_1: torch.Tensor, 
        alpha_idxs_1: torch.IntTensor,
        cs_1: torch.Tensor,
        gammas_1: torch.Tensor, vs_1: torch.Tensor,
        mu_2: torch.Tensor, ijk_2: torch.IntTensor,
        alphas_2: torch.Tensor, 
        alpha_idxs_2: torch.IntTensor, 
        cs_2: torch.IntTensor) -> torch.Tensor:
    r"""Evaluates the cartesian GTO pairwise orbital overlap matrix 
    between all orbitals on adjacent sites under the influence of a
    potential at the first site

    .. math::

        M_{i,j} = \langle \phi_{1,i} | V_1 | \phi_{2,j} \rangle

    Since each site may have multiple orbitals (indexed by i and j) that share
    the same alpha values, pairwise batching of the overlap integrals between
    sites is supported. This allows for the passing of a smaller list of alphas 
    (of size B_1 and B_2), which are then indexed in each orbital by 
    `alpha_idxs+1` and `alpha_idxs_2`, which are arrays of length I_1 > B_1 
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
        cs_1 (torch.Tensor): Gaussian coeffficients of orbital 1 exa  orbital 2
                                 (shape: [..., B_2, N_2])
        alpha_idxs_2 (torch.IntTensor): Batch indexes i (0 < i < B_2) of the 
                                        evaluated set of alphas. If batching
                                        of alphas is not used, should be all
                                        zeros (shape: [...,I_2])
        cs_2 (torch.IntTensor): Gaussian coeffficients of orbital 2 
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
    _gammas_1 = gammas_1.reshape(mu_batch_size, -1)

    # calculate primitive overlap matrix
    T = sto_ng_block_primitive_potential_overlap_matrix(
        _mu_1, _alphas_1, 
        _gammas_1,
        _mu_2, _alphas_2)

    # flatten leading dimensions of indexing tensors
    _ijk_1 = ijk_1.transpose(-2, -1).reshape(mu_batch_size, 3, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _ijk_2 = ijk_2.transpose(-2, -1).reshape(mu_batch_size, 3, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _alpha_idxs_1 = alpha_idxs_1.reshape(mu_batch_size, 1, -1, 1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)
    _alpha_idxs_2 = alpha_idxs_2.reshape(mu_batch_size, 1, 1, -1)\
        .expand(mu_batch_size, 3, i1_size, i2_size)

    _dim = torch.arange(3, device=ijk_1.device)\
        .view(1, 3, 1, 1).expand(*_ijk_1.shape)
    _i = torch.arange(mu_batch_size, device=ijk_1.device)\
        .view(-1, 1, 1, 1).expand(*_ijk_1.shape)

    _cs_1 = cs_1.reshape(mu_batch_size, i1_size, -1)
    _cs_2 = cs_2.reshape(mu_batch_size, i2_size, -1)
    _vs_1 = vs_1.reshape(mu_batch_size, -1)

    # retrieve necessary indices from the T tensor
    T_ijk = T[_i, :, _alpha_idxs_1, _alpha_idxs_2, :, :, _dim, _ijk_1, _ijk_2]\
        .prod(-6)

    # compute the weighted sum of orbital overlaps
    M = torch.einsum(
        '...im, ...g, ...ijgmn, ...jn -> ...ij', 
        _cs_1, _vs_1, T_ijk, _cs_2)\
        .reshape(*mu_batch_shape, i1_size, i2_size)

    return M


def sto_ng_norm_coeff(
        ijk: torch.IntTensor, 
        alphas: torch.Tensor, 
        alpha_idxs: torch.IntTensor, 
        cs: torch.Tensor) -> torch.Tensor:
    r"""Evaluates the normalization coefficients of cartesian GTOs.

    Args:
        ijk (torch.IntTensor): Cartesian powers of x,y,z for the orbitals
                               (shape: [..., I, 3])
        alphas (torch.Tensor): Gaussian widths of the orbitals
                               (shape: [..., B, N])
        alpha_idxs (torch.IntTensor): Batch indexes i (0 <= i < B) of the 
                                      evaluated set of alphas. If batching
                                      of alphas is not used, should be all
                                      zeros (shape: [...,I])
        cs (torch.Tensor): Gaussian coeffficients of orbitals 
                           (shape: [..., I, N])

    Returns:
        torch.Tensor: The normalization constants of the orbitals
                      coefficients (shape: [..., I]).
    """

    mu_batch_shape = ijk.shape[:-2]
    mu_batch_size = torch.prod(torch.tensor(mu_batch_shape, dtype=torch.int))
    i_size = ijk.shape[-2]

    mu = torch.zeros((mu_batch_size, 3), device=alphas.device)
    T = sto_ng_block_primitive_overlap_matrix(mu, alphas, mu, alphas)

    _ijk = ijk.transpose(-2, -1).reshape(mu_batch_size, 3, -1)\
        .expand(mu_batch_size, 3, i_size)
    _alpha_idxs = alpha_idxs.reshape(mu_batch_size, 1, -1)\
        .expand(*_ijk.shape)
    _dim = torch.arange(3, device=ijk.device)\
        .view(1, 3, 1).expand(*_ijk.shape)

    _i = torch.arange(mu_batch_size, device=ijk.device)\
        .view(-1, 1, 1).expand(*_ijk.shape)

    _cs = cs.reshape(mu_batch_size, i_size, -1)

    T_ijk = T[_i, _alpha_idxs, _alpha_idxs, :, :, _dim, _ijk, _ijk].prod(-4)

    C_norm = torch.sqrt(torch.einsum('...im, ...imn, ...in -> ...i',
                                     _cs, T_ijk, _cs))

    return C_norm.reshape(*mu_batch_shape, i_size)


def cho_ng(
        x: torch.Tensor, mu: torch.Tensor, l: torch.IntTensor,
        m: torch.IntTensor, alphas: torch.Tensor, 
        cs: torch.Tensor) -> torch.Tensor:
    r"""Evaluates a cubic harmonic orbital with the radial function represented 
    as powers of the radius times a contraction of :math:`N` primitive 
    gaussians.

    The contraction coefficients `cs` are not normalized when the orbital is
    evaluated.

    Args:
        x (torch.Tensor): Points x in space (shape: [...,3])
        mu (torch.Tensor): Orbital center (chaps: [3])
        l (torch.IntTensor): Angular quantum numbers (shape: [1])
        m (torch.IntTensor): Magnetic quantum numbers (shape: [1])
        alphas (torch.Tensor): Gaussian widths (shape: [N])
        cs (torch.Tensor): Gaussian coefficients (shape: [N])

    Returns:
        torch.Tensor: Value of the unnormalized orbital at points x 
                      (shape: [...])
    """

    m_idx = l + m
    X_l = _GTO_CHO_TENSORS[l]  # _get_flat_tensor(_GTO_CHO_FLAT_TENSORS, l)
    X_lm = X_l[m_idx].view(-1)

    X_lm_idx, = torch.nonzero(X_lm, as_tuple=True)
    X_ijks = _GTO_IJK_VALUES[l]  # _get_flat_tensor(_GTO_IJK_FLAT_TENSORS, l)
    X_lm_ijks = X_ijks[X_lm_idx].view(-1, 3)
    X_lm_coeffs = X_lm[X_lm_idx].unsqueeze(-1)

    sto_3d_component = torch.vmap(
        lambda h, ijk, : h*sto_ng(x, mu, ijk, alphas, cs),
        in_dims=-2, out_dims=-1)

    psi = torch.sum(sto_3d_component(X_lm_coeffs, X_lm_ijks), axis=-1)\
        .squeeze(-1)

    return psi


def cho_ng_norm_coeff(
        l: torch.IntTensor, alphas: torch.Tensor, 
        alpha_idxs: torch.IntTensor, 
        cs: torch.Tensor) -> torch.Tensor:
    """Evaluates the normalization coefficients of gaussian cubic
    harmonic orbitals.

    Args:
        l (torch.IntTensor): Angular quantum number for the orbitals 
                             (shape: [..., I])
        alphas (torch.Tensor): Gaussian widths of the orbitals
                               (shape: [..., B, N])
        alpha_idxs (torch.IntTensor): Batch indexes i (0 <= i < B) of the 
                                      evaluated set of alphas. If batching
                                      of alphas is not used, should be all
                                      zeros (shape: [...,I])
        cs (torch.Tensor): Gaussian coeffficients of orbitals 
                           (shape: [..., I, N])

    Returns:
        torch.Tensor: The normalization constants of the orbitals
                      coefficients (shape: [..., I]).
    """
    l_zeros = torch.zeros_like(l)
    ijk = torch.stack((l, l_zeros, l_zeros), dim=-1)

    return sto_ng_norm_coeff(ijk, alphas, alpha_idxs, cs)


def cho_overlap_matrix_metadata(
        l_i: torch.Tensor,
        sparse: bool = False) -> tuple:
    """Computes the necessary metadata tensors for a single dimension 
    (i.e. rows or columns) of a cubic harmonic orbital overlap matrix.

    Args:
        l_i (torch.IntTensor): A 1D sequence of angular momenta `l`
                            in the overlap matrix (shape: [I]).
        sparse (bool): If True, returns a sparse representation of
                       the cho_tensor, which is recommended if l_i
                       is long and contains large l values. Defaults
                       to False.

    Returns:
        tuple: A tuple containing the following:

        - cho_tensor (torch.Tensor): 
            A matrix representing the transformation 
            from cartesian to CHO orbitals.
        - l_ijk (torch.IntTensor): 
            A list of all cartesian ijk values for which 
            overlap orbitals are computed.
        - l_alpha_idxs (torch.IntTensor): 
            A list of indices corresponding to the alpha 
            values indexed for each block.
        - l_block_idxs (torch.IntTensor): 
            A tensor of starting indices for
            each block in the overlap matrix.
    """

    # dynamically construct block cho transformation matrix
    cho_tensor_blocks = (_GTO_CHO_TENSORS[l.item()] for l in l_i)

    if sparse:
        cho_tensor = sparse_coo_block_diag(*cho_tensor_blocks)
    else:
        cho_tensor = torch.block_diag(*cho_tensor_blocks)

    # dynamically construct ijk indices
    l_ijk = torch.cat([_GTO_IJK_VALUES[l.item()] for l in l_i], dim=-2)

    # dynamically construct block alpha indices and coefficients
    l_alpha_idxs = torch.cat([
        torch.tensor([i]*_GTO_IJK_VALUES[l.item()].shape[0])
        for i, l in enumerate(l_i)
    ], dim=-1)

    # dynamically construct overlap matrix block indices
    l_block_sizes = torch.tensor([
        _GTO_CHO_TENSORS[l.item()].shape[0]
        for l in l_i
    ])
    l_block_idxs = torch.cumsum(l_block_sizes, dim=-1) - l_block_sizes

    return cho_tensor, l_ijk, l_alpha_idxs, l_block_idxs


def e3_cho_overlap_matrix_metadata(
        l_i: torch.Tensor,
        sparse: bool = True) -> tuple:
    """Computes the necessary metadata tensors for a single dimension
       (i.e. rows or columns) of a cubic harmonic orbital overlap matrix
       where the orbital coefficients are equivariant, meaning they 
       vary within each block.

    Args:
        l_i (torch.IntTensor): A 1D sequence of angular momenta `l`
                            in the overlap matrix (shape: [I]).
        sparse (bool, optional): If True, returns a sparse representation of
                       the cho_tensor, which is recommended if l_i
                       is long and contains large l values. Defaults
                       to False.. Defaults to True.

    Returns:
        tuple: A tuple containing the following:

        - cho_tensor (torch.Tensor): 
            A matrix representing the transformation 
            from cartesian to CHO orbitals.
        - l_ijk (torch.IntTensor): 
            A list of all cartesian ijk values for which 
            overlap orbitals are computed.
        - l_alpha_idxs (torch.IntTensor): 
            A list of indices corresponding to the alpha 
            values indexed for each block.
        - l_block_idxs (torch.IntTensor): 
            A tensor of starting indices for
            each block in the overlap matrix.
        - c_block_tensor (torch.IntTensor)
    """

    # dynamically construct block cho transformation matrix
    cho_tensor_blocks = (_GTO_E3_CHO_TENSORS[l.item()] for l in l_i)

    if sparse:
        cho_tensor = sparse_coo_block_diag(*cho_tensor_blocks)
    else:
        cho_tensor = torch.block_diag(*cho_tensor_blocks)

    # dynamically construct ijk indices
    l_ijk = torch.cat([_GTO_E3_IJK_VALUES[l.item()] for l in l_i], dim=-2)

    # dynamically construct block alpha indices and coefficients
    l_alpha_idxs = torch.cat([
        torch.tensor([i]*_GTO_E3_IJK_VALUES[l.item()].shape[0])
        for i, l in enumerate(l_i)
    ], dim=-1)

    # dynamically construct overlap matrix block indices
    c_block_sizes = torch.tensor([
        _GTO_E3_CHO_TENSORS[l.item()].shape[0]
        for l in l_i
    ])
    c_idxs = torch.cumsum(c_block_sizes, dim=0) - c_block_sizes

    # dynamically construct block c indices:
    l_c_idxs = torch.cat([
        c_idx + _GTO_E3_CHO_TENSORS[l.item()].nonzero()[:, 0]
        for l, c_idx in zip(l_i, c_idxs)
    ])

    return cho_tensor, l_ijk, l_alpha_idxs, l_c_idxs


def _cho_block_data(l_i: torch.Tensor, 
                    l_cs_i: torch.Tensor,
                    cho_data: tuple = None) -> tuple:
    """ Builds the data necessary for a cho block overlap matrix. The
        alpha and c values are padded to fit the batch size."""

    batch_shape = l_cs_i.shape[:-2]
    batch_ndim = l_cs_i.ndim - 2

    cho_tensor, l_ijk, l_alpha_idxs, _ = \
        cho_data if cho_data is not None else \
        cho_overlap_matrix_metadata(l_i)

    l_block_cs = l_cs_i[..., l_alpha_idxs, :]

    # pad alpha indices and coefficients to batch size
    l_alpha_idxs = l_alpha_idxs\
        .view(*([1]*batch_ndim), -1)\
        .expand(*batch_shape, -1)

    l_ijk = l_ijk\
        .view(*([1]*batch_ndim), -1, 3)\
        .expand(*batch_shape, -1, 3)

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
        cho_data_2: tuple = None) -> torch.Tensor:
    """Evaluates the cubic harmonic GTO pairwise orbital overlap matrix 
    between different sites with different angular momenta blocks. 

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
        torch.Tensor: Pairwise overlap matrix of orbitals 
                      (shape: [H_1, H_2]).
    """
    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = \
        cho_data_1 if cho_data_1 is not None else \
        cho_overlap_matrix_metadata(l_1)

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = \
        cho_data_2 if cho_data_2 is not None else \
        cho_overlap_matrix_metadata(l_2)

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]
    
    M = sto_ng_overlap_matrix(
        mu_1, l_ijk_1,
        l_alphas_1, l_alpha_idxs_1, l_block_cs_1,
        mu_2, l_ijk_2,
        l_alphas_2, l_alpha_idxs_2, l_block_cs_2)

    M_X = cho_tensor_1 @ M @ cho_tensor_2.T

    return M_X


def cho_ng_potential_overlap_matrix(
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
        cho_data_2: tuple = None) -> torch.Tensor:
    """Evaluates the cubic harmonic GTO pairwise orbital overlap matrix 
    between blocks of orbitals under the influence of 
    a cartesian potential function centered at the first orbital. 
    The blocks are evaluated for a fixed set of angular momenta l.

    Args:
        mu_1 (torch.Tensor): Center of orbital 1 (shape: [I_1, 3])
        l_1 (torch.IntTensor): Angular momenta of blocks for orbital 1
                               (shape: [I_1])
        l_alphas_1 (torch.Tensor): Gaussian widths of blocks for orbital 1
                                   (shape: [I_1, N_1])
        l_cs_1 (torch.Tensor): Gaussian coefficients of orbital 1
                               (shape: [I_1, N_1])
        gammas_1: (torch.Tensor): Gaussian widths of potential 1
                                  (shape: [M_1])
        vs_1: (torch.Tensor): Gaussian coefficients of potential 1
                                  (shape: [M_1])
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
        torch.Tensor: Pairwise overlap matrix
                      (shape: [H_1, H_2]).
    """
    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, _ = \
        cho_data_1 if cho_data_1 is not None else \
        cho_overlap_matrix_metadata(l_1)

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, _ = \
        cho_data_2 if cho_data_2 is not None else \
        cho_overlap_matrix_metadata(l_2)

    l_block_cs_1 = l_cs_1[l_alpha_idxs_1, :]
    l_block_cs_2 = l_cs_2[l_alpha_idxs_2, :]

    M = sto_ng_potential_overlap_matrix(
        mu_1, l_ijk_1,
        l_alphas_1, l_alpha_idxs_1, l_block_cs_1,
        gammas_1, vs_1,
        mu_2, l_ijk_2,
        l_alphas_2, l_alpha_idxs_2, l_block_cs_2)

    M_X = cho_tensor_1 @ M @ cho_tensor_2.T

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
        e3_cho_data_2: tuple = None) -> torch.Tensor:
    """Evaluates the cubic harmonic GTO pairwise orbital overlap matrix 
    between different sites with different angular momenta blocks.
    The blocks are evaluated for a set of alpha coefficients that are
    fixed for each angular momentum value l, but where the coefficients 
    `e3_cs_1` and `e3_cs_2` are different for each orbital in the block.

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
    e3_cho_tensor_1, e3_ijk_1, e3_alpha_idxs_1, e3_c_idxs_1 = \
        e3_cho_data_1 if e3_cho_data_1 is not None else \
        e3_cho_overlap_matrix_metadata(l_1)

    e3_cho_tensor_2, e3_ijk_2, e3_alpha_idxs_2, e3_c_idxs_2 = \
        e3_cho_data_2 if e3_cho_data_2 is not None else \
        e3_cho_overlap_matrix_metadata(l_2)

    cs_1 = e3_cs_1[e3_c_idxs_1]
    cs_2 = e3_cs_2[e3_c_idxs_2]

    M = sto_ng_overlap_matrix(
        mu_1, e3_ijk_1,
        l_alphas_1, e3_alpha_idxs_1, cs_1,
        mu_2, e3_ijk_2,
        l_alphas_2, e3_alpha_idxs_2, cs_2)

    M_X = e3_cho_tensor_1 @ M @ e3_cho_tensor_2.T

    return M_X


def e3_cho_ng_potential_overlap_matrix(
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
        e3_cho_data_2: tuple = None) -> torch.Tensor:
    """Evaluates the cubic harmonic GTO pairwise orbital overlap matrix 
    between blocks of orbitals under the influence of 
    a cartesian potential function centered at the first orbital. 
    The blocks are evaluated for a set of alpha coefficients that are
    fixed for each angular momentum value l, but where the coefficients 
    `e3_cs_1` and `e3_cs_2` are different for each orbital in the block.

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
        
    Returns:
        torch.Tensor: Pairwise overlap matrix
                      (shape: [H_1, H_2]).
    """
    
    e3_cho_tensor_1, e3_ijk_1, e3_alpha_idxs_1, e3_c_idxs_1 = \
        e3_cho_data_1 if e3_cho_data_1 is not None else \
        e3_cho_overlap_matrix_metadata(l_1)

    e3_cho_tensor_2, e3_ijk_2, e3_alpha_idxs_2, e3_c_idxs_2 = \
        e3_cho_data_2 if e3_cho_data_2 is not None else \
        e3_cho_overlap_matrix_metadata(l_2)

    cs_1 = e3_cs_1[e3_c_idxs_1]
    cs_2 = e3_cs_2[e3_c_idxs_2]

    M = sto_ng_potential_overlap_matrix(
        mu_1, e3_ijk_1,
        l_alphas_1, e3_alpha_idxs_1, cs_1,
        gammas_1, vs_1,
        mu_2, e3_ijk_2,
        l_alphas_2, e3_alpha_idxs_2, cs_2)

    M_X = e3_cho_tensor_1 @ M @ e3_cho_tensor_2.T

    return M_X


def cho_ng_block_overlap_matrix(
        mu_1: torch.Tensor, l_1: torch.IntTensor,
        l_alphas_1: torch.Tensor, 
        l_cs_1: torch.Tensor,
        mu_2: torch.Tensor, l_2: torch.IntTensor,
        l_alphas_2: torch.Tensor, 
        l_cs_2: torch.Tensor,
        cho_data_1: tuple = None,
        cho_data_2: tuple = None) -> torch.Tensor:
    """Evaluates the cubic harmonic GTO pairwise orbital overlap matrix 
    between batches of identically-sized blocks of orbitals 
    with fixed angular momenta l.

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

    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, l_block_cs_1 = \
        _cho_block_data(l_1, l_cs_1)

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, l_block_cs_2 = \
        _cho_block_data(l_2, l_cs_2)

    M = sto_ng_block_overlap_matrix(
        mu_1, l_ijk_1,
        l_alphas_1, l_alpha_idxs_1, l_block_cs_1,
        mu_2, l_ijk_2,
        l_alphas_2, l_alpha_idxs_2, l_block_cs_2)

    M_X = torch.einsum('mi, ...ij, jn -> ...mn',
                       cho_tensor_1,
                       M,
                       cho_tensor_2.mT)

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
        cho_data_2: tuple = None) -> torch.Tensor:
    """Evaluates the cubic harmonic GTO pairwise orbital overlap matrix 
    between blocks of orbitals under the influence of 
    a cartesian potential function centered at the first orbital. 
    The blocks are evaluated for a fixed set of angular momenta l.

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

    cho_tensor_1, l_ijk_1, l_alpha_idxs_1, l_block_cs_1 = \
        _cho_block_data(l_1, l_cs_1, cho_data=cho_data_1)

    cho_tensor_2, l_ijk_2, l_alpha_idxs_2, l_block_cs_2 = \
        _cho_block_data(l_2, l_cs_2, cho_data=cho_data_2)

    V_X = sto_ng_block_potential_overlap_matrix(
        mu_1, l_ijk_1,
        l_alphas_1, l_alpha_idxs_1, l_block_cs_1,
        gammas_1, vs_1,
        mu_2, l_ijk_2,
        l_alphas_2, l_alpha_idxs_2, l_block_cs_2)

    M_X = torch.einsum('mi, ...ij, jn -> ...mn',
                       cho_tensor_1,
                       V_X,
                       cho_tensor_2.mT)

    return M_X


_CHO_OCCUPATION_ORDER = torch.tensor([
    (nl_sum - l, l)
    for nl_sum in range(1, 9)
    for l in reversed(range((nl_sum+1)//2))
])
""" Order of cubic harmonic orbital occupations for 
atoms expanded in a GTO basis"""

_CHO_BASIS_SIZES = torch.tensor([
    2*l + 1
    for l in range(GTO_L_MAX+1)
])
""" Cubic harmonic basis sizes for each
anglar momentum l"""


def _get_valence_shells(Z: int) -> list:
    """ Returns the valence shell angular momenta of an
    element with atomic number Z."""
    if Z <= 2:
        l_values = [0]
    else:
        for block in _CHO_OCCUPATION_ORDER:
            _, l_block = block
            Z -= 2*_CHO_BASIS_SIZES[l_block].item()
            if Z <= 0:
                break

        if l_block < 2:
            l_values = [0, 1]
        elif l_block == 2:
            l_values = [0, 1, 2]
        else:
            l_values = [0, 1, 2, 3]

    return l_values


_ELEMENT_CHO_VALENCE_SHELLS = {
    z: _get_valence_shells(z)
    for z in range(1, 119)
}

_CANONICAL_CHO_BASIS_NAMES = {
    0: [r's'],
    1: [r'p_{y}', r'p_{z}', r'p_{x}'],
    2: [r'd_{xy}', r'd_{yz}', r'd_{z^2}', 
        r'd_{xz}', r'd_{x^2 - y^2}'],
    3: [r'f_{y(3x^2 - y^2)}', r'f_{xyz}', r'f_{yz^2}', 
        r'f_{z^3}', r'f_{xz^2}', r'f_{z(x^2 - y^2)}', 
        r'f_{x(x^2-3y^2)}']
}
""" Map of l (angular momentum) -> names of orbitals 
in cubic harmonic basis"""


def _get_cho_orbital_name(l: int, i: int):
    """ Returns name of orbital with angular momentum l
    and index i (0 <= i < 2l+1)."""
    assert 0 <= l
    assert 0 <= i < 2*l+1

    l = int(l)
    if l in _CANONICAL_CHO_BASIS_NAMES:
        orbital = _CANONICAL_CHO_BASIS_NAMES[l][i]
    else:
        orb_symbol = chr(ord('f') + (l-3))\
            if l <= 23 else f'({l})'
        orbital = orb_symbol + '_{' + str(i) + '}'

    return orbital
