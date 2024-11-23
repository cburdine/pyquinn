import torch

from math import factorial, sqrt, comb, sin, cos
import numpy as np
from itertools import product

_GTO_L_MAX = 6
""" Maximum Angular momentum supported by default"""

def _make_gto_ijk_coeffs(l_max):
    """ Generates a tensor of coefficients corresponding to IJK values up to l=l_max"""
    return torch.tensor([
            factorial(i)/factorial(2*i)
            for i in range(l_max+1)
        ], requires_grad=False)

_GTO_IJK_COEFFS = _make_gto_ijk_coeffs(_GTO_L_MAX)
""" Normalizing constant factors for ijk gaussian-type orbitals up to l=_GTO_L_MAX"""

def _make_gto_ijk_values(l):
    """ Generates a tensor of all IJK values wiht angular momentum l"""

    return torch.tensor([
        [l-(j+k),j,k]
        for k in range(l+1)
        for j in range(l+1-k)
    ], requires_grad=False)

_GTO_IJK_VALUES = {
    l : _make_gto_ijk_values(l)
    for l in range(_GTO_L_MAX+1)
}
""" IJK values for each angular momentum  l up to l = _GTO_L_MAX"""

def _make_gto_hermite_tensor(l_max):
    """ Generates a matrix of Hermite polynomial coefficients up to order n = 2*l+1"""
    n = 2*(l_max + 1)
    H = torch.zeros((n,n), requires_grad=False)
    for j in range(n):
        for p in range(j//2 + 1):
            H[j,j-2*p] = factorial(j) * (-1)**p * 2**(j-2*p) / (factorial(p)*factorial(j-2*p))

    return H

_GTO_HERMITE_TENSOR = _make_gto_hermite_tensor(_GTO_L_MAX)
""" Matrix of Hermite polynomial coefficients up to order n = 2*_GTO_MAX + 1"""

def _make_gto_alpha_tensor(l_max):
    """ Generates a matrix of alpha coefficients for multiplying the product of Gaussian primitives up to l = _GTO_L_MAX"""
    n = l_max+1
    A = torch.zeros((n,n), requires_grad=False)
    for l in range(n):
        norm_coeff = ((2)**(1/2) * 8**l * factorial(l) / factorial(2*l))**(1/2)
        for i in range(l//2 + 1):
            A[l,l-2*i] = norm_coeff * factorial(l)/ ( 2**l * factorial(i) * factorial(l-2*i))
                          
    return A

_GTO_ALPHA_TENSOR = _make_gto_alpha_tensor(_GTO_L_MAX)
""" Matrix of alphas coefficients up to l = _GTO_L_MAX"""

def _make_gto_cho_tensor(l, ijks=None) -> torch.Tensor:
    """Generates a matrix mapping the basis of l2-normalized Gaussian functions with i+j+k=l
       to the basis of orthonormal 2l+1 cubic harmonics with angular momentum l

    Args:
        l (int): Angular momentum of Gaussian basis functions (l = i+j+k)
        ijks (torch.Tensor, optional): Matrix of ordered ijk values (shape: [n,3]). If None,
                                       uses the default ordering of ijk values for the given l. 
                                       Defaults to None.

    Returns:
        torch.Tensor: Matrix of coefficients (shape: [2l+1, len(ijks)])
    """

    if ijks is None:
        ijks = _make_gto_ijk_values(l).detach().numpy()
    ijk_coeffs = _make_gto_ijk_coeffs(l)

    n_ijks = len(ijks)
    n_chos = 2*l+1
    X = torch.zeros((n_chos, n_ijks))
    
    ijk_idxs = { tuple(ijk) : i for i, ijk in enumerate(ijks) }

    ijk_norm_coeffs = {
        tuple(ijk) : (ijk_coeffs[torch.tensor(ijk)].prod(-1) * factorial(2*l) / factorial(l))**(-1/2)
        for ijk in ijks
    }

    # generate unnormalized cubic harmonic matrix:
    for i, m_signed in enumerate(range(-l,l+1)):
        m = abs(m_signed)
        gamma_0 = 1.0 if m == 0 else sqrt(2*(factorial(l-m) / factorial(l+m)))
        for k in range((l-m)//2 + 1):
            N_lmk = (comb(l,k)*comb(2*l - 2*k, l) * factorial(l-2*k)) / factorial(l + (-2*k) + (-m))
            gamma_lmk = gamma_0 * N_lmk * ((-1)**k / 2**l)

            for p in range(m+1):
                if m_signed == 0:
                    sigma_lmk = comb(m,p)
                elif (m_signed < 0):
                    sigma_lmk = comb(m,p)*((-1)**((m-p)//2) - (-1)**(((m-p)+1)//2)) //2 # equals: sin(torch.pi/2 * (m-p))*comb(m,p)
                else:
                    sigma_lmk = comb(m,p)*((-1)**((m-p)//2) + (-1)**(((m-p)+1)//2)) //2 # equals: cos(torch.pi/2 * (m-p))*comb(m,p)
                
                for s in range(k+1):
                    for t in range(s+1):
                        ijk = (p + 2*(k-s),      # i
                               (m-p)+2*(s-t),    # j
                               (l-2*k-m) + 2*t)  # k
                        assert(ijk in ijks)
                        N_ks = comb(k,s) * comb(s,t)
                        norm_coeff = ijk_norm_coeffs[ijk]
                        X[i, ijk_idxs[ijk]] += gamma_lmk*sigma_lmk*norm_coeff * N_ks

    return X

_GTO_CHO_TENSORS = {
    l : _make_gto_cho_tensor(l,
     _GTO_IJK_VALUES[l].detach().numpy())
    for l in range(_GTO_L_MAX+1)
}
""" Dictionary with matrices mapping gaussian bases to cubic harmonics for each l up to l=_GTO_L_MAX."""
                

def gaussian_3d(x: torch.Tensor, mu: torch.Tensor, beta : torch.Tensor) -> torch.Tensor:
    """Evaluates a 3D unnormalized Gaussian primitive function



    Args:
        x (torch.Tensor): Point x in space (shape: [...,3])
        mu (torch.Tensor): Gaussian center (shape: [...,3])
        beta (torch.Tensor): Gaussian band widths (shape: [...]) 

    Returns:
        torch.Tensor: Value of Gaussians
    """
    return torch.exp(-beta*torch.sum((x-mu)**2, dim=-1, keepdim=True))

def l1_gaussian_3d(x, mu, gamma):
    return (gamma/torch.pi)**(3/2) * torch.exp(-gamma*torch.sum((x-mu)**2, dim=-1, keepdim=True))

def l2_gaussian_3d(x, mu, alpha):
    return (2*alpha/torch.pi)**(3/4) * torch.exp(-alpha*torch.sum((x-mu)**2, dim=-1, keepdim=True))

def l2_gaussian_3d_product(mu_1, alpha_1, c_1, mu_2, alpha_2, c_2):

    gamma_prod = alpha_1 + alpha_2
    gamma_frac = alpha_1*alpha_2 / gamma_prod
    gamma_l2_frac = 2*torch.sqrt(alpha_1*alpha_2) / gamma_prod
    c_prod = c_1*c_2*(gamma_l2_frac)**(3/2) \
        * gaussian_3d(mu_1, mu_2, gamma_frac)
    mu_prod = (alpha_1 * mu_1 + alpha_2 * mu_2)/gamma_prod

    return torch.cat([mu_prod, gamma_prod, c_prod], dim=-1)

def gto_3d(x, mu, ijk, alpha):

    dx = x - mu
    psi = torch.exp(-alpha * ((dx)**2).sum(-1, keepdim=True)) \
        * (dx**(ijk)).prod(-1, keepdim=True) \
        * (2*alpha/torch.pi)**(3/4) \
        * (8*alpha)**(ijk.sum(-1, keepdim=True)/2) \
        * torch.sqrt((_GTO_IJK_COEFFS[ijk]).prod(-1, keepdim=True))

    return psi

def sto_ng_3d(x, mu, ijk, alphas, cs):
    gto_3d_component = torch.vmap(lambda a,c : c*gto_3d(x, mu, ijk, a), in_dims=-1, out_dims=-1)
    psi = torch.sum(gto_3d_component(alphas, cs), axis=-1).squeeze(-1)

    return psi

def sto_ng_prmitive_overlap_matrix(mu_1, alphas_1, mu_2, alphas_2):

    _alphas_1 = alphas_1.unsqueeze(-1)
    _alphas_2 = alphas_2.unsqueeze(-2)
    sqrt_etas = torch.sqrt(_alphas_1 * _alphas_2 / (_alphas_1 + _alphas_2))

    dx = (mu_1 - mu_2)
    dx = dx.view(*dx.shape, 1, 1)

    hermite_powers = torch.arange(_GTO_HERMITE_TENSOR.shape[-1]).view(*([1]*sqrt_etas.ndim),-1)

    sqrt_eta_powers = sqrt_etas.unsqueeze(-1) ** hermite_powers
    r_powers = dx.unsqueeze(-1) ** hermite_powers

    hermite_coeffs = sqrt_eta_powers * torch.einsum(
        'jp, ...ip -> ...ij',
        _GTO_HERMITE_TENSOR,
        (sqrt_eta_powers * r_powers))

    alphas_1_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1])
    alphas_2_powers = torch.arange(_GTO_ALPHA_TENSOR.shape[-1])
    j_powers = alphas_1_powers.view(-1,  1) + alphas_2_powers.view( 1, -1)

    sqrt_alphas_1 = torch.sqrt(alphas_1).unsqueeze(-1)
    sqrt_alphas_2 = torch.sqrt(alphas_2).unsqueeze(-1)

    eta_gaussians = torch.exp(-(sqrt_etas * dx)**2) * sqrt_etas / ((_alphas_1 * _alphas_2)**(1/4))

    A_1 = (sqrt_alphas_1 ** (-alphas_1_powers.unsqueeze(0))) * ((-1) ** alphas_1_powers.unsqueeze(0))
    A_2 = (sqrt_alphas_2 ** (-alphas_2_powers.unsqueeze(0)))
    J_12 = hermite_coeffs[...,j_powers]
    
    T = torch.einsum('ij, mj, ...dmnjk, ...dmn, nk, kl -> ...mndil',
                        _GTO_ALPHA_TENSOR,
                        A_1, J_12, eta_gaussians, A_2,
                        _GTO_ALPHA_TENSOR.T)
    return T

def sto_ng_3d_overlap_matrix(mu_1, ijk_1, alphas_1, cs_1, mu_2, ijk_2, alphas_2, cs_2):

    T = sto_ng_prmitive_overlap_matrix(mu_1, alphas_1, mu_2, alphas_2)
    _ijk_1 = ijk_1.transpose(-2,-1).view(3,1,-1).expand(3,ijk_2.shape[0], ijk_1.shape[0])
    _ijk_2 = ijk_2.transpose(-2,-1).view(3,-1,1).expand(3,ijk_2.shape[0], ijk_1.shape[0])
    _dim = torch.arange(3).view(3,1,1).expand(*_ijk_1.shape)

    T_ijk = T[..., _dim, _ijk_1, _ijk_2].prod(-3)
    
    M = torch.einsum('m, ...mnij, n -> ...ij', cs_1, T_ijk, cs_2)
    return M

def sto_ng_3d_norm(alphas, cs, ijks):
    mu = torch.zeros((1,3))
    M = sto_ng_3d_overlap_matrix(mu, ijks, alphas, cs, mu, ijks, alphas, cs)

    return torch.sqrt(M.sum())

def cho_ng_3d(x, mu, l, m, alphas, cs):
    assert(abs(m) < l)

    m_idx = l + m
    X_lm = _GTO_CHO_TENSORS[l][m_idx]
    X_lm_idx, = torch.nonzero(X_lm, as_tuple=True)
    X_lm_ijks = _GTO_IJK_VALUES[l][X_lm_idx]
    X_lm_coeffs = X_lm[X_lm_idx]

    sto_3d_component = torch.vmap(
        lambda h, ijk, : h*sto_ng_3d(x, mu, ijk, alphas, cs),
        in_dims=-1, out_dims=-1)

    psi = torch.sum(sto_3d_component(X_lm_coeffs, X_lm_ijks), axis=-1).squeeze(-1)

    return psi


def cho_ng_3d_overlap_matrix(mu_1, l_1, alphas_1, cs_1, mu_2, l_2, alphas_2, cs_2):
    ijk_1 = _GTO_IJK_VALUES[l_1]
    ijk_2 = _GTO_IJK_VALUES[l_2]
    
    M = sto_ng_3d_overlap_matrix(mu_1, ijk_1, alphas_1, cs_1,
                                 mu_2, ijk_2, alphas_2, cs_2)

    M_X = torch.einsum('mi, ...ij, jn -> ...mn',
                         _GTO_CHO_TENSORS[l_1],
                         M,
                         _GTO_CHO_TENSORS[l_2].mT)

    return M_X

def cho_ng_3d_norm(x, mu, l, m, alphas, cs):
    mu = torch.zeros((1,3))
    M_X = cho_ng_3d_overlap_matrix(mu, l, alphas, cs, mu, l, alphas, cs)

    return M_X.sum()

def cho_ng_3d_site_overlap_matrix(mu_1, l_blocks_1, l_block_alphas_1, l_block_cs_1,
                                  mu_2, l_blocks_2, l_block_alphas_2, l_block_cs_2):
    
    cho_tensor_1 = torch.block_diag(_GTO_CHO_TENSORS[l] for l in l_blocks_1)
    cho_tensor_2 = torch.block_diag(_GTO_CHO_TENSORS[l] for l in l_blocks_2)
    ijk_1 = torch.hstack(_GTO_IJK_VALUES[l] for l in l_blocks_1)
    ijk_1 = torch.hstack(_GTO_IJK_VALUES[l] for l in l_blocks_2)


    M = sto_ng_3d_overlap_matrix(mu_1, ijk_1, alphas_1, cs_1,
                                 mu_2, ijk_2, alphas_2, cs_2)


    M = sto_ng_3d_overlap_matrix(mu_1, ijk_1, alphas_1)

    M_X = torch.einsum('mi, ...ij, jn -> ...mn',
                         _GTO_CHO_TENSORS[l_1],
                         M,
                         _GTO_CHO_TENSORS[l_2].mT)
    

