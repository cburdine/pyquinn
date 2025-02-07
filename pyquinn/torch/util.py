import torch
from torch.nn.modules import activation

from typing import Callable

Activation = Callable[..., torch.nn.Module]


def get_activation_fn(act: str) -> Activation:
    """Gets a Pytorch activation function by name

    Args:
        act (str): The name of the activation function (in lower case)

    Raises:
        ValueError: _description_

    Returns:
        torch.nn.Module: _description_
    """

    # get list from activation submodule as lower-case
    activations_lc = [str(a).lower() for a in activation.__all__]

    if (act := str(act).replace('_', '').lower()) in activations_lc:

        # match actual name from lower-case list, return function/factory
        idx = activations_lc.index(act)
        act_name = activation.__all__[idx]
        act_func = getattr(activation, act_name)
        return act_func
    else:
        raise ValueError(
            f"Cannot load activation function for string <{act}>.")


def sparse_csr_block_diag(*tensors):

    if not tensors:
        raise ValueError("Input list of sparse tensors cannot be empty.")

    # Validate input tensors
    for t in tensors:
        if t.ndim != 2:
            raise ValueError('All input tensors must have ndim == 2.')

    # Initialize indices and values for the block diagonal tensor
    crow_indices = []
    col_indices = []
    values = []

    block_nnz_idx = 0
    block_col_idx = 0
    block_row_idx = 0

    # combine nonzero elements of block tensors:
    for tensor in tensors:

        sparse_tensor = tensor.to_sparse_csr()

        values.append(sparse_tensor.values())
        crow_indices.append(sparse_tensor.crow_indices()[:-1] + block_nnz_idx)
        col_indices.append(sparse_tensor.col_indices() + block_col_idx)

        block_nnz_idx += sparse_tensor._nnz()
        block_row_idx += tensor.shape[-2]
        block_col_idx += tensor.shape[-1]

    # add last nnz index:
    crow_indices.append(torch.tensor([block_nnz_idx]))

    values = torch.cat(values)
    crow_indices = torch.cat(crow_indices)
    col_indices = torch.cat(col_indices)

    return torch.sparse_csr_tensor(
        crow_indices=crow_indices, 
        col_indices=col_indices, 
        values=values, size=(block_row_idx, block_col_idx))


def sparse_coo_block_diag(*tensors):

    if not tensors:
        raise ValueError("Input list of sparse tensors cannot be empty.")

    # Validate input tensors
    for t in tensors:
        if t.ndim != 2:
            raise ValueError('All input tensors must have ndim == 2.')

    # Initialize indices and values for the block diagonal tensor
    indices = []
    values = []

    block_idx = torch.zeros((2, 1), dtype=torch.int64)
    # combine nonzero elements of block tensors:
    for tensor in tensors:

        sparse_tensor = tensor.to_sparse_coo()

        values.append(sparse_tensor.values())
        indices.append(sparse_tensor.indices() + block_idx)

        block_idx[0] += tensor.shape[0]
        block_idx[1] += tensor.shape[1]

    values = torch.cat(values, dim=-1)
    indices = torch.cat(indices, dim=-1)

    return torch.sparse_coo_tensor(
        indices=indices, 
        values=values, 
        size=(block_idx[0].item(), 
              block_idx[1].item()))
