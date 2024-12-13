import torch


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
