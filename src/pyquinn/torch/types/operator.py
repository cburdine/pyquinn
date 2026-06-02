import numpy as np
import torch

from pyquinn.types.operator import BackendOperator, Operator


def _batch_shape(onsite_values: torch.Tensor, coupling_values: torch.Tensor):

    onsite_batch = tuple(onsite_values.shape[:-1]) if onsite_values.ndim > 1 else ()
    coupling_batch = tuple(coupling_values.shape[:-1]) if coupling_values.ndim > 1 else ()

    if onsite_batch and coupling_batch and onsite_batch != coupling_batch:
        raise ValueError(
            "Leading batch dimensions of `onsite_values` and `coupling_values` must match "
            f"(got: {onsite_batch} != {coupling_batch})"
        )

    return onsite_batch if onsite_batch else coupling_batch


def _ensure_batched_entries(values: torch.Tensor, batch_shape: tuple[int, ...], entry_count: int):

    if values.shape[-1] != entry_count:
        raise ValueError(
            "Shape of last dimension must match entry count " f"(got: {values.shape[-1]} != {entry_count})"
        )

    if not batch_shape:
        return values

    if values.ndim == 1:
        return values.view((1,) * len(batch_shape) + (entry_count,)).expand(batch_shape + (entry_count,))

    return values


def build_nonperiodic_matrix(
    onsite_values: torch.Tensor,
    couplings: torch.IntTensor,
    coupling_values: torch.Tensor,
):

    N = onsite_values.shape[-1]
    E = couplings.shape[0]
    batch_shape = _batch_shape(onsite_values, coupling_values)

    onsite_values = _ensure_batched_entries(onsite_values, batch_shape, N)
    coupling_values = _ensure_batched_entries(coupling_values, batch_shape, E)

    if batch_shape:
        batch_size = int(np.prod(batch_shape))
        onsite_flat = onsite_values.reshape(batch_size, N)
        coupling_flat = coupling_values.reshape(batch_size, E)
    else:
        batch_size = 1
        onsite_flat = onsite_values.reshape(1, N)
        coupling_flat = coupling_values.reshape(1, E)

    H_diag_idxs = (N + 1) * torch.arange(N, device=onsite_values.device, dtype=torch.int64)
    H_ij_idxs = (N * couplings[:, 0] + couplings[:, 1]).to(dtype=torch.int64)

    H = torch.zeros(
        (batch_size, N * N),
        dtype=coupling_values.dtype,
        device=onsite_values.device,
    )

    H[:, H_diag_idxs] += onsite_flat
    H.scatter_add_(1, H_ij_idxs.unsqueeze(0).expand(batch_size, -1), coupling_flat)

    H = H.reshape(batch_shape + (N, N))

    return H


def build_matrix(
    onsite_values: torch.Tensor,
    couplings: torch.IntTensor,
    coupling_values: torch.Tensor,
    site_indices: torch.IntTensor,
    images: torch.IntTensor,
    k_pts: torch.Tensor,
    frac_positions: torch.IntTensor,
    sparse: bool = False,
):

    N = onsite_values.shape[-1]
    E = couplings.shape[0]
    N_k = k_pts.shape[0]
    batch_shape = _batch_shape(onsite_values, coupling_values)

    onsite_values = _ensure_batched_entries(onsite_values, batch_shape, N)
    coupling_values = _ensure_batched_entries(coupling_values, batch_shape, E)

    if batch_shape:
        batch_size = int(np.prod(batch_shape))
        onsite_flat = onsite_values.reshape(batch_size, N)
        coupling_flat = coupling_values.reshape(batch_size, E)
    else:
        batch_size = 1
        onsite_flat = onsite_values.reshape(1, N)
        coupling_flat = coupling_values.reshape(1, E)

    frac_dists = frac_positions[site_indices[couplings[:, 1]]] - frac_positions[site_indices[couplings[:, 0]]]

    H_ij_phases = torch.exp(2.0j * torch.pi * ((frac_dists + images) @ k_pts.mT)).T

    H_diag_idxs = (N + 1) * torch.arange(N, device=onsite_values.device, dtype=torch.int64)
    H_ij_idxs = (N * couplings[:, 0] + couplings[:, 1]).to(dtype=torch.int64)

    if sparse:
        H_ij_values = coupling_flat.unsqueeze(1) * H_ij_phases.unsqueeze(0)
        H_values_total = torch.cat(
            (
                onsite_flat.unsqueeze(1).expand(-1, N_k, -1),
                H_ij_values,
            ),
            dim=-1,
        )

        H_ij_total = torch.cat((H_diag_idxs, H_ij_idxs), dim=0)
        row_idxs = torch.div(H_ij_total, N, rounding_mode="floor")
        col_idxs = H_ij_total % N

        H_values_sparse = H_values_total.permute(2, 1, 0).reshape(N + E, N_k, *batch_shape)

        H_indices = torch.stack((row_idxs, col_idxs), dim=0)

        return torch.sparse_coo_tensor(
            indices=H_indices,
            values=H_values_sparse,
            size=(N, N, N_k) + batch_shape,
            dtype=H_values_sparse.dtype,
            device=H_values_sparse.device,
        ).coalesce()

    H_ij_values = coupling_flat.unsqueeze(1) * H_ij_phases.unsqueeze(0)

    H = torch.zeros(
        (batch_size, N_k, N * N),
        dtype=H_ij_values.dtype,
        device=onsite_values.device,
    )

    H[:, :, H_diag_idxs] += onsite_flat.unsqueeze(1)
    H.scatter_add_(2, H_ij_idxs.view(1, 1, -1).expand(batch_size, N_k, -1), H_ij_values)

    H = H.reshape(batch_shape + (N_k, N, N))
    H = torch.movedim(H, -3, 0)

    return H


class TorchOperator(BackendOperator):

    @staticmethod
    def compile(
        operator: Operator,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.complex64,
        idx_dtype: torch.dtype = torch.int64,
    ):

        onsite_values_np = np.asarray(operator.onsite_values)
        if onsite_values_np.ndim > 1:
            onsite_values_np = np.moveaxis(onsite_values_np, 0, -1)

        coupling_values_np = np.asarray(operator.coupling_values)
        if coupling_values_np.ndim > 1:
            coupling_values_np = np.moveaxis(coupling_values_np, 0, -1)

        onsite_values = torch.tensor(onsite_values_np, device=device, dtype=dtype)
        couplings = torch.tensor(operator.couplings, device=device, dtype=torch.int32)

        coupling_values = torch.tensor(coupling_values_np, device=device, dtype=dtype)
        site_indices = torch.tensor(operator.site_indices, device=device, dtype=idx_dtype)
        orbital_indices = torch.tensor(operator.orbital_indices, device=device, dtype=idx_dtype)

        if operator.images is None:
            images = torch.zeros((couplings.shape[0], 3), device=device, dtype=idx_dtype)
        else:
            images = torch.tensor(operator.images, device=device, dtype=idx_dtype)

        return TorchOperator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
            images=images,
        )

    def __init__(
        self,
        onsite_values: torch.Tensor,
        couplings: torch.IntTensor,
        coupling_values: torch.Tensor,
        site_indices: torch.IntTensor,
        orbital_indices: torch.IntTensor,
        images: torch.IntTensor,
    ):

        super().__init__(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
            images=images,
        )

    def matrix(
        self,
        k_pts: torch.Tensor,
        frac_positions: torch.Tensor,
        sparse: bool = False,
    ):
        return build_matrix(
            self.onsite_values,
            self.couplings,
            self.coupling_values,
            self.site_indices,
            self.images,
            k_pts,
            frac_positions,
            sparse=sparse,
        )

    def detach(self):
        return TorchOperator(
            onsite_values=self.onsite_values.detach(),
            couplings=self.couplings.detach(),
            coupling_values=self.coupling_values.detach(),
            site_indices=self.site_indices.detach(),
            orbital_indices=self.orbital_indices.detach(),
            images=self.images.detach(),
        )

    def clone(self):
        return TorchOperator(
            onsite_values=self.onsite_values.clone(),
            couplings=self.couplings.clone(),
            coupling_values=self.coupling_values.clone(),
            site_indices=self.site_indices.clone(),
            orbital_indices=self.orbital_indices.clone(),
            images=self.images.clone(),
        )

    def to(self, *args, **kwargs):

        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            return self

        return TorchOperator(
            onsite_values=self.onsite_values.to(*args, **kwargs),
            couplings=self.couplings.to(*args, **kwargs),
            coupling_values=self.coupling_values.to(*args, **kwargs),
            site_indices=self.site_indices.to(*args, **kwargs),
            orbital_indices=self.orbital_indices.to(*args, **kwargs),
            images=self.images.to(*args, **kwargs),
        )

    def extract(self) -> Operator:
        onsite_values = self.onsite_values.detach().cpu().numpy()
        if onsite_values.ndim > 1:
            onsite_values = np.moveaxis(onsite_values, -1, 0)

        coupling_values = self.coupling_values.detach().cpu().numpy()
        if coupling_values.ndim > 1:
            coupling_values = np.moveaxis(coupling_values, -1, 0)

        return Operator(
            onsite_values=onsite_values,
            couplings=self.couplings.detach().cpu().numpy(),
            coupling_values=coupling_values,
            site_indices=self.site_indices.detach().cpu().numpy(),
            orbital_indices=self.orbital_indices.detach().cpu().numpy(),
            images=self.images.detach().cpu().numpy(),
        )

    @staticmethod
    def load(filename: str, device: torch.device | None = None):
        operator = torch.load(filename, weights_only=True, map_location=device)

        if not isinstance(operator, TorchOperator):
            raise OSError(f"Expected an instance of TorchOperator; " f"got: {type(operator)}")

        return operator

    def save(self, filename: str):
        torch.save(self, filename)


torch.serialization.add_safe_globals([TorchOperator])
