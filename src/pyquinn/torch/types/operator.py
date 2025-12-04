from pyquinn.types.operator import Operator, BackendOperator
import torch


def build_nonperiodic_matrix(
        onsite_values: torch.Tensor,
        couplings: torch.IntTensor,
        coupling_values: torch.Tensor):

    N = onsite_values.shape[0]
    
    # make diagonal and off-diagonal indices
    H_diag_idxs = (N+1)*torch.arange(N, device=onsite_values.device)
    H_ij_idxs = (N*couplings[:, 0] + couplings[:, 1])

    # build dense Hamiltonian
    H = torch.zeros(
        N*N, 
        dtype=coupling_values.dtype,
        device=onsite_values.device)

    H[H_diag_idxs] += onsite_values
    H.put_(H_ij_idxs, coupling_values, accumulate=True)

    # restore original shape
    H = H.reshape((N, N))

    return H


def build_matrix(
        onsite_values: torch.Tensor,
        couplings: torch.IntTensor,
        coupling_values: torch.Tensor,
        site_indices: torch.IntTensor,
        images: torch.IntTensor,
        k_pts: torch.IntTensor,
        frac_positions: torch.IntTensor):
    
    N = onsite_values.shape[0]
    N_k = k_pts.shape[0]

    # make diagonal components
    H_diag_idxs = (N+1)*torch.arange(N, device=onsite_values.device)

    # make off-diagonal components
    frac_dists = frac_positions[site_indices[couplings[:, 1]]] - \
        frac_positions[site_indices[couplings[:, 0]]]

    H_ij_phases = torch.exp(2.j*torch.pi*((frac_dists + images) @ k_pts.mT))
    H_ij_values = coupling_values*H_ij_phases.T
    H_ij_idxs = (N*couplings[:, 0] + couplings[:, 1])\
        .unsqueeze(0).expand(H_ij_values.shape[0], -1)

    # build dense hamiltonian
    H = torch.zeros((N_k, N*N),
                    dtype=H_ij_values.dtype,
                    device=onsite_values.device)
    
    H[:, H_diag_idxs] += onsite_values
    H.scatter_add_(1, H_ij_idxs, H_ij_values)

    # restore original shape
    H = H.reshape((N_k, N, N))

    return H

class TorchOperator(BackendOperator):
    
    @staticmethod
    def compile(
            operator: Operator,
            device: torch.device = 'cpu',
            dtype: torch.dtype = torch.complex64,
            idx_dtype: torch.dtype = torch.int64):
        
        onsite_values = torch.tensor(
            operator.onsite_values, device=device, dtype=dtype)
        couplings = torch.tensor(
            operator.couplings, dtype=torch.int32)

        coupling_values = torch.tensor(
            operator.coupling_values, device=device, dtype=dtype)
        site_indices = torch.tensor(
            operator.site_indices, device=device, dtype=idx_dtype)
        orbital_indices = torch.tensor(
            operator.orbital_indices, device=device, dtype=idx_dtype)
        
        if operator.images is None:
            images = torch.zeros(
                (couplings.shape[0], 3), device=device, dtype=idx_dtype)
        else:
            images = torch.tensor(
                operator.images, device=device, dtype=idx_dtype)

        return TorchOperator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
            images=images)

    def __init__(
            self,
            onsite_values: torch.Tensor,
            couplings: torch.IntTensor,
            coupling_values: torch.Tensor,
            site_indices: torch.IntTensor, 
            orbital_indices: torch.IntTensor,
            images: torch.IntTensor):

        super().__init__(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
            images=images)

    def matrix(
            self,
            k_pts: torch.Tensor,
            frac_positions: torch.Tensor):

        return build_matrix(
            self.onsite_values,
            self.couplings,
            self.coupling_values,
            self.site_indices,
            self.images,
            k_pts,
            frac_positions
        )

    def detach(self):
        return TorchOperator(
            onsite_values=self.onsite_values.detach(),
            couplings=self.couplings.detach(),
            coupling_values=self.coupling_values.detach(),
            site_indices=self.site_indices.detach(),
            orbital_indices=self.orbital_indices.detach(),
            images=self.images.detach())

    def clone(self):
        return TorchOperator(
            onsite_values=self.onsite_values.clone(),
            couplings=self.couplings.clone(),
            coupling_values=self.coupling_values.clone(),
            site_indices=self.site_indices.clone(),
            orbital_indices=self.orbital_indices.clone(),
            images=self.images.clone())

    def to(self, *args, **kwargs):

        # short circuit to() function
        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            return self

        return TorchOperator(
            onsite_values=self.onsite_values.to(*args, **kwargs),
            couplings=self.couplings.to(*args, **kwargs),
            coupling_values=self.coupling_values.to(*args, **kwargs),
            site_indices=self.site_indices.to(*args, **kwargs),
            orbital_indices=self.orbital_indices.to(*args, **kwargs),
            images=self.images.to(*args, **kwargs))

    def extract(self) -> Operator:
        return Operator(
            onsite_values=self.onsite_values.detach().numpy(),
            couplings=self.couplings.detach().numpy(),
            coupling_values=self.coupling_values.detach().numpy(),
            site_indices=self.site_indices.detach().numpy(),
            orbital_indices=self.orbital_indices.detach().numpy(),
            images=self.images.detach().numpy())

    @staticmethod
    def load(filename: str, device: torch.device | None = None):
        operator = torch.load(filename, weights_only=True, map_location=device)

        if not isinstance(operator, TorchOperator):
            raise IOError(
                f'Expected an instance of TorchOperator; '
                f'got: {type(operator)}')

        return operator

    def save(self, filename: str):
        torch.save(self, filename)

        
torch.serialization.add_safe_globals([TorchOperator])