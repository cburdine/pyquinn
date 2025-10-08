from pyquinn.types.operator import Operator, BackendOperator
import torch

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