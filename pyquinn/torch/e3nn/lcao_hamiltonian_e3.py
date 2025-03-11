from pyquinn.torch.lcao.lcao_hamiltonian import TBModel
from pyquinn.torch.e3nn.atomwise_e3 import AtomwiseE3Conv

import e3nn.o3 as o3
import torch

def _get_irrep_transform_matrix(irreps: o3.Irreps):

    change_of_coord = torch.tensor([
    # this specifies the change of basis yzx -> xyz
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
    ])

    D = irreps.D_from_matrix(change_of_coord)

    return D


def _make_atomiwise_e3_nn(
        nn: AtomwiseE3Conv | None = None, 
        out_irreps: o3.Irreps | None = None,
        name: str = 'e3_nn',
        z_max: int = 118,
        nn_kwargs: dict | None = None):

    nn_kwargs = {} if nn_kwargs is None else nn_kwargs

    # if AtomwiseConv instance given, check that input/output features are correct:
    if isinstance(nn, AtomwiseConv):
        if nn.in_features != in_features:
            raise ValueError(
                f'Error in creating AtomwiseConv model. '
                f'Expected {name}.in_features == {in_features} (got {nn.in_features}).')

        if nn.out_features != out_features:
            raise ValueError(
                f'Error in creating AtomwiseConv model. '
                f'Expected {name}.out_features == {out_features} (got {nn.out_features}).'
            )


def _make_shell_irrep_mapping(orbital_shell_idxs):
    pass
        
    
class VariationalE3TBModel(TBModel):

    def __init__(self,
                 base_model: TBModel | None = None,
                 freeze: bool = False,
                 variational_nn: AtomwiseE3Conv | None = None,
                 onsite_nn: AtomwiseE3Conv | None = None,
                 onsite_weight: float = 1.0,
                 variational_weight: float = 1.0,
                 basis_weight: float = 1.0,
                 variational_nn_kwargs: dict | None = None,
                 onsite_nn_kwargs: dict | None = None,
                 **base_model_kwargs):

        if base_model is not None:
            super().__init__(
                eigensolver=base_model.eigensolver,
                spin=base_model.spin,
                basis=base_model.basis,
                basis_size=base_model.basis_size,
                potential_basis=base_model.potential_basis,
                potential_basis_size=base_model.potential_basis_size,
                energy_scale=base_model.energy_scale,
                orbital_shells=base_model.get_orbital_shells_dict(),
                **base_model_kwargs
            )
            self._load_base_model_weights(base_model=base_model, freeze=freeze)
        else:
            super().__init__(**base_model_kwargs)

        n_shells = self.orbital_shell_idxs.shape[-1]
        shell_ls, shell_l_idxs, shell_l_counts = torch.unique(
            self.orbital_shell_idxs, -1)

        
        in_features = (n_shells * self.basis_size) + \
            self.potential_basis_size
        
        variational_out_features = (
            (n_shells * self.basis_size) +
            (n_shells * self.logscale_orbital_alphas.size(-1)) +
            self.potential_basis_size + 
            self.logscale_potential_gammas.size(-1))
        
        onsite_out_features = n_shells
        
        self.onsite_weight = onsite_weight
        self.variational_weight = variational_weight
        self.basis_weight = basis_weight

        if onsite_nn_kwargs is None:
            onsite_nn_kwargs = {} 
        if variational_nn_kwargs is None:
            variational_nn_kwargs = {}

        self.onsite_nn = _make_atomiwise_e3_nn(
            nn=onsite_nn,
            out_features=onsite_out_features,
            in_features=None,
            name='onsite_nn',
            nn_kwargs=onsite_nn_kwargs,
        )

        self.variational_nn = _make_atomiwise_e3_nn(
            nn=variational_nn,
            out_features=variational_out_features,
            in_features=None,
            name='variational_nn',
            nn_kwargs=variational_nn_kwargs,
        )