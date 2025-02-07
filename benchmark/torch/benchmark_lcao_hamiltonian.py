
from itertools import product
import torch
from torch.utils import benchmark

from pyquinn.torch.lcao.lcao_hamiltonian import LCAOModel, TightBindingModel


def _make_lcao_model_hamiltonian_args(
        element: int, atoms_per_dim: int, n=2, sparse: bool = False):

    dx = torch.linspace(0, atoms_per_dim-1, atoms_per_dim)
    x, y, z = torch.meshgrid((dx, dx, dx), indexing='ij')
    positions = torch.vstack((x.flatten(), y.flatten(), z.flatten())).T

    atomic_numbers = torch.full((positions.shape[0],), element)

    d_idx = torch.cat([torch.arange(-n, 0), torch.arange(1, n+1)], dim=0)
    d_i, d_j, d_k = torch.meshgrid((d_idx, d_idx, d_idx), indexing='ij')
    neighbor_d_idx = (d_i.flatten()*atoms_per_dim**2 + 
                      d_j.flatten()*atoms_per_dim + 
                      d_k.flatten()) % (atoms_per_dim**3)

    neighbors = (torch.arange(positions.shape[0]).view(-1, 1)
        + neighbor_d_idx.view(1, -1)) % (atoms_per_dim**3)

    return {
        'atomic_numbers': atomic_numbers, 
        'positions': positions, 
        'neighbors': neighbors,
        'sparse': sparse
    }


python_lcao_models = {
    (n, m): LCAOModel(basis_size=n, potential_basis_size=m)
    for n, m in product([3, 5], [3, 5])
}

compiled_lcao_models = {
    (n, m): torch.compile(LCAOModel(basis_size=n, potential_basis_size=m))
    for n, m in product([3, 5], [3, 5])
}


def benchmark_lcao_model_hamiltonian(results):

    num_threads = torch.get_num_threads()
    print(f'Benchmarking LCAOModel.hamiltonian_matrix() on {num_threads} threads...')

    def _run_benchmark(n, m, size, results, threads=None):

        t0 = benchmark.Timer(
            stmt='python_lcao_models[(n,m)].hamiltonian_matrix(**kwargs)',
            setup='from __main__ import python_lcao_models', # noqa
            globals={
                'n': n, 'm': m,
                'kwargs': 
                    _make_lcao_model_hamiltonian_args(
                        element=29, atoms_per_dim=size, n=1)}, # noqa
            num_threads=num_threads,
            label='LCAOModel.hamiltonian_matrix()',
            sub_label=f'Cu [{size},{size},{size}] (n={n},m={m})',
            description='python',
        ).blocked_autorange(min_run_time=0.2)
        results.append(t0)

        t1 = benchmark.Timer(
            stmt='compiled_lcao_models[(n,m)].hamiltonian_matrix(**kwargs)',
            setup='from __main__ import compiled_lcao_models', # noqa
            globals={
                'n': n, 'm': m,
                'kwargs': 
                    _make_lcao_model_hamiltonian_args(
                        element=29, atoms_per_dim=size, n=1)}, # noqa
            num_threads=num_threads,
            label='LCAOModel.hamiltonian_matrix()',
            sub_label=f'Cu [{size},{size},{size}] (n={n},m={m})',
            description='torch.compile',
        ).blocked_autorange(min_run_time=0.2)
        results.append(t1)

    for n, m in python_lcao_models.keys():
        for size in [3, 5, 7]:
            _run_benchmark(n, m, size, results)


if __name__ == '__main__':

    results = []
    benchmark_lcao_model_hamiltonian(results)

    compare = benchmark.Compare(results)
    compare.print()