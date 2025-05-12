import torch
from torch.utils import benchmark


from pyquinn.torch.lcao.gto import (
    cho_overlap_matrix_metadata,
    cho_ng_overlap_matrix,
    cho_ng_potential_overlap_matrix,
)


def _make_gto_overlap_args(idx_size, n_size=5, sparse=False):
    _ls = torch.tensor([3]*idx_size, dtype=torch.int)
    cho_data_1 = cho_overlap_matrix_metadata(_ls, sparse=sparse)
    cho_data_2 = cho_overlap_matrix_metadata(_ls, sparse=sparse)
    return {
        'mu_1': torch.randn((idx_size, 3)),
        'l_1': torch.tensor([3]*idx_size, dtype=torch.int),
        'l_alphas_1': torch.abs(torch.randn((idx_size, n_size))),
        'l_cs_1': torch.randn((idx_size, n_size)),
        'mu_2': torch.randn((idx_size, 3)),
        'l_2': torch.tensor([3]*idx_size, dtype=torch.int),
        'l_alphas_2': torch.abs(torch.randn((idx_size, n_size))),
        'l_cs_2': torch.randn((idx_size, n_size)),
        'cho_data_1': cho_data_1,
        'cho_data_2': cho_data_2
    }


def _make_gto_potential_overlap_args(
        idx_size, n_size=5, m_size=5, sparse=False):

    _ls = torch.tensor([3]*idx_size, dtype=torch.int)
    cho_data_1 = cho_overlap_matrix_metadata(_ls, sparse=sparse)
    cho_data_2 = cho_overlap_matrix_metadata(_ls, sparse=sparse)
    return {
        'mu_1': torch.randn((idx_size, 3)),
        'l_1': torch.tensor([3]*idx_size, dtype=torch.int),
        'l_alphas_1': torch.abs(torch.randn((idx_size, n_size))),
        'l_cs_1': torch.randn((idx_size, n_size)),
        'gammas_1': torch.abs(torch.randn((m_size))),
        'vs_1': torch.randn((m_size)),
        'mu_2': torch.randn((idx_size, 3)),
        'l_2': torch.tensor([3]*idx_size, dtype=torch.int),
        'l_alphas_2': torch.abs(torch.randn((idx_size, n_size))),
        'l_cs_2': torch.randn((idx_size, n_size)),
        'cho_data_1': cho_data_1,
        'cho_data_2': cho_data_2
    }


python_cho_ng_overlap_matrix = cho_ng_overlap_matrix
compiled_cho_ng_overlap_matrix = \
    torch.compile(cho_ng_overlap_matrix, dynamic=True)

python_cho_ng_potential_overlap_matrix = cho_ng_potential_overlap_matrix
compiled_cho_ng_potential_overlap_matrix = \
    torch.compile(cho_ng_potential_overlap_matrix, dynamic=True)


def benchmark_cho_overlap(results):

    num_threads = torch.get_num_threads()
    print(f'Benchmarking cho_overlap on {num_threads} threads...')

    def _run_benchmark(n, size, results, threads=None):
        num_threads = torch.get_num_threads() if threads is None else threads

        t0 = benchmark.Timer(
            stmt='python_cho_ng_overlap_matrix(**kwargs)',
            setup='from __main__ import python_cho_ng_overlap_matrix',
            globals={
                'kwargs': _make_gto_overlap_args(size, n, sparse=False)},
            num_threads=num_threads,
            label='CHO Overlap',
            sub_label=f'[{size},{size}] (n={n})',
            description='python',
        ).blocked_autorange(min_run_time=0.2)
        results.append(t0)

        t1 = benchmark.Timer(
            stmt='compiled_cho_ng_overlap_matrix(**kwargs)',
            setup='from __main__ import compiled_cho_ng_overlap_matrix',
            globals={
                'kwargs': _make_gto_overlap_args(size, n, sparse=False)},
            num_threads=num_threads,
            label='CHO Overlap',
            sub_label=f'[{size},{size}] (n={n})',
            description='torch.compile'
        ).blocked_autorange(min_run_time=0.2)
        results.append(t1)

        t2 = benchmark.Timer(
            stmt='compiled_cho_ng_overlap_matrix(**kwargs)',
            setup='from __main__ import compiled_cho_ng_overlap_matrix',
            globals={
                'kwargs': _make_gto_overlap_args(size, n, sparse=True)},
            num_threads=num_threads,
            label='CHO Overlap',
            sub_label=f'[{size},{size}] (n={n})',
            description='torch.compile (sparse)'
        ).blocked_autorange(min_run_time=0.2)
        results.append(t2)

    for n in [5, 10]:
        for size in [1, 4, 16, 64]:
            _run_benchmark(n, size, results)


def benchmark_cho_potential_overlap(results):
    num_threads = torch.get_num_threads()
    print(f'Benchmarking cho_potential_overlap on {num_threads} threads...')

    def _run_benchmark(n, m, size, results, threads=None):
        num_threads = torch.get_num_threads() if threads is None else threads

        t0 = benchmark.Timer(
            stmt='python_cho_ng_potential_overlap_matrix(**kwargs)',
            setup='from __main__ import python_cho_ng_potential_overlap_matrix', # noqa
            globals={
                'kwargs': 
                    _make_gto_potential_overlap_args(size, n, m, sparse=False)}, # noqa
            num_threads=num_threads,
            label='CHO Potential Overlap',
            sub_label=f'[{size},{size}] (n={n},m={m})',
            description='python',
        ).blocked_autorange(min_run_time=0.2)
        results.append(t0)

        t1 = benchmark.Timer(
            stmt='compiled_cho_ng_potential_overlap_matrix(**kwargs)',
            setup='from __main__ import compiled_cho_ng_potential_overlap_matrix', # noqa
            globals={
                'kwargs': 
                    _make_gto_potential_overlap_args(size, n, m, sparse=False)}, # noqa
            num_threads=num_threads,
            label='CHO Potential Overlap',
            sub_label=f'[{size},{size}] (n={n},m={m})',
            description='torch.compile'
        ).blocked_autorange(min_run_time=0.2)
        results.append(t1)

        t2 = benchmark.Timer(
            stmt='compiled_cho_ng_potential_overlap_matrix(**kwargs)',
            setup='from __main__ import compiled_cho_ng_potential_overlap_matrix', # noqa
            globals={
                'kwargs': 
                    _make_gto_potential_overlap_args(size, n, m, sparse=True)}, # noqa
            num_threads=num_threads,
            label='CHO Potential Overlap',
            sub_label=f'[{size},{size}] (n={n},m={m})',
            description='torch.compile (sparse)'
        ).blocked_autorange(min_run_time=0.2)
        results.append(t2)

    for n in [5, 10]:
        for m in [5, 10]:
            for size in [1, 4, 16, 64]:
                _run_benchmark(n, m, size, results)


if __name__ == '__main__':

    results = []

    # set cache size limit to maximum number of benchmark experiments
    torch._dynamo.config.cache_size_limit = 16

    benchmark_cho_overlap(results)
    benchmark_cho_potential_overlap(results)

    compare = benchmark.Compare(results)
    compare.print()
