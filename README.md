# PyQuinn
A Framework for Quantum-Informed Neural Networks Framework in Pytorch


## Development

This package is stil under development, so our development procedures are likely to change in the near future until we have settled on a 
good set of standard practices. 

### Installation

In this directory, run:
```
pip install -e .
```
(For now, an editable install with the `-e` flag is sufficient for development. As the package grows larger, we may recommend something different.)

### Running Tests

In this directory, run:
```
python3 -m unittest discover
```

### Linting

WE are currently using [flake8](https://flake8.pycqa.org/en/latest/) for linting. (You can install flake8 using `python -m pip install flake8`).

To lint all files, run the command:
```
flake8 ./pyquinn/
```

### Building Documentation

We are curently building documentation using [sphinx](https://www.sphinx-doc.org/en/master/). In the `docs/` directory run:
```
make html
```

You can view the static html documentation in a web browser by opening `docs/build/html/index.html`.

### Run Benchmarks

The benchmarks are currently standalone executable python files in the `benchmark/` directory. For example, to run the benchmarks for the `pyquinn.torch.lcao.gto` submodule, run the command
```
python3 ./benchmark/torch/benchmark_lcao_gto.py
```

---

## Feature TODO List:

- [x] Add basic functionality of the `torch.lcao.gto` subpackage 
- [ ] Add basic support for spin-paired states in the `LCAOModel` and `TightBindingModel` modules.
- [ ] Implement support for spin-polarized states in the `LCAOModel` and `TightBindingModel` modules.
- [ ] Add support for evaluating $\hat{x}$ and $\hat{p}$ matrices in the `LCAOModel` and `TightBindingModel` modules. 

## Development TODO List:

- [x] Add Unit Tests using the `unittest` module
- [x] Add support for linting
- [x] Build documentation
- [ ] Add benchmarking suite
- [ ] Add a tutorial repository
- [ ] Add a CI pipline (via github actions)