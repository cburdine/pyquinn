# PyQuinn
A Framework for Quantum-Informed Neural Networks Framework in Pytorch

## About
---

(Package summary will go here)

## Development
---
This package is still under development, so our development procedures are likely to change in the near future until we have settled on a good set of standard practices. 

### Installation

In this directory, run:
```
pip install -e .
```
(For now, an editable install with the `-e` flag is sufficient for development. As the package grows larger, we may recommend something different.)

### (Optional) Installing Development Tools
Development tools are now provided via an optional dependency group in `pyproject.toml`. To install them, run:
```
pip install -e ".[dev]"
```

### Running Tests

In this directory, run:
```
python -m unittest discover

```

### Linting

We use:
- [black](https://github.com/psf/black) for formatting
- [ruff](https://github.com/astral-sh/ruff) for linting

Both tools are configured in `pyproject.toml`, including a fixed target Python version (`py310`) and line length (80 chars) so results are consistent even if contributors have newer Python versions installed.

To format all files, run:
```
black .
```

To lint all files, run:
```
ruff check .
```

To auto-fix what Ruff can fix safely, run:
```
ruff check . --fix
```

To lint a specific directory, for example:
```
ruff check src/pyquinn/
```
Many IDEs (such as VS Code also support extensions for linters)

### Building Documentation

Before building the documentation, make sure that you have the development tools installed (see instructions above).

We are curently building documentation using [sphinx](https://www.sphinx-doc.org/en/master/). In the `docs/` directory run:
```
make html
```

You now view the static html documentation in a web browser by opening `docs/build/html/index.html` or by running a local server with the command
```
python -m http.server --directory docs/build/html
```

### Run Benchmarks

The benchmarks are currently standalone executable python files in the `benchmark/` directory. For example, to run the benchmarks for the `pyquinn.torch.lcao.gto` submodule, run the command
```
python benchmark/torch/benchmark_lcao_gto.py
```

---

## Feature TODO List:

- [x] Add basic functionality of the `torch.lcao.gto` subpackage 
- [x] Add basic support for spin-paired states in the `LCAOHamiltonian` and `TBHamiltonian` modules.
- [x] Add support for evaluating $\hat{x}$ and $\hat{p}$ matrices in the `LCAOHamiltonian` and `TBHamiltonian` modules. 
- [ ] Implement support for spin-polarized states in the `LCAOHamiltonian` and `TBHamiltonian` modules.

## Development TODO List:

- [x] Add Unit Tests using the `unittest` module
- [x] Add support for linting
- [x] Build documentation
- [x] Add benchmarking suite
- [x] Migrate linter to [Black](https://github.com/psf/black)
- [ ] Add a tutorial repository
- [ ] Migrate unit testing to PyTest
- [ ] Add a CI pipline (via github actions)


