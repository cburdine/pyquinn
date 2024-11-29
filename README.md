# pyquinn
Quantum-Informed Neural Networks Framework in Pytorch


## Development
* Installation
In this directory, run:
```
pip install -e .
```
(For now, an editable install with the `-e` flag is sufficient for development)

* Running Tests
In this directory, run:
```
python3 -m unittest discover
```

* Linting
In this directory, run:
```
flake8 ./pyquinn
```

* Build Documentation
In the `docs/` directory run:
```
make html
```

You can open the static html documentation in a web browser by opening `docs/build/html/index.html`.

## TODO:

[x] Add Unit Tests using the `unittest` module
[x] Add support for linting
[x] Build Documentation