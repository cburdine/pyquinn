from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'tqdm',
    'ase',
    'torch',
    'torch_geometric',
    'e3nn'
]

setup(
    name='pyquinn',
    version='0.1',
    install_requires=INSTALL_REQUIRES,
    description='A package for building Quantum-Informed Neural Networks.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # Tells setuptools that packages are inside src/
)
