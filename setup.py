from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'tqdm',
    'ase',
    'torch',
    'torch_geometric'
]

setup(
    name='pyquinn',
    version='0.1',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages()
)
