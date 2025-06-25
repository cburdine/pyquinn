Quickstart
==========

Installation
------------

To use PyQuinn, you will need to install the necessary dependencies.
Since this project is still under development, you can see the most
up-to-date installation instructions in the `README.md` file.

**Installing from Source**

To install PyQuinn from source, you can use `pip`:
.. code-block:: console
    
    $ pip install -e .

Optional Dependencies
-----------------------

Installing PyQuinn ensures that all core dependencies are installed, including the default
Pytorch backend. However, there may be some additional optional dependencies you may want to install:

**Visualization Packages**

To visualize atomic structures and plot predicted electronic bands, charge densities, etc., you will need to install either `matplotlib` (default) or `plotly`.

    $ pip install matplotlib plotly

We recommend using `plotly` for interactive visualizations.


**Additional Backend Packages**

Currently, only the default Pytorch (`torch`) backend is supported. However, in the future
we aim to add support for other popular frameworks, such as [JAX](https://docs.jax.dev/en/latest/index.html) and (Tensorflow)[https://www.tensorflow.org/].

Basic Usage
-----------

**Building and Compiling an AtomGraph**

```
from pyquinn.types import AtomGraph

# create a 2D periodic graph of carbon atoms
graph = AtomGraph(
    atoms=['C', 'C'],
    positions=[
        [0.0,   0.0,   0.0], 
        [1.23,  0.71,  0.0]],
    unit_cell=[
        [2.46,  0.0,  0.0], 
        [-1.23, 2.13, 0.0], 
        [0.0,   0.0,  0.0]],
    periodic=(True, True, False))

# compile graph for PyTorch with neighbor cutoff radius of 2.0 Angstroms
torch_graph = graph.compile(backend='torch', cutoffs={'C' : 2.0})

# access Pytorch tensors associated with compiled graph
print(torch_graph.atomic_numbers)
print(torch_graph.positions)
print(torch_graph.neighbors)
```




