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
    
    pip install -e .

Optional Dependencies
---------------------

Installing PyQuinn ensures that all core dependencies are installed, including the default
Pytorch backend. However, there may be some additional optional dependencies you may want to install:

**Visualization Packages**

To visualize atomic structures and plot predicted electronic bands, charge densities, etc., you will need to install either `matplotlib` (default) or `plotly`.

.. code-block:: console

    pip install matplotlib plotly

We recommend using `plotly` for interactive visualizations inside Python notebooks.


**Additional Backend Packages**

Currently, only the default Pytorch (`torch`) backend is supported. However, in the future
we aim to add support for other popular frameworks, such as `JAX`_ and `Tensorflow`_.

.. _JAX: https://docs.jax.dev/en/latest/index.html
.. _Tensorflow: https://www.tensorflow.org/

Examples of Basic Usage
-----------------------

* **Building and Compiling an AtomGraph**

.. code-block:: python

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

    # compile graph for PyTorch with a C cutoff radius of 2.0 Angstroms
    torch_graph = graph.compile(backend='torch', cutoffs={'C' : 2.0})

    # access Pytorch tensors associated with the compiled graph
    print(torch_graph.atomic_numbers)
    print(torch_graph.positions)
    print(torch_graph.neighbors)

* **Retrieving and Visualizing Structures from Databases**

.. code-block:: python

    from pyquinn.io.databases import COD
    import plotly

    # connect to the Crystallography Open Database (COD)
    with COD() as cod:

        # query the COD for structures
        ids = cod.get_atomgraph_ids_with_elements(['Si', 'O'])
        print('# of results:', len(ids))

        # retrieve an example atomgraph
        graph = cod.get_atomgraph_by_id(ids[0])

    # visualize structure
    fig = graph.plot()
    

* **Training a Basic Atomwise Model**

.. code-block:: python

    from pyquinn.types import AtomGraph, AtomGraphDataset
    from pyquinn.torch.models import TorchAtomwiseModel

    import pyquinn.torch.nn as pqnn

    # H2 bond lengths (Angstroms) and energies (eV)
    bond_lengths =  [ 0.4, 0.7, 1.0, 1.5, 2.0, 3.0 ]
    energies = [ 20.0, -4.4, -3.2, -1.2, -0.3, 0.0 ]

    # build the dataset
    dataset = AtomGraphDataset()
    for d, E in zip(bond_lengths, energies):
        dataset.append(AtomGraph(
            atoms=['H', 'H'],
            positions=[[0,0,0],[0,0,d]],
            graph_outputs=[E],
        ))

    # compile the dataset with an interaction cutoff of 4.0 Angstroms
    torch_dataset = dataset.compile(backend='torch', cutoffs={'H': 4.0})

    # make an Atomwise model to predict total energy:
    model = TorchAtomwiseModel(
        atomwise_pool='sum',
        out_features=1,
    )

    # fit model to the dataset with a learning rate of 0.1
    history = model.fit(torch_dataset, lr=0.1, verbosity=0)

    # Print the final training loss:
    print('Final training loss:', history['train_loss'][-1])





