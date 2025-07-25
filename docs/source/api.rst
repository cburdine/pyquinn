API
===

Overview
--------
PyQuINN provides a high-level, backend-agnostic interface for constructing
and manipulating atomistic graph representations of materials, training symmetry-aware
neural network models, and performing fast, differentiable electronic 
structure inference. At its core, PyQuINN defines a set of abstract 
graph-based primitives—such as :class:`~pyquinn.types.AtomGraph`, 
:class:`~pyquinn.types.BandStructure`, and :class:`~pyquinn.types.AtomGraphDataset`—that 
operate consistently across different machine learning backends. These primitives
are all contained in the :mod:`pyquinn.types` module.

Currently, only the PyTorch backend is supported, but support for in the future
we aim to add support for other popular frameworks, such as `JAX`_ and `Tensorflow`_ is
planned in future releases.

.. _JAX: https://docs.jax.dev/en/latest/index.html
.. _Tensorflow: https://www.tensorflow.org/



Backend API
---------------------
The PyQuinn API is designed to be modular and extensible: users can symbolically manipulate 
atomic structures using the :class:`~pyquinn.types.AtomGraph` object and compile them to 
backend-specific representations the conform to a common API
(e.g., see :class:`pyquinn.types.atom_graph.BackendAtomGraph``) 
using a unified `.compile(backend='<backend>')` interface. 

Although the primitives conform to a common API, strict adherence is not enforced for 
other objects, such as neural network modules. For these objects, PyQuINN 
instead adheres to the idioms and conventions of the target backend, while trying to be
as consistent as possible when these conventions are not violated. As an example, 
the Pytorch backend uses native PyTorch :class:`torch.nn.Module` subclasses, 
and respects device placement and dtype specification when working with objects. 
This design empowers users to build models while benefiting from the performance
and flexible tooling of their preferred machine learning framework.


Module Tree
---------------------

.. autosummary::
    :toctree: generated
    :recursive:

    pyquinn