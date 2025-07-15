.. PyQuinn documentation master file, created by
   sphinx-quickstart on Sat Nov 23 17:07:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyQuinn's documentation!
===================================

**PyQuinn** is a Pytorch-based Python Library for the development of Quantum-Informed Neural Networks (QuINNs).


*What is a Quantum-Informed neural network?*

QuINNs are a type of Physics-Informed Neural Network (PINN), which learn mean-field solutions 
to the electronic time-independent Schrodinger equation for molecules and solid state materials.
The PyQuinn framework provides a simple and extensible framework for rapidly building and training 
QuINN models, as well as more traditional graph neural networks. QuINNs can be trained to
predict atom-wise properties, force fields, and even macroscopic material properties from raw atomic
structures.


For installation instructions, see the :doc:`quickstart` section.

.. toctree::
   :maxdepth: 1
   :caption: Table of Contents:
   
   quickstart
   api



Quick Reference
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
