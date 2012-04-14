Welcome to WaveBlocksND's documentation!
========================================

.. image:: _static/waveblocks.png

Reusable building blocks for simulations with semiclassical wavepackets for
solving the time-dependent Schrödinger equation.

Contents:

.. toctree::
   :maxdepth: 2

Source code documentation
=========================


WaveBlocks Classes
-------------------

Basic numerics
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/Grid
   waveblocks_classes/DenseGrid
   waveblocks_classes/TensorProductGrid

   waveblocks_classes/QuadratureRule
   waveblocks_classes/GaussHermiteQR
   waveblocks_classes/TensorProductQR

   waveblocks_classes/Utils

Basic quantum mechanics
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/WaveFunction

   waveblocks_classes/MatrixPotential
   waveblocks_classes/MatrixPotential1S

   waveblocks_classes/PotentialFactory

Wavepackets
^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/BasisShape
   waveblocks_classes/HyperCubicShape

   waveblocks_classes/Wavepacket
   waveblocks_classes/HagedornWavepacketBase
   waveblocks_classes/HagedornWavepacket
   waveblocks_classes/HagedornWavepacketInhomogeneous

   waveblocks_classes/Quadrature
   waveblocks_classes/HomogeneousQuadrature
   waveblocks_classes/InhomogeneousQuadrature

   waveblocks_classes/BasisTransformation
   waveblocks_classes/BasisTransformationWF
   waveblocks_classes/BasisTransformationHAWP


Observables
^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/Observables
   waveblocks_classes/ObservablesHAWP


Time propagation
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/Propagator
   waveblocks_classes/FourierPropagator

Simulation result storage I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   waveblocks_classes/IOManager

   waveblocks_classes/IOM_plugin_parameters

Other classes
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/GlobalDefaults
   waveblocks_classes/ParameterLoader
   waveblocks_classes/ParameterProvider
   waveblocks_classes/Utils


Etc
===

.. toctree::
   :maxdepth: 2

   citation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
