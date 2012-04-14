"""The WaveBlocks Project

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

__version__ = 0.1

# Grids
from Grid import Grid
from DenseGrid import DenseGrid
from TensorProductGrid import TensorProductGrid
from GridFactory import GridFactory

# Wavefunctions
from WaveFunction import WaveFunction

# Potentials
from MatrixPotential import MatrixPotential
from MatrixPotential1S import MatrixPotential1S
from MatrixPotential2S import MatrixPotential2S
from MatrixPotentialMS import MatrixPotentialMS

from PotentialFactory import PotentialFactory



from KineticOperator import KineticOperator


from FourierPropagator import FourierPropagator



from IOManager import IOManager


# Basis shapes
from BasisShape import BasisShape
from HyperCubicShape import HyperCubicShape

# Wavepackets
from Wavepacket import Wavepacket
from HagedornWavepacketBase import HagedornWavepacketBase
from HagedornWavepacket import HagedornWavepacket
from HagedornWavepacketInhomogeneous import HagedornWavepacketInhomogeneous

# Quadrature rules
from QuadratureRule import QuadratureRule
from GaussHermiteQR import GaussHermiteQR
from TensorProductQR import TensorProductQR

# Inner products
from Quadrature import Quadrature
from HomogeneousQuadrature import HomogeneousQuadrature
from InhomogeneousQuadrature import InhomogeneousQuadrature

# Basis transformations
from BasisTransformation import BasisTransformation
from BasisTransformationWF import BasisTransformationWF
from BasisTransformationHAWP import BasisTransformationHAWP



# TODO: Recheck these files:
from ParameterLoader import ParameterLoader
from ParameterProvider import ParameterProvider

from TimeManager import TimeManager

from SimulationLoop import SimulationLoop
from SimulationLoopFourier import SimulationLoopFourier



from Observables import Observables
from ObservablesHAWP import ObservablesHAWP




# Enable dynamic plugin loading for IOManager
import sys
import os

plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(plugin_dir)
