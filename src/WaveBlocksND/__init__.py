"""The WaveBlocks Project

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

__version__ = 0.1

# Math
from ComplexMath import ContinuousSqrt

# Grids
from Grid import Grid
from DenseGrid import DenseGrid
from TensorProductGrid import TensorProductGrid
from GridFactory import GridFactory
from GridWrapper import GridWrapper

# Wavefunctions
from WaveFunction import WaveFunction

# Potentials
from MatrixPotential import MatrixPotential
from MatrixPotential1S import MatrixPotential1S
from MatrixPotential2S import MatrixPotential2S
from MatrixPotentialMS import MatrixPotentialMS

from PotentialFactory import PotentialFactory



from KineticOperator import KineticOperator

# Time Propagators
from Propagator import Propagator
from FourierPropagator import FourierPropagator
from HagedornPropagator import HagedornPropagator



from IOManager import IOManager

# Basis shapes
from BasisShape import BasisShape
from HyperCubicShape import HyperCubicShape
from HyperbolicCutShape import HyperbolicCutShape
from LimitedHyperbolicCutShape import LimitedHyperbolicCutShape

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
from BlockFactory import BlockFactory

from ParameterLoader import ParameterLoader
from ParameterProvider import ParameterProvider

from TimeManager import TimeManager

from SimulationLoop import SimulationLoop
from SimulationLoopFourier import SimulationLoopFourier
from SimulationLoopHagedorn import SimulationLoopHagedorn



from Observables import Observables
from ObservablesHAWP import ObservablesHAWP

#from MatrixExponential import MatrixExponential
from MatrixExponentialFactory import MatrixExponentialFactory

# Enable dynamic plugin loading for IOManager
import sys
import os

plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(plugin_dir)
