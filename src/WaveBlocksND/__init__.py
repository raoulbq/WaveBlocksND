"""The WaveBlocks Project

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013, 2014 R. Bourquin
@license: Modified BSD License
"""

__version__ = 0.2

# Math
from ComplexMath import ContinuousSqrt

# Grids
from Grid import Grid
from DenseGrid import DenseGrid
from TensorProductGrid import TensorProductGrid
from GridWrapper import GridWrapper

from PhaseSpaceLattice import PhaseSpaceLattice

# Wavefunctions
from WaveFunction import WaveFunction

# Potentials
from MatrixPotential import MatrixPotential
from MatrixPotential1S import MatrixPotential1S
from MatrixPotential2S import MatrixPotential2S
from MatrixPotentialMS import MatrixPotentialMS

from KineticOperator import KineticOperator

# Time Propagators
from Propagator import Propagator
from FourierPropagator import FourierPropagator
from HagedornPropagator import HagedornPropagator
from MagnusPropagator import MagnusPropagator
from SemiclassicalPropagator import SemiclassicalPropagator
from HagedornPropagatorInhomogeneous import HagedornPropagatorInhomogeneous
from SplittingParameters import SplittingParameters

from NLSEPropagator import NLSEPropagator

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

from Gradient import Gradient
from GradientHAWP import GradientHAWP

# Linear combinations
from LinearCombinationOfWavepackets import LinearCombinationOfWavepackets
from LinearCombinationOfWPs import LinearCombinationOfWPs
from LinearCombinationOfHAWPs import LinearCombinationOfHAWPs

from GradientLCWP import GradientLCWP
from GradientLinearCombinationHAWP import GradientLinearCombinationHAWP

# Quadrature rules
from QuadratureRule import QuadratureRule
from TrapezoidalQR import TrapezoidalQR
from GaussHermiteOriginalQR import GaussHermiteOriginalQR
from GaussHermiteQR import GaussHermiteQR
from GaussLaguerreQR import GaussLaguerreQR
from TensorProductQR import TensorProductQR
from SmolyakQR import SmolyakQR

# Inner products
from InnerProduct import InnerProduct
from InnerProduct import InnerProductException
from HomogeneousInnerProduct import HomogeneousInnerProduct
from InhomogeneousInnerProduct import InhomogeneousInnerProduct

from HomogeneousInnerProductLCWP import HomogeneousInnerProductLCWP
from InhomogeneousInnerProductLCWP import InhomogeneousInnerProductLCWP

from Quadrature import Quadrature
from DirectQuadrature import DirectQuadrature
from DirectHomogeneousQuadrature import DirectHomogeneousQuadrature
from DirectInhomogeneousQuadrature import DirectInhomogeneousQuadrature

from SymbolicIntegral import SymbolicIntegral
from SymbolicIntegral0 import SymbolicIntegral0
from GaussianIntegral import GaussianIntegral
from NSDInhomogeneous import NSDInhomogeneous

from SparsityOracle import SparsityOracle
from SparsityOracleTrue import SparsityOracleTrue
from SparsityOraclePSHAWP import SparsityOraclePSHAWP
from SparsityOracleGIHAWP import SparsityOracleGIHAWP

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
from SimulationLoopHagedornInhomogeneous import SimulationLoopHagedornInhomogeneous

from Observables import Observables
from ObservablesHAWP import ObservablesHAWP
from ObservablesMixedHAWP import ObservablesMixedHAWP
from ObservablesLCWP import ObservablesLCWP

# Enable dynamic plugin loading for IOManager
import sys
import os

plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(plugin_dir)
