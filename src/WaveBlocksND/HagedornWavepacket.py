"""The WaveBlocks Project

This file contains the class which represents a homogeneous Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, array, eye, atleast_2d, angle
from numpy.linalg import det

from HagedornWavepacketBase import HagedornWavepacketBase
from HyperCubicShape import HyperCubicShape
from ComplexMath import ContinuousSqrt

__all__ = ["HagedornWavepacket"]


class HagedornWavepacket(HagedornWavepacketBase):
    r"""This class represents homogeneous vector valued Hagedorn wavepackets
    :math:`\Psi` with :math:`N` components in :math:`D` space dimensions.
    """

    def __init__(self, dimension, ncomponents, eps):
        r"""Initialize a new homogeneous Hagedorn wavepacket.

        :param dimension: The space dimension :math:`D` the packet has.
        :param ncomponents: The number :math:`N` of components the packet has.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon` of the basis functions.
        :return: An instance of :py:class:`HagedornWavepacket`.
        """
        self._dimension = dimension
        self._number_components = ncomponents

        self._eps = eps

        # The basis shapes K_i
        self._basis_shapes = []
        # The coefficients c^i
        self._coefficients = []

        for d in xrange(self._number_components):
            # Default basis shapes for all components
            bs = HyperCubicShape( self._dimension*[1] )
            self._basis_shapes.append(bs)

            # A Gaussian
            self._coefficients.append(zeros((bs.get_basis_size(),1), dtype=complexfloating))

        # Cache basis sizes
        self._basis_sizes = [ bs.get_basis_size() for bs in self._basis_shapes ]

        # Default parameters of harmonic oscillator eigenstates
        q = zeros((self._dimension, 1))
        p = zeros((self._dimension, 1))
        Q = eye(self._dimension)
        P = 1.0j * eye(self._dimension)
        S = 0.0

        # The parameter set Pi
        self._Pis = [q, p, Q, P, S]

        # No quadrature set
        self._QE = None

        # Function for taking continuous roots
        self._sqrt = ContinuousSqrt(reference=angle(det(Q)))


    def __str__(self):
        r""":return: A string describing the Hagedorn wavepacket :math:`\Psi`.
        """
        s = ("Homogeneous Hagedorn wavepacket with "+str(self._number_components)
             +" component(s) in "+str(self._dimension)+" space dimension(s)\n")
        return s


    def _get_sqrt(self, component):
        r"""Compytibility method
        """
        return self._sqrt


    def get_description(self):
        r"""Return a description of this wavepacket object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "HagedornWavepacket"
        d["dimension"] = self._dimension
        d["ncomponents"] = self._number_components
        d["eps"] = self._eps
        if self._QE is not None:
            d["quadrature"] = self._QE.get_description()
        return d


    def clone(self, keepid=False):
        # Parameters of this packet
        params = self.get_description()
        # Create a new Packet
        # TODO: Consider using the block factory
        other = HagedornWavepacket(params["dimension"],
                                   params["ncomponents"],
                                   params["eps"])
        # If we wish to keep the packet ID
        if keepid is True:
            other.set_id(self.get_id())
        # And copy over all (private) data
        # Basis shapes are immutable, no issues with sharing same instance
        other.set_basis_shapes(self.get_basis_shapes())
        other.set_parameters(self.get_parameters())
        other.set_coefficients(self.get_coefficients())
        # Quadratures are immutable, no issues with sharing same instance
        other.set_quadrature(self.get_quadrature())
        # The complex root cache
        other._sqrt = self._sqrt.clone()

        return other


    def get_parameters(self, component=None, aslist=False, key=("q","p","Q","P","S")):
        r"""Get the Hagedorn parameter set :math:`\Pi` of the wavepacket :math:`\Psi`.

        :param component: Dummy parameter for API compatibility with the inhomogeneous packets.
        :param aslist: Return a list of :math:`N` parameter tuples. This is for API compatibility
                       with inhomogeneous packets.
        :return: The Hagedorn parameter set :math:`\Pi = (q, p, Q, P, S)` in this order.
        """
        Pilist = []
        for k in key:
            if k == "q":
                Pilist.append(self._Pis[0])
            elif k == "p":
                Pilist.append(self._Pis[1])
            elif k == "Q":
                Pilist.append(self._Pis[2])
            elif k == "P":
                Pilist.append(self._Pis[3])
            elif k == "S":
                Pilist.append(self._Pis[4])
            elif k == "adQ":
                Pilist.append(array(self._get_sqrt(component).get(), dtype=complexfloating))
            else:
                raise KeyError("Invalid parameter key: "+str(key))

        if aslist is True:
            return self._number_components * [ Pilist ]

        return Pilist


    def set_parameters(self, Pi, component=None, key=("q","p","Q","P","S")):
        r"""Set the Hagedorn parameters :math:`\Pi` of the wavepacket :math:`\Psi`.

        :param Pi: The Hagedorn parameter set :math:`\Pi = (q, p, Q, P, S)` in this order.
        :param component: Dummy parameter for API compatibility with the inhomogeneous packets.
        """
        for k, item in zip(key, Pi):
            if k == "q":
                self._Pis[0] = atleast_2d(array(item, dtype=complexfloating))
            elif k == "p":
                self._Pis[1] = atleast_2d(array(item, dtype=complexfloating))
            elif k == "Q":
                self._Pis[2] = atleast_2d(array(item, dtype=complexfloating))
            elif k == "P":
                self._Pis[3] = atleast_2d(array(item, dtype=complexfloating))
            elif k == "S":
                self._Pis[4] = atleast_2d(array(item, dtype=complexfloating))
            elif k == "adQ":
                self._get_sqrt(component).set(item)
            else:
                raise KeyError("Invalid parameter key: "+str(key))
