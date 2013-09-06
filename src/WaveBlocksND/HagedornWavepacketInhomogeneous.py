"""The WaveBlocks Project

This file contains the class which represents an inhomogeneous Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, array, eye, atleast_2d, angle, squeeze
from numpy.linalg import det

from HagedornWavepacketBase import HagedornWavepacketBase
from HyperCubicShape import HyperCubicShape
from ComplexMath import ContinuousSqrt

__all__ = ["HagedornWavepacketInhomogeneous"]


class HagedornWavepacketInhomogeneous(HagedornWavepacketBase):
    r"""This class represents inhomogeneous vector valued Hagedorn wavepackets
    :math:`\Psi` with :math:`N` components in :math:`D` space dimensions.
    """

    def __init__(self, dimension, ncomponents, eps):
        r"""Initialize a new in homogeneous Hagedorn wavepacket.

        :param dimension: The space dimension :math:`D` the packet has.
        :param ncomponents: The number :math:`N` of components the packet has.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon` of the basis functions.
        :return: An instance of :py:class:`HagedornWavepacketInhomogeneous`.
        """
        self._dimension = dimension
        self._number_components = ncomponents

        self._eps = eps

        # The parameter sets Pi_i
        self._Pis = []
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

            # Default parameters of harmonic oscillator eigenstates
            q = zeros((self._dimension, 1), dtype=complexfloating)
            p = zeros((self._dimension, 1), dtype=complexfloating)
            Q = eye(self._dimension, dtype=complexfloating)
            P = 1.0j * eye(self._dimension, dtype=complexfloating)
            S = zeros((1, 1), dtype=complexfloating)

            self._Pis.append([q, p, Q, P, S])

        # Cache basis sizes
        self._basis_sizes = [ bs.get_basis_size() for bs in self._basis_shapes ]

        # No inner product set
        self._IP = None

        # Function for taking continuous roots
        self._sqrt = [ ContinuousSqrt(angle(det(self._Pis[n][2]))) for n in xrange(self._number_components) ]


    def __str__(self):
        r""":return: A string describing the Hagedorn wavepacket :math:`\Psi`.
        """
        s = ("Inhomogeneous Hagedorn wavepacket with "+str(self._number_components)
             +" component(s) in "+str(self._dimension)+" space dimension(s)\n")
        return s


    def _get_sqrt(self, component):
        r"""Compatibility method
        """
        return self._sqrt[component]


    def get_description(self):
        r"""Return a description of this wavepacket object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "HagedornWavepacketInhomogeneous"
        d["dimension"] = self._dimension
        d["ncomponents"] = self._number_components
        d["eps"] = self._eps
        if self._IP is not None:
            d["innerproduct"] = self._IP.get_description()
        return d


    def clone(self, keepid=False):
        # Parameters of this packet
        params = self.get_description()
        # Create a new Packet
        # TODO: Consider using the block factory
        other = HagedornWavepacketInhomogeneous(params["dimension"],
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
        # Innerproducts are stateless and finally immutable,
        # no issues with sharing same instance
        other.set_innerproduct(self.get_innerproduct())
        # The complex root caches
        other._sqrt = [ item.clone() for item in self._sqrt ]

        return other


    def get_parameters(self, component=None, aslist=False, key=("q","p","Q","P","S")):
        r"""Get the Hagedorn parameter set :math:`\Pi_i` of each component :math`\Phi_i`
        of the wavepacket :math:`\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` whose
                          parameters :math:`\Pi_i` we want to get.
        :param aslist: Dummy parameter for API compatibility with the homogeneous packets.
        :return: A list with all parameter sets :math:`\Pi_i` or a single parameter set.
                 The parameters :math:`\Pi_i = (q_i, p_i, Q_i, P_i, S_i)` are always in this order.
        """
        if component is None:
            components = xrange(self._number_components)
        else:
            components = [component]

        Pilist = []
        for index in components:
            tmp = []
            for k in key:
                if k == "q":
                    tmp.append(self._Pis[index][0])
                elif k == "p":
                    tmp.append(self._Pis[index][1])
                elif k == "Q":
                    tmp.append(self._Pis[index][2])
                elif k == "P":
                    tmp.append(self._Pis[index][3])
                elif k == "S":
                    tmp.append(self._Pis[index][4])
                elif k == "adQ":
                    tmp.append(array(self._get_sqrt(index).get(), dtype=complexfloating))
                else:
                    raise KeyError("Invalid parameter key: "+str(key))

            Pilist.append(tmp)

        if component is not None:
            return Pilist[0]

        return Pilist


    def set_parameters(self, Pi, component=None, key=("q","p","Q","P","S")):
        r"""Set the Hagedorn parameter set :math:`\Pi_i` of each component :math`\Phi_i`
        of the wavepacket :math:`\Psi`.

        :param Pi: The parameter sets :math:`\Pi_i = (q_i, p_i, Q_i, P_i, S_i)` with its values in this order.
        :type Pi: A single tuple or a list of tuples
        :param component: The index :math:`i` of the component :math:`\Phi_i` whose parameters :math:`\Pi_i` we want to update.
        """
        if component is None:
            component = xrange(self._number_components)
        else:
            component = [component]
            Pi = [Pi]

        for index, pic in zip(component, Pi):
            for k, item in zip(key, pic):
                if k == "q":
                    self._Pis[index][0] = atleast_2d(array(item, dtype=complexfloating))
                elif k == "p":
                    self._Pis[index][1] = atleast_2d(array(item, dtype=complexfloating))
                elif k == "Q":
                    self._Pis[index][2] = atleast_2d(array(item, dtype=complexfloating))
                elif k == "P":
                    self._Pis[index][3] = atleast_2d(array(item, dtype=complexfloating))
                elif k == "S":
                    self._Pis[index][4] = atleast_2d(array(item, dtype=complexfloating))
                elif k == "adQ":
                    self._get_sqrt(index).set(squeeze(item))
                else:
                    raise KeyError("Invalid parameter key: "+str(key))
