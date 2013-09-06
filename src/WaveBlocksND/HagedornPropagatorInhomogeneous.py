"""The WaveBlocks Project

This file contains the Hagedorn propagator class for inhomogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import dot, eye, atleast_2d
from numpy.linalg import inv, det

from Propagator import Propagator
from BlockFactory import BlockFactory
from ComplexMath import cont_angle

__all__ = ["HagedornPropagatorInhomogeneous"]


class HagedornPropagatorInhomogeneous(Propagator):
    r"""This class can numerically propagate given initial values :math:`\Psi` in
    a potential :math:`V(x)`. The propagation is done for a given set of inhomogeneous
    Hagedorn wavepackets neglecting interaction."""

    def __init__(self, parameters, potential, packets=[]):
        r"""Initialize a new :py:class:`HagedornPropagatorInhomogeneous` instance.

        :param parameters: A :py:class:`ParameterProvider` instance containing at least
                           the key ``dt`` for providing the timestep :math:`\tau`.
        :type parameters: A :py:class:`ParameterProvider` instance

        :param potential: The potential :math:`V(x)` the wavepacket :math:`\Psi` feels during the time propagation.
        :param packet: The initial homogeneous Hagedorn wavepacket :math:`\Psi` we propagate in time.
        :raises ValueError: If the number of components of :math:`\Psi` does not match
                            the number of energy levels :math:`\lambda_i` of the potential.
        """
        # The potential :math:`V(x)` the packet(s) feel.
        self._potential = potential

        # Number :math:`N` of components the wavepacket :math:`\Psi` has got.
        self._number_components = self._potential.get_number_components()
        self._dimension = self._potential.get_dimension()

        # A list of Hagedorn wavepackets :math:`\Psi` together with some codata
        # At the moment we do not use any codata here.
        # TODO: Do not use a list but better use a hashtable by packet IDs?
        self._packets = packets[:]

        # Keep a reference to the parameter provider instance
        self._parameters = parameters

        self._dt = self._parameters["dt"]

        # The relative mass scaling matrix M
        if self._parameters.has_key("mass_scaling"):
            self._M = atleast_2d(self._parameters["mass_scaling"])
            assert self._M.shape == (self._dimension, self._dimension)
            self._Minv = inv(self._M)
        else:
            # No mass matrix given. Scale all masses equally
            self._M = eye(self._dimension)
            self._Minv = self._M

        # Decide about the matrix exponential algorithm to use
        self.__dict__["_matrix_exponential"] = BlockFactory().create_matrixexponential(parameters)

        # Precalculate the potential splittings needed
        self._potential.calculate_local_quadratic()
        self._potential.calculate_local_remainder()


    def __str__(self):
        r"""Prepare a printable string representing the :py:class:`HagedornPropagatorInhomogeneous` instance."""
        return "Inhomogeneous Hagedorn propagator for " + str(self._number_components) + " components.\n"


    def add_wavepacket(self, packet):
        r"""Add a new wavepacket :math:`\Psi` to the list of propagated wavepackets.

        :param packet: The new wavepacket :math:`\Psi`.
        :type packet: A tuple :math:`(\Psi,)` with :math:`\Psi` a :py:class:`HagedornWavepacketInhomogeneous`
                      instance.
        """
        self._packets.append(tuple(packet))


    # TODO: Consider removig this, duplicate
    def get_number_components(self):
        r""":return: The number :math:`N` of components :math:`\Phi_i` of :math:`\Psi`.
        """
        return self._number_components


    def get_wavepackets(self, packet=None):
        r"""Return the wavepackets :math:`\{\Psi_i\}_i` that take part in the time propagation by the
        current :py:class:`HagedornPropagatorInhomogeneous` instance.

        :param packet: The index :math:`i` (in this list) of a single packet :math:`\Psi_i` that is
                       to be returned. If set to ``None`` (default) return the full list with all packets.
        :type packet: Integer or ``None``
        :return: A list of :py:class:`HagedornWavepacketInhomogeneous` instances or a single instance.
        """
        if packet is None:
            return [ p[0] for p in self._packets ]
        else:
            return self._packets[packet][0]


    def set_wavepackets(self, packetlist):
        r"""Set the list :math:`\{\Psi_i\}_i` of wavepackets that the propagator will propagate.

        :param packetlist: A list of new wavepackets :math:`\Psi_i` and their codata to propagate.
        :type packetlist: A list of :math:`(\Psi_i,)` tuples.
        """
        self._packets = packetlist[:]


    def propagate(self):
        r"""Given a wavepacket :math:`\Psi` at time :math:`t` compute the propagated
        wavepacket at time :math:`t + \tau`. We perform exactly one timestep of size
        :math:`\tau` here. This propagation is done for all packets in the list
        :math:`\{\Psi_i\}_i` and neglects any interaction between two packets.
        """
        # Cache some parameter values
        dt = self._dt
        Mi = self._Minv

        # Propagate all packets
        for packet in self._packets:
            # Unpack, no codata:
            packet = packet[0]
            eps = packet.get_eps()
            key = ("q", "p", "Q", "P", "S", "adQ")

            # Do a kinetic step of dt/2
            for component in xrange(self._number_components):
                q, p, Q, P, S, adQ = packet.get_parameters(component=component, key=key)
                q = q + 0.5 * dt * dot(Mi, p)
                Q = Q + 0.5 * dt * dot(Mi, P)
                S = S + 0.25 * dt * dot(p.T, dot(Mi, p))
                adQn = cont_angle(det(Q), reference=adQ)[0]
                packet.set_parameters((q, p, Q, P, S, adQn), component=component, key=key)

            # Do a potential step with the local quadratic part
            for component in xrange(self._number_components):
                q, p, Q, P, S = packet.get_parameters(component=component)
                V = self._potential.evaluate_local_quadratic_at(q, diagonal_component=component)

                p = p - dt * V[1]
                P = P - dt * dot(V[2], Q)
                S = S - dt * V[0]
                packet.set_parameters((q, p, Q, P, S), component=component)

            # Do a potential step with the local non-quadratic Taylor remainder
            innerproduct = packet.get_innerproduct()
            F = innerproduct.build_matrix(packet, packet, self._potential.evaluate_local_remainder_at)

            coefficients = packet.get_coefficient_vector()
            coefficients = self._matrix_exponential(F, coefficients, dt/eps**2)
            packet.set_coefficient_vector(coefficients)

            # Do a kinetic step of dt/2
            for component in xrange(self._number_components):
                q, p, Q, P, S, adQ = packet.get_parameters(component=component, key=key)
                q = q + 0.5 * dt * dot(Mi, p)
                Q = Q + 0.5 * dt * dot(Mi, P)
                S = S + 0.25 * dt * dot(p.T, dot(Mi, p))
                adQn = cont_angle(det(Q), reference=adQ)[0]
                packet.set_parameters((q, p, Q, P, S, adQn), component=component, key=key)
