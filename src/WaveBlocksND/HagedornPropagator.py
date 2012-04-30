"""The WaveBlocks Project

This file contains the Hagedorn propagator class for homogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
from numpy import dot

from Propagator import Propagator
from MatrixExponentialFactory import MatrixExponentialFactory


class HagedornPropagator(Propagator):
    r"""This class can numerically propagate given initial values :math:`\Psi` in
    a potential :math:`V(x)`. The propagation is done for a given set of homogeneous
    Hagedorn wavepackets neglecting interaction."""

    def __init__(self, parameters, potential, packets=[]):
        r"""Initialize a new :py:class:`HagedornPropagator` instance.

        :param parameters: A :py:class:`ParameterProvider` instance containing at least
                           the key ``dt`` for providing the timestep :math:`\tau`.
        :type parameters: A :py:class:`ParameterProvider` instance

        :param potential: The potential :math:`V(x)` the wavepacket :math:`\Psi` feels during the time propagation.
        :param packet: The initial homogeneous Hagedorn wavepacket :math:`\Psi` we propagate in time.
        :param leading_component: The leading component index :math:`\chi`.


        :raises ValueError: If the number of components of :math:`\Psi` does not match
                            the number of energy levels :math:`\lambda_i` of the potential.
        """
        # The potential :math:`V(x)` the packet(s) feel.
        self._potential = potential

        # Number :math:`N` of components the wavepacket :math:`\Psi` has got.
        self._number_components = self._potential.get_number_components()

        # A list of Hagedorn wavepackets :math:`\Psi` together with some codata
        # like the leading component :math:`\chi` which is the index of the eigenvalue
        # :math:`\lambda_\chi` of the potential :math:`V` that is responsible for
        # propagating the Hagedorn parameters.
        # TODO: We assume a list of (packet, leading_component) tuples here. Generalize tuples to dicts!
        # TODO: Do not use a list but better use a hashtable by packet IDs?
        self._packets = packets[:]

        # Keep a reference to the parameter provider instance
        self._parameters = parameters

        self._dt = self._parameters["dt"]

        # Decide about the matrix exponential algorithm to use
        self.__dict__["_matrix_exponential"] = MatrixExponentialFactory().get_matrixexponential(parameters)

        # Precalculate the potential splittings needed
        self._prepare_potential()


    def __str__(self):
        r"""Prepare a printable string representing the :py:class:`HagedornPropagator` instance."""
        return "Hagedorn propagator for " + str(self._number_components) + " components.\n"


    def _prepare_potential(self):
        """Precalculate the potential splittings needed
        """
        for chi in set([ p[1] for p in self._packets ]):
            self._potential.calculate_local_quadratic(diagonal_component=chi)
            self._potential.calculate_local_remainder(diagonal_component=chi)


    def add_wavepacket(self, packet):
        r"""Add a new wavepacket :math:`\Psi` to the list of propagated wavepackets.

        :param packet: The new wavepacket :math:`\Psi` and its leading component :math:`\chi \in [0,N-1]`.
        :type packet: A tuple :math:`(\Psi, \chi)` with :math:`\Psi` a :py:class:`HagedornWavepacket`
                      instance and :math:`\chi` an integer.
        """
        self._packets.append(tuple(packet))
        self._prepare_potential()


    # TODO: Consider removig this, duplicate
    def get_number_components(self):
        r""":return: The number :math:`N` of components :math:`\Phi_i` of :math:`\Psi`.
        """
        return self._number_components


    def get_wavepackets(self, packet=None):
        r"""Return the wavepackets :math:`\{\Psi_i\}_i` that take part in the time propagation by the
        current :py:class:`HagedornPropagator` instance.

        :param packet: The index :math:`i` (in this list) of a single packet :math:`\Psi_i` that is
                       to be returned. If set to ``None`` (default) return the full list with all packets.
        :type packet: Integer or ``None``
        :return: A list of :py:class:`HagedornWavepacket` instances or a single instance.
        """
        # TODO: Does not return leading components. Add this if needed somewhere.
        if packet is None:
            return [ p[0] for p in self._packets ]
        else:
            return self._packets[packet][0]


    def set_wavepackets(self, packetlist):
        """Set the list :math:`\{\Psi_i\}_i` of wavepackets that the propagator will propagate.

        :param packetlist: A list of new wavepackets :math:`\Psi_i` and their
                           leading components :math:`\chi_i` to propagate.
        :type packetlist: A list of :math:`(\Psi_i, \chi_i)` tuples.
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

        # Propagate all packets
        for packet, leading_chi in self._packets:
            eps = packet.get_eps()

            # Do a kinetic step of dt/2
            q, p, Q, P, S = packet.get_parameters()
            q = q + 0.5 * dt * p
            Q = Q + 0.5 * dt * P
            S = S + 0.25 * dt * dot(p.T, p)
            packet.set_parameters((q, p, Q, P, S))

            # Do a potential step with the local quadratic part
            q, p, Q, P, S = packet.get_parameters()
            V = self._potential.evaluate_local_quadratic_at(q, diagonal_component=leading_chi)

            p = p - dt * V[1]
            P = P - dt * dot(V[2], Q)
            S = S - dt * V[0]
            packet.set_parameters((q, p, Q, P, S))

            # Do a potential step with the local non-quadratic taylor remainder
            quadrature = packet.get_quadrature()
            F = quadrature.build_matrix(packet, operator=partial(self._potential.evaluate_local_remainder_at, diagonal_component=leading_chi))

            coefficients = packet.get_coefficient_vector()
            coefficients = self._matrix_exponential(F, coefficients, dt/eps**2)
            packet.set_coefficient_vector(coefficients)

            # Do a kinetic step of dt/2
            q, p, Q, P, S = packet.get_parameters()
            q = q + 0.5 * dt * p
            Q = Q + 0.5 * dt * P
            S = S + 0.25 * dt * dot(p.T, p)
            packet.set_parameters((q, p, Q, P, S))
