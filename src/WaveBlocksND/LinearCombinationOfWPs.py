"""The WaveBlocks Project

This file contains the class which represents linear combinations
of general but compatible wavepackets of any kind.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, complexfloating, atleast_2d, delete, vstack

from LinearCombinationOfWavepackets import LinearCombinationOfWavepackets

__all__ = ["LinearCombinationOfWPs"]


class LinearCombinationOfWPs(LinearCombinationOfWavepackets):
    r"""This class represents linear combinations
    of general but compatible wavepackets of any kind.
    """

    def __init__(self, dimension, number_components, number_packets=0):
        r"""Initialize a new linear combination of general wavepackets. This
        object represents :math:`\Upsilon := \sum_{j=0}^{J-1} c_j \Psi_j`.
        All :math:`J` wavepackets :math:`\Psi_j` have the same number :math:`N`
        components and are defined in the :math:`D` dimensional space.

        :param dimension: The space dimension :math:`D` the packets have.
        :param ncomponents: The number :math:`N` of components the packets have.
        :return: An instance of :py:class:`LinearCombinationOfWPs`.
        """
        self._dimension = dimension
        self._number_components = number_components
        self._packets = []
        self._packet_ids = []
        self._number_packets = number_packets
        self._coefficients = zeros((number_packets,1), dtype=complexfloating)

        # TODO: Handle multi-component packets
        assert number_components == 1


    def __str__(self):
        r"""
        :return: A string describing the linear combination of general
                 wavepackets :math:`\Upsilon = \sum_{j=0}^J c_j \Psi_j`.
        """
        s = ("Linear combination of "+str(self._number_packets)+" general wavepackets, each with "
            +str(self._number_components)+" component(s) in "+str(self._dimension)+" space dimension(s)\n")
        return s


    def get_description(self):
        r"""Return a description of this linear combination object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "LinearCombinationOfWPs"
        d["dimension"] = self._dimension
        d["ncomponents"] = self._number_components
        return d


    def clone(self, keepid=False):
        # Parameters of this packet
        params = self.get_description()
        # Create a new linear combination
        # TODO: Consider using the block factory
        other = LinearCombinationOfWPs(params["dimension"],
                                       params["ncomponents"])
        newpackets = [ wp.clone(keepid=keepid) for wp in self.get_wavepackets() ]
        other.add_wavepackets(newpackets, self.get_coefficients())
        return other


    def add_wavepacket(self, packet, coefficient=1.0):
        r"""Add a new wavepacket to the linear combination.

        :param packet: The new wavepacket :math:`\Psi_j` to add.
        :type packet: A :py:class:`Wavepacket` subclass instance.
        :param coefficient: The corresponding coefficient :math:`c_j`, default is 1.0.
        """
        if not packet.get_dimension() == self._dimension:
            raise ValueError("Number of dimensions does not match.")
        if not packet.get_number_components() == self._number_components:
            raise ValueError("Number of components does not match.")
        # Note: we do not test that the varepsilon parameter matches.
        pid = packet.get_id()
        if pid in self._packet_ids:
            raise ValueError("Adding packet failed due to duplicate id.")

        self._packets.append(packet)
        self._number_packets = self._number_packets + 1
        self._packet_ids.append(pid)
        self._coefficients = vstack([self._coefficients, atleast_2d(coefficient)])


    def add_wavepackets(self, packetlist, coefficients=None):
        r"""Add a list of new wavepackets to the linear combination.

        :param packetlist: A list of new wavepackets :math:`\{\Psi_j\}`.
        :type packetlist: A list of :py:class:`Wavepacket` subclass instances.
        :param coefficients: The corresponding coefficient vector :math:`c`, default
                             is a vector of all 1.0.
        """
        if coefficients is not None and len(packetlist) != coefficients.size:
            raise ValueError("Differently many packets and coefficients given.")

        pids = []
        for packet in packetlist:
            if not packet.get_dimension() == self._dimension:
                raise ValueError("Number of dimensions does not match.")
            if not packet.get_number_components() == self._number_components:
                raise ValueError("Number of components does not match.")
            # Note: we do not test that the varepsilon parameter matches.
            pid = packet.get_id()
            pids.append(pid)
            if pid in self._packet_ids:
                raise ValueError("Adding packet failed due to duplicate id.")

        if coefficients is None:
            coefficients = ones((len(packetlist),1))

        self._packets.extend(packetlist)
        self._number_packets = self._number_packets + len(packetlist)
        self._packet_ids.extend(pids)
        self._coefficients = vstack([self._coefficients, atleast_2d(coefficients).reshape((-1,1))])


    def remove_wavepacket(self, index):
        r"""Remove a wavepacket :math:`\Psi_j` from the linear combination.

        :param index: The index :math:`0 \leq j < J` of the packet to remove.
        """
        self._packets.pop(index)
        self._number_packets = self._number_packets - 1
        self._packet_ids.pop(index)
        self._coefficients = delete(self._coefficients, index).reshape((-1,1))


    def get_wavepacket(self, index):
        r"""Get the wavepacket :math:`\Psi_j` from the linear combination.

        :param index: The index :math:`0 \leq j < J` of the packet to retrieve.
        :return: The wavepacket :math:`\Psi_j`.
        :type: A :py:class:`Wavepacket` subclass instance.
        """
        return self._packets[index]


    def get_wavepackets(self):
        r"""Get a list of all wavepackets :math:`\Psi_j` in the linear combination.

        :return: A list of all wavepackets :math:`\Psi_j`.
        :type: A list of :py:class:`Wavepacket` subclass instances.
        """
        return tuple(self._packets)


    def set_wavepackets(self, packetlist):
        r"""Set the list :math:`\{\Psi_j\}_j` of new wavepackets.

        :param packetlist: A list of new wavepackets :math:`\Psi_j`.
        :type packetlist: A list of :py:class:`Wavepacket` subclass instances.
        """
        if not len(packetlist) == self._number_packets:
            raise ValueError("Wrong number of new packets.")

        self._packets = packetlist[:]


    def get_coefficient(self, index):
        r"""Get the coefficient :math:`c_j` of the wavepacket :math:`\Psi_j`.

        :param index: The index :math:`0 \leq j < J` of the coefficient to retrieve.
        :return: The coefficient :math:`c_j`.
        """
        return self._coefficients[index]


    def set_coefficient(self, index, coefficient):
        r"""Set the coefficient :math:`c_j` of the wavepacket :math:`\Psi_j`.

        :param index: The index :math:`0 \leq j < J` of the coefficient to retrieve.
        :param coefficient: The coefficient :math:`c_j`.
        """
        self._coefficients[index] = coefficient


    def get_coefficients(self):
        r"""Get the vector with all coefficients :math:`c_j` of all wavepackets :math:`\Psi_j`.

        :return: The vector :math:`c` of all coefficients :math:`c_j`. The vector is of
                 shape :math:`(J, 1)`.
        :type: An :py:class:`ndarray`
        """
        return self._coefficients.copy()


    def set_coefficients(self, coefficients):
        r"""Update all the coefficients :math:`c` of :math:`\Upsilon`.

        :param coefficients: The vector :math:`c`.
        :type coefficients: An :py:class:`ndarray`
        """
        if not coefficients.size == self._number_packets:
            raise ValueError("Wrong number of new coefficients.")

        self._coefficients = coefficients.copy().reshape((-1,1))


    def evaluate_at(self, grid, component=None):
        r"""Evaluate the linear combination of wavepackets :math:`\Upsilon` at
        the given nodes :math:`\gamma`.

        :param grid: The grid :math:`\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:meth:`get_nodes` method.
        :param component: The index :math:`i` of a single component to evaluate.
                          (Defaults to ``None`` for evaluating all components.)
        :return: A list of arrays or a single array containing the values of the
                 :math:`\Phi_i` at the nodes :math:`\gamma`.
        """
        if self._number_packets == 0:
            raise ValueError("No packets in the linear combination to evaluate.")

        # Split one off to get the result array shape
        if self._number_packets > 0:
            vals = self._packets[0].evaluate_at(grid, component=component, prefactor=True)
            if component is None:
                result = [ self._coefficients[0] * val for val in vals ]
            else:
                result = self._coefficients[0] * vals

        for index, packet in enumerate(self._packets[1:]):
            vals = packet.evaluate_at(grid, component=component, prefactor=True)
            if component is None:
                result = [ res + self._coefficients[index+1] * val for res, val in zip(result, vals) ]
            else:
                result = result + self._coefficients[index+1] * vals

        return result
