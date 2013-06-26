"""The WaveBlocks Project

This file contains the class which represents linear combinations
of general but compatible Hagedorn wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, atleast_2d, hstack, vstack, delete, conjugate, transpose, dot

from LinearCombinationOfWavepackets import LinearCombinationOfWavepackets

__all__ = ["LinearCombinationOfHAWPs"]


class LinearCombinationOfHAWPs(LinearCombinationOfWavepackets):
    r"""This class represents linear combinations
    of general but compatible Hagedorn wavepackets.
    """

    def __init__(self, dimension, number_components):
        r"""Initialize a new linear combination of Hagedorn wavepackets. This
        object represents :math:`\Upsilon := \sum_{j=0}^{J-1} c_j \Psi_j`.
        All :math:`J` wavepackets :math:`\Psi_j` have the same number :math:`N`
        components and are defined in the :math:`D` dimensional space.

        :param dimension: The space dimension :math:`D` the packets have.
        :param ncomponents: The number :math:`N` of components the packets have.
        :return: An instance of :py:class:`LinearCombinationOfHAWPs`.
        """
        self._dimension = dimension
        self._number_components = number_components
        self._packets = []
        self._number_packets = 0
        self._coefficients = zeros((0,1), dtype=complexfloating)

        # TODO: Handle multi-component packets
        assert number_components == 1


    def __str__(self):
        r""":return: A string describing the linear combination of
        Hagedorn wavepackets :math:`\Upsilon`.
        """
        s = ("Linear combination of "+str(self._number_packets)+" Hagedorn wavepackets, each with "
            +str(self._number_components)+" component(s) in "+str(self._dimension)+" space dimension(s)\n")
        return s


    def get_description(self):
        r"""Return a description of this linear combination object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "LinearCombinationOfHAWPs"
        d["dimension"] = self._dimension
        d["ncomponents"] = self._number_components
        return d


    def add_wavepacket(self, packet, coefficient=1.0):
        r"""Add a new wavepacket to the linear combination.

        :param packet: The new wavepacket :math:`\Psi_j` to add.
        :type packet: A :py:class:`HagedornWavepacket` or :py:class:`HagedornWavepacketInhomogeneous`
        :param coefficient: The corresponding coefficient :math:`c_j`, default is 1.0.
        """
        if not packet.get_dimension() == self._dimension:
            raise ValueError("Number of dimensions does not match.")
        if not packet.get_number_components() == self._number_components:
            raise ValueError("Number of components does not match.")
        # Note: we do not test that the varepsilon parameter matches.

        self._packets.append(packet)
        self._number_packets = self._number_packets + 1
        self._coefficients = vstack([self._coefficients, atleast_2d(coefficient)])


    def remove_wavepacket(self, index):
        r"""Remove a wavepacket :math:`\Psi_j` from the linear combination.

        :param index: The index :math:`0 \leq j < J` of the packet to remove.
        """
        self._packets.pop(index)
        self._number_packets = self._number_packets - 1
        self._coefficients = delete(self._coefficients, index).reshape((-1,1))


    def get_wavepacket(self, index):
        r"""Get the wavepacket :math:`\Psi_j` from the linear combination.

        :param index: The index :math:`0 \leq j < J` of the packet to retrieve.
        """
        return self._packets[index]


    def get_wavepackets(self):
        r"""Get a list of all wavepackets :math:`\Psi_j` in the linear combination.
        """
        return tuple(self._packets)


    def get_coefficient(self, index):
        r"""Get the coefficient :math:`c_j` of the wavepacket :math:`\Psi_j`.

        :param index: The index :math:`0 \leq j < J` of the coefficient to retrieve.
        """
        return self._coefficients[index]


    def get_coefficients(self):
        r"""Get the vector with all coefficients :math:`c_j` of all wavepackets :math:`\Psi_j`.
        """
        return self._coefficients.copy()


    def evaluate_at(self, grid, component=None):
        r"""Evaluate the linear combination of wavepackets :math:`\Upsilon` at
        the given nodes :math:`\gamma`.

        :param grid: The grid :math:`\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:meth:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component to evaluate.
                          (Defaults to ``None`` for evaluating all components.)
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i` at the nodes :math:`\gamma`.
        """
        # Split one off to get the result array shape
        if self._number_packets > 0:
            result = self._packets[0].evaluate_at(grid, component=component, prefactor=True)

        for packet in self._packets[1:]:
            vals = packet.evaluate_at(grid, component=component, prefactor=True)
            if component is None:
                for index, val in enumerate(vals):
                    result[index] = result[index] + val
            else:
                result = result + vals

        return result
