"""The WaveBlocks Project

This file contains the basic interface for linear
combinations of general but compatible wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["LinearCombinationOfWavepackets"]


class LinearCombinationOfWavepackets(object):
    r"""This class is an abstract interface to linear combinations
    of general but compatible wavepackets.
    """

    def __init__(self, dimension, ncomponents):
        r"""Initialize a new linear combination of wavepackets. This
        object represents :math:`\Upsilon := \sum_{j=0}^{J-1} c_j \Psi_j`.
        All :math:`J` wavepackets :math:`\Psi_j` have the same number :math:`N`
        components and are defined in the :math:`D` dimensional space.

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def __str__(self):
        r"""
        :return: A string describing the linear combination of wavepackets.
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def clone(self):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'clone' not yet implemented.")


    def get_dimension(self):
        r"""
        :return: The space dimension :math:`D` of all the wavepackets :math:`\Psi_j`.
        """
        return self._dimension


    def get_number_components(self):
        r"""
        :return: The number :math:`N` of components all the wavepackets :math:`\Psi_j` have.
        """
        return self._number_components


    def get_number_packets(self):
        r"""
        :return: The number :math:`J` of wavepackets in the linear combination :math:`\Upsilon := \sum_{j=0}^{J-1} c_j \Psi_j`.
        """
        return self._number_packets


    def add_wavepacket(self, packet, coefficient):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def add_wavepackets(self, packets, coefficients):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def remove_wavepacket(self, index):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def get_wavepacket(self, index):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def get_wavepackets(self):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def set_wavepackets(self, packetlist):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def get_coefficient(self, index):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def get_coefficients(self):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def set_coefficients(self, coefficients):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")
