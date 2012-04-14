"""The WaveBlocks Project

This file contains the basic interface for general wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""


class Wavepacket(object):
    r"""This class is an abstract interface to wavepackets in general.
    """

    def __init__(self, parameters):
        r"""Initialize a wavepacket object that represents :math:`\Psi`. A wavepacket
        has :math:`N` components :math:`\Phi_i` with :math:`i \in 1 \ldots N`. Each
        component is defined over :math:`D` dimensional space.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def __str__(self):
        r""":return: A string describing the wavepacket.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def clone(self):
        r"""Clone the wavepacket. Return a new copy of the wavepacket and
        make sure that all references between the two wavepackets get broken.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def gen_id(self):
        r"""Generate an (unique) ID per wavepacket instance.
        """
        # TODO: Better id generating function! Maybe use UUIDs?
        self._id = id(self)


    def get_id(self):
        r"""Return the packet ID of this wavepacket instance.
        The ID may be used for storing packets in associative lists.

        :return: The ID of the current instance.
        """
        if not hasattr(self, "_id"):
            self.gen_id()

        return self._id


    def set_id(self, anid):
        r"""Manually set a new ID for the current wavepacket instance.

        :param anid: The new ID.
        :type anid: int
        """
        # Currently we force integers as ID s. However in general IDs
        # could be of any object type, as long as we can avoid name clashes.
        assert(type(anid) is int)
        self._id = anid


    def get_dimension(self):
        r""":return: The space dimension :math:`D` of the wavepacket :math:`\Psi`.
        """
        return self._dimension


    def get_number_components(self):
        r""":return: The number :math:`N` of components the wavepacket :math:`\Psi` has.
        """
        return self._number_components
