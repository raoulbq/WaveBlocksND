"""The WaveBlocks Project

This file contains the basic interface for general wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import time
import hashlib

__all__ = ["Wavepacket"]


class Wavepacket(object):
    r"""This class is an abstract interface to wavepackets in general.
    """

    def __init__(self, parameters):
        r"""Initialize a wavepacket object that represents :math:`\Psi`. A wavepacket
        has :math:`N` components :math:`\Phi_i` with :math:`i \in 1 \ldots N`. Each
        component is defined over :math:`D` dimensional space.

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def __str__(self):
        r""":return: A string describing the wavepacket.

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def clone(self):
        r"""Clone the wavepacket. Return a new copy of the wavepacket and
        make sure that all references between the two wavepackets get broken.

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def gen_id(self):
        r"""Generate an (unique) ID per wavepacket instance.
        """
        # Generate the packet ID from the currect time as well as
        # memory location and take the md5 hash. The hash can be
        # assumed collision free for this usecase. The 'id' part
        # assures that different instances get different IDs even
        # in case the timer resolution is too small. The 'time' part
        # assures that in case one instance gets deleted and another
        # created at the same location, their IDs still differ.
        self._id = hashlib.md5(str(id(self))+str(time.time())).hexdigest()


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
