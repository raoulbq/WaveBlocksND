"""The WaveBlocks Project

This file contains the interface for general sparsity oracles.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["SparsityOracle"]


class SparsityOracle(object):
    r"""This class defines the interface for general sparsity oracles.
    """

    def __init__(self):
        r"""Initialize an oracle for estimating if a specific overlap integral is zero.

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'SparsityOracle' is an abstract interface.")


    def is_not_zero(self, pacbra, packet):
        r"""Try to estimate if the overlap integral between the two
        given packets is zero or at least negligible.

        :param pacbra: The packet that is used for the 'bra' part.
        :param packet: The packet that is used for the 'ket' part.
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        # TODO: Consider more general API and call signature
        raise NotImplementedError("'SparsityOracle' is an abstract interface.")
