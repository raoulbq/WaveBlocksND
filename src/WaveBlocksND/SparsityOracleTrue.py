"""The WaveBlocks Project

This file contains the code for a sparsity oracle
always saying the integrals are non-zero.


@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from SparsityOracle import SparsityOracle

__all__ = ["SparsityOracleTrue"]


class SparsityOracleTrue(SparsityOracle):
    r"""This class implements an oracle always saying
    the integrals are non-zero.
    """

    def __init__(self):
        r"""Initialize an oracle for estimating if a specific overlap integral
        :math:`\langle \Psi_k | \Psi_l \rangle` is approximately zero. This
        oracle always returns ``False``.
        """


    def is_not_zero(self, pacbra, packet, component=0):
        r"""Try to estimate if the overlap integral :math:`\langle \Psi_k | \Psi_l \rangle`
        is zero or at least negligible.

        :param pacbra: The packet :math:`\Psi_k` that is used for the 'bra' part.
        :param packet: The packet :math:`\Psi_l` that is used for the 'ket' part.
        :return: ``True`` independent of any input or condition.
        """
        return True
