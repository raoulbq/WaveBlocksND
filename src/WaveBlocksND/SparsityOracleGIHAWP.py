"""The WaveBlocks Project

Thie file contains the code for a sparsity oracle looking
at Gaussian integral of both packets.


@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, ones, abs, sqrt, dot, atleast_1d
from numpy.linalg import norm

from SparsityOracle import SparsityOracle
from GaussianIntegral import GaussianIntegral

__all__ = ["SparsityOracleGIHAWP"]


class SparsityOracleGIHAWP(SparsityOracle):
    r"""This class implements an oracle by looking at Gaussian integrals.
    """

    def __init__(self, threshold=1e-8):
        r"""Initialize an oracle for estimating if a specific overlap integral
        :math:`\langle \Psi_k | O | \Psi_l \rangle` is approximately zero. The
        oracle works by approximating :math:`\langle \Psi_k | \Psi_l \rangle`
        with a Gaussian integral. If

        .. math::
            \langle \Psi_k | \Psi_l \rangle \approx \langle \Psi_k^G | \Psi_l^G \rangle \leq \tau

        the value :math:`\langle \Psi_k | O | \Psi_l \rangle` is considered
        to be zero. Of course this may fail depending on the form of the
        operator :math:`O` or the basis shape :math:`\mathfrak{K}`.

        .. warning::
            This code is highly experimental.

        :param threshold: The threshold :math:`\tau` in the Gaussian integral criterion.
                          The default value of :math:`10^{-8}` should be reasonable in most cases.
        """
        self._threshols = threshold
        self._gi = GaussianIntegral()


    def is_not_zero(self, pacbra, packet, component=0):
        r"""Try to estimate if the overlap integral :math:`\langle \Psi_k | \Psi_l \rangle`
        is zero or at least negligible.

        :param pacbra: The packet :math:`\Psi_k` that is used for the 'bra' part.
        :param packet: The packet :math:`\Psi_l` that is used for the 'ket' part.
        :return: ``True`` or ``False`` whether the inner product is negligible.
        """
        Q = self._gi.preform_quadratur(pacbra, packet)
        return abs(abs(Q) > self._threshold)
