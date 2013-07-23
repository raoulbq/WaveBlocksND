"""The WaveBlocks Project

Thie file contains the code for a sparsity oracle looking
at the phase space distance between both packets.


@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import abs, sqrt

from SparsityOracle import SparsityOracle

__all__ = ["SparsityOraclePSHAWP"]


class SparsityOraclePSHAWP(SparsityOracle):
    r"""This class implements an oracle by looking at a phase space distance.
    """

    def __init__(self, factor=1.5):
        r"""Initialize an oracle for estimating if a specific overlap integral
        :math:`\langle \Psi_k | \Psi_l \rangle` is approximately zero. The
        oracle works by computing first and second moments :math:`\mu` and :math:`\sigma`
        of the highest order function :math:`\phi_i` of both wavepackts :math:`\Psi_k`
        and :math:`\Psi_l` for both, position and momentum. Then we compute the
        estimators:

        .. math::
            |q_k - q_l| \leq \alpha |\sigma^q_k + \sigma^q_l|

        and

        .. math::
            |p_k - p_l| \leq \alpha |\sigma^p_k + \sigma^p_l|

        Note that the current implementation is for one-dimensional packets only.

        :param factor: The factor :math:`\alpha` in the phase space distance.
                       The default value of 1.5 should be reasonable in most cases.
        """
        self._factor = factor


    def is_not_zero(self, pacbra, packet, component=0):
        r"""Try to estimate if the overlap integral :math:`\langle \Psi_k | \Psi_l \rangle`
        is zero or at least negligible.

        :param pacbra: The packet :math:`\Psi_k` that is used for the 'bra' part.
        :param packet: The packet :math:`\Psi_l` that is used for the 'ket' part.
        :return: ``True`` or ``False`` whether the inner product is negligible.
        """
        # TODO: Generalize the oracle to higher dimensions
        assert pacbra.get_dimension() == 1
        assert packet.get_dimension() == 1

        qbra, Qbra, pbra, Pbra = pacbra.get_parameters(key=("q", "Q", "p", "P"))
        qket, Qket, pket, Pket = packet.get_parameters(key=("q", "Q", "p", "P"))

        Kbra = pacbra.get_basis_shapes(component=component).get_basis_size() - 1
        Kket = packet.get_basis_shapes(component=component).get_basis_size() - 1

        eps = packet.get_eps()

        sigqbra = eps*abs(Qbra)/sqrt(2.0) * sqrt(2*Kbra +1)
        sigqket = eps*abs(Qket)/sqrt(2.0) * sqrt(2*Kket +1)

        sigpbra = eps*abs(Pbra)/sqrt(2.0) * sqrt(2*Kbra +1)
        sigpket = eps*abs(Pket)/sqrt(2.0) * sqrt(2*Kket +1)

        return (abs(qbra - qket) <= self._factor * (sigqbra+sigqket) and
                abs(pbra - pket) <= self._factor * (sigpbra+sigpket))
