"""The WaveBlocks Project

Thie file contains the code for a sparsity oracle looking
at the phase space distance between both packets.


@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, ones, abs, sqrt, dot, atleast_1d
from numpy.linalg import norm

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
            \|q_k - q_l\| \leq \alpha ( \|\sigma^q_k\| + \|\sigma^q_l\| )

        and

        .. math::
            \|p_k - p_l\| \leq \alpha ( \|\sigma^p_k\| + \|\sigma^p_l\| )

        :param factor: The factor :math:`\alpha` in the phase space distance.
                       The default value of 1.5 should be reasonable in most cases.
        """
        self._factor = factor
        self._bias_bra = False
        self._bias_ket = False


    def is_not_zero(self, pacbra, packet, component=0):
        r"""Try to estimate if the overlap integral :math:`\langle \Psi_k | \Psi_l \rangle`
        is zero or at least negligible.

        :param pacbra: The packet :math:`\Psi_k` that is used for the 'bra' part.
        :param packet: The packet :math:`\Psi_l` that is used for the 'ket' part.
        :return: ``True`` or ``False`` whether the inner product is negligible.
        """
        eps = packet.get_eps()
        qbra, Qbra, pbra, Pbra = pacbra.get_parameters(key=("q", "Q", "p", "P"))
        qket, Qket, pket, Pket = packet.get_parameters(key=("q", "Q", "p", "P"))

        # First strategy
        # TODO: Can there be a 'wrong' largest index in case there are more than one?
        #kbra = array(pacbra.get_basis_shapes(component=component).find_largest_index())
        #kket = array(packet.get_basis_shapes(component=component).find_largest_index())

        # Second strategy
        #Kbra = pacbra.get_basis_shapes(component=component)
        #indices = array([ node for node in Kbra.get_node_iterator() ])
        #kbra = indices.max(axis=0)
        #Kket = packet.get_basis_shapes(component=component)
        #indices = array([ node for node in Kket.get_node_iterator() ])
        #kket = indices.max(axis=0)

        # Third strategy
        kbra = array(pacbra.get_basis_shapes(component=component).find_largest_index())
        kket = array(packet.get_basis_shapes(component=component).find_largest_index())

        # Bias for small k
        if self._bias_bra:
            if norm(kbra) < self._bra_min_k_norm:
                kbra = self._bra_min_k
        if self._bias_ket:
            if norm(kket) < self._ket_min_k_norm:
                kket = self._ket_min_k

        D = pacbra.get_dimension()
        kbra = norm(kbra) / sqrt(D) * ones(D)
        kket = norm(kket) / sqrt(D) * ones(D)

        # Compute second moments
        sigqbra = eps/sqrt(2.0) * sqrt(dot(abs(Qbra)**2, 2*kbra+1))
        sigqket = eps/sqrt(2.0) * sqrt(dot(abs(Qket)**2, 2*kket+1))

        sigpbra = eps/sqrt(2.0) * sqrt(dot(abs(Pbra)**2, 2*kbra+1))
        sigpket = eps/sqrt(2.0) * sqrt(dot(abs(Pket)**2, 2*kket+1))

        return (norm(qbra - qket) <= self._factor * (norm(sigqbra)+norm(sigqket)) and
                norm(pbra - pket) <= self._factor * (norm(sigpbra)+norm(sigpket)))


    def bias(self, bramink=None, ketmink=None):
        r"""Bias the sparsity oracle. The oracle tends to underestimate
        the wavepacket spread for small basis shapes :math:`\mathfrak{K}`
        and therefore small maximal indices :math:`k \in \mathfrak{K}`.
        This method allows to set a minimal :math:`k` for both the bra
        and ket independently.

        :param bramink: Minimal :math:`k` for :math:`\langle \Psi |`
        :param ketmink: Minimal :math:`k` for :math:`| \Psi \rangle`
        """
        if bramink is not None:
            self._bra_min_k = bramink
            self._bra_min_k_norm = norm(atleast_1d(bramink))
            self._bias_bra = True

        if ketmink is not None:
            self._ket_min_k = ketmink
            self._ket_min_k_norm = norm(atleast_1d(ketmink))
            self._bias_ket = True
