"""The WaveBlocks Project

This file contains code for the delegation of the evaluation of inner products
of two linear combinations of wavepackets. The class defined here can compute
brakets, inner products and expectation values and the matrix elements of an
arbitrary operator.

@author: R. Bourquin
@copyright: Copyright (C) 2013, 2014 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, conjugate, transpose, dot, sum, cumsum, array, repeat, reshape

from InnerProduct import InnerProduct
from InnerProductCompatibility import InnerProductCompatibility

__all__ = ["InhomogeneousInnerProductLCWP"]


class InhomogeneousInnerProductLCWP(InnerProduct, InnerProductCompatibility):

    def __init__(self, delegate=None, oracle=None):
        r"""
        This class computes the inhomogeneous inner product
        :math:`\langle\Upsilon|f|\Upsilon^\prime\rangle` of two linear combinations
        :math:`\Upsilon` and :math:`\Upsilon^\prime`. The delegate inner
        product class used for computing :math:`\langle\Psi|f|\Psi^\prime\rangle`
        has to be of *inhomogeneous* type.

        :param delegate: The delegate inner product.
        :type delegate: A :py:class:`InnerProduct` subclass instance.
        :param oracle: The sparsity oracle to use. If the variable is ``None``
                       no oracle is used and all integrals are computed.

        .. note:: Make sure to use an inhomogeneous inner product here.
        """
        # Pure convenience to allow setting of quadrature instance in constructor
        self.set_delegate(delegate)
        self.set_oracle(oracle)


    def __str__(self):
        return "Inhomogeneous inner product of linear combinations computed by " + str(self._delegate)


    def get_description(self):
        r"""Return a description of this inner product object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "InhomogeneousInnerProductLCWP"
        d["delegate"] = self._delegate.get_description()
        return d


    def get_kind(self):
        return ("homogeneous", "inhomogeneous",)


    def require_kind(self):
        return ("inhomogeneous",)


    def get_oracle(self):
        r"""Return the sparsity oracle in use or ``None``.
        """
        return self._oracle


    def set_oracle(self, new_oracle):
        r"""Set the sparsity oracle.

        :param new_oracle: The new oracle to use. If the variable is ``None``
                           no oracle is used and all integrals are computed.
        """
        if new_oracle is not None:
            self._oracle = new_oracle
            self._obey_oracle = True
        else:
            self._oracle = None
            self._obey_oracle = False


    def quadrature(self, lcbra, lcket=None, operator=None, component=None, eval_at_once=False):
        r"""Delegates the evaluation of :math:`\langle\Upsilon|f|\Upsilon^\prime\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param lcbra: The linear combination :math:`\Upsilon` from the bra with :math:`J` summands :math:`\Psi_j`.
        :param lcket: The linear combination :math:`\Upsilon^\prime` from the ket with :math:`J^\prime` summands :math:`\Psi_j^\prime`.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :param component: The index :math:`i` of the component :math:`\Phi_j` of :math:`\Psi_j`. If set only those
                          components will be taken into account for the computation.
        :type component: Integer or ``None``, default is ``None``.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        :return: The value of :math:`\langle\Upsilon|f|\Upsilon^\prime\rangle`.
        :type: An :py:class:`ndarray`.
        """
        # Allow to omit the ket if it is the same as the bra
        if lcket is None:
            lcket = lcbra

        # Packets can in principle have different number of components
        if component is not None:
            Nbra = array([1] * lcbra.get_number_packets())
            Nket = array([1] * lcket.get_number_packets())
        else:
            Nbra = array([ wp.get_number_components() for wp in lcbra.get_wavepackets() ])
            Nket = array([ wp.get_number_components() for wp in lcket.get_wavepackets() ])

        M = self.build_matrix(lcbra, lcket, operator=operator, component=component, eval_at_once=eval_at_once)

        cbra = lcbra.get_coefficients()
        cket = lcket.get_coefficients()
        cbra = repeat(cbra, Nbra)
        cket = repeat(cket, Nket)
        return dot(conjugate(transpose(cbra)), dot(M, cket))


    def build_matrix(self, lcbra, lcket=None, operator=None, component=None, eval_at_once=False):
        r"""Delegates the computation of the matrix elements of :math:`\langle\Upsilon|f|\Upsilon^\prime\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        The matrix is computed without including the coefficients :math:`c_j` and :math:`c_j^\prime`.

        :param lcbra: The linear combination :math:`\Upsilon` from the bra with :math:`J` summands :math:`\Psi_j`.
        :param lcket: The linear combination :math:`\Upsilon^\prime` from the ket with :math:`J^\prime` summands :math:`\Psi_j^\prime`.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :param component: The index :math:`i` of the component :math:`\Phi_j` of :math:`\Psi_j`. If set only those
                          components will be taken into account for the computation.
        :type component: Integer or ``None``, default is ``None``.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        :return: A matrix of size :math:`\sum_{j\in J} N_j \times \sum_{j^\prime\in J^\prime} N_{j^\prime}`.
        :type: An :py:class:`ndarray`.
        """
        # Allow to omit the ket if it is the same as the bra
        if lcket is None:
            lcket = lcbra

        pacbras = lcbra.get_wavepackets()
        packets = lcket.get_wavepackets()

        # Packets can in principle have different number of components
        if component is not None:
            Nbra = [1] * lcbra.get_number_packets()
            Nket = [1] * lcket.get_number_packets()
        else:
            Nbra = [ wp.get_number_components() for wp in pacbras ]
            Nket = [ wp.get_number_components() for wp in packets ]

        # The partition scheme of the block vectors and block matrix
        partitionb = [0] + list(cumsum(Nbra))
        partitionk = [0] + list(cumsum(Nket))

        result = zeros((sum(Nbra), sum(Nket)), dtype=complexfloating)

        if self._obey_oracle:
            for row, pacbra in enumerate(pacbras):
                for col, packet in enumerate(packets):
                    if self._oracle.is_not_zero(pacbra, packet, component=component):
                        Q = self._delegate.quadrature(pacbra, packet, operator=operator, diag_component=component, eval_at_once=eval_at_once)
                        Q = reshape(Q, (Nbra[row], Nket[col]))
                        # Put the result into the global storage
                        result[partitionb[row]:partitionb[row+1], partitionk[col]:partitionk[col+1]] = reshape(Q, (Nbra[row], Nket[col]))
        else:
            for row, pacbra in enumerate(pacbras):
                for col, packet in enumerate(packets):
                    Q = self._delegate.quadrature(pacbra, packet, operator=operator, diag_component=component, eval_at_once=eval_at_once)
                    Q = reshape(Q, (Nbra[row], Nket[col]))
                    # Put the result into the global storage
                    result[partitionb[row]:partitionb[row+1], partitionk[col]:partitionk[col+1]] = Q

        return result
