"""The WaveBlocks Project

This file contains code for the delegation of the evaluation of inner products
of two linear combinations of wavepackets. The class defined here can compute
brakets, inner products and expectation values and the matrix elements of an
arbitrary operator.

@author: R. Bourquin
@copyright: Copyright (C) 2013, 2014 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, conjugate, transpose, dot, cumsum, sum, reshape, array, repeat

from InnerProduct import InnerProduct
from InnerProductCompatibility import InnerProductCompatibility

__all__ = ["HomogeneousInnerProductLCWP"]


class HomogeneousInnerProductLCWP(InnerProduct, InnerProductCompatibility):

    def __init__(self, delegate=None, oracle=None):
        r"""
        Note that although this class computes the homogeneous inner product
        :math:`\langle\Upsilon|f|\Upsilon\rangle` of a single linear combination
        :math:`\Upsilon` with itself, the delegate inner product class used
        for computing :math:`\langle\Psi|f|\Psi^\prime\rangle` has still to
        be of *inhomogeneous* type.

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
        return "Homogeneous inner product of linear combinations computed by " + str(self._delegate)


    def get_description(self):
        r"""Return a description of this inner product object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "HomogeneousInnerProductLCWP"
        d["delegate"] = self._delegate.get_description()
        return d


    def get_kind(self):
        return ("homogeneous",)


    def require_kind(self):
        return ("homogeneous",)


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


    def quadrature(self, lcket, operator=None, component=None, eval_at_once=False):
        r"""Delegates the evaluation of :math:`\langle\Upsilon|f|\Upsilon\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param lcket: The linear combination :math:`\Upsilon` with :math:`J` summands :math:`\Psi_j`.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N}`.
        :param component: The index :math:`i` of the component :math:`\Phi_j` of :math:`\Psi_j`. If set only those
                          components will be taken into account for the computation.
        :type component: Integer or ``None``, default is ``None``.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        :return: The value of :math:`\langle\Upsilon|f|\Upsilon\rangle`.
        :type: An :py:class:`ndarray`.
        """
        # Packets can in principle have different number of components
        if component is not None:
            N = array([1] * lcket.get_number_packets())
        else:
            N = array([ wp.get_number_components() for wp in lcket.get_wavepackets() ])

        M = self.build_matrix(lcket, operator=operator, component=component, eval_at_once=eval_at_once)

        c = lcket.get_coefficients()
        c = repeat(c, N)
        return dot(conjugate(transpose(c)), dot(M, c))


    def build_matrix(self, lcket, operator=None, component=None, eval_at_once=False):
        r"""Delegates the computation of the matrix elements of :math:`\langle\Upsilon|f|\Upsilon\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        The matrix is computed without including the coefficients :math:`c_j`.

        :param lcket: The linear combination :math:`\Upsilon` with :math:`J` summands :math:`\Psi_j`.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N}`.
        :param component: The index :math:`i` of the component :math:`\Phi_j` of :math:`\Psi_j`. If set only those
                          components will be taken into account for the computation.
        :type component: Integer or ``None``, default is ``None``.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        :return: A matrix of size :math:`\sum_{j\in J} N_j \times \sum_{j\in J} N_{j}`.
        :type: An :py:class:`ndarray`.
        """
        packets = lcket.get_wavepackets()

        # Packets can in principle have different number of components
        if component is not None:
            N = [1] * lcket.get_number_packets()
        else:
            N = [ wp.get_number_components() for wp in packets ]

        # The partition scheme of the block vectors and block matrix
        partition = [0] + list(cumsum(N))

        result = zeros((sum(N), sum(N)), dtype=complexfloating)

        # Elements below the diagonal
        if self._obey_oracle:
            for row, pacbra in enumerate(packets):
                for col, packet in enumerate(packets[:row]):
                    if self._oracle.is_not_zero(pacbra, packet, component=component):
                        Q = self._delegate.quadrature(pacbra, packet, operator=operator, diag_component=component, eval_at_once=eval_at_once)
                        Q = reshape(Q, (N[row], N[col]))
                        # Put the result into the global storage
                        result[partition[row]:partition[row+1], partition[col]:partition[col+1]] = Q
        else:
            for row, pacbra in enumerate(packets):
                for col, packet in enumerate(packets[:row]):
                    Q = self._delegate.quadrature(pacbra, packet, operator=operator, diag_component=component, eval_at_once=eval_at_once)
                    Q = reshape(Q, (N[row], N[col]))
                    # Put the result into the global storage
                    result[partition[row]:partition[row+1], partition[col]:partition[col+1]] = Q

        result = result + conjugate(transpose(result))

        # Diagonal Elements
        for d, packet in enumerate(packets):
            Q = self._delegate.quadrature(packet, packet, operator=operator, diag_component=component, eval_at_once=eval_at_once)
            Q = reshape(Q, (N[d], N[d]))
            # Put the result into the global storage
            result[partition[d]:partition[d+1], partition[d]:partition[d+1]] = Q

        return result
