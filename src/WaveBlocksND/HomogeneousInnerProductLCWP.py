"""The WaveBlocks Project

This file contains code for the delegation of the evaluation of inner products
of two linear combinations of wavepackets. The class defined here can compute
brakets, inner products and expectation values and the matrix elements of an
arbitrary operator.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, conjugate, transpose, dot

from InnerProduct import InnerProduct

__all__ = ["HomogeneousInnerProductLCWP"]


class HomogeneousInnerProductLCWP(InnerProduct):

    def __init__(self, ip=None, oracle=None):
        r"""
        Note that although this class computes the homogeneous inner product
        :math:`\langle\Upsilon|f|\Upsilon\rangle` of a single linear combination
        :math:`\Upsilon` with itself, the delegate inner product class used
        for computing :math:`\langle\Psi|f|\Psi^\prime\rangle` has still to
        be of *inhomogeneous* type.

        :param ip: The delegate inner product.
        :type ip: A :py:class:`InnerProduct` subclass instance.
        :param oracle: The sparsity oracle to use. If the variable is ``None``
                       no oracle is used and all integrals are computed.
        """
        # Pure convenience to allow setting of quadrature instance in constructor
        if ip is not None:
            self.set_quadrature(ip)
        else:
            self._quad = None

        self.set_oracle(oracle)


    def __str__(self):
        return "Homogeneous inner product of linear combinations computed by " + str(self._quad)


    def get_description(self):
        r"""Return a description of this inner product object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "HomogeneousInnerProductLCWP"
        d["delegate"] = self._quad.get_description()
        return d


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


    def quadrature(self, lcket, operator=None, component=None):
        r"""Delegates the evaluation of :math:`\langle\Upsilon|f|\Upsilon\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param lcket: The linear combination :math:`\Upsilon` with :math:`J` summands :math:`\Psi_j`.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N}`.
        :return: The value of :math:`\langle\Upsilon|f|\Upsilon\rangle`.
        :type: An :py:class:`ndarray`.
        """
        J = lcket.get_number_packets()
        packets = lcket.get_wavepackets()

        M = zeros((J, J), dtype=complexfloating)

        # Elements below the diagonal
        if self._obey_oracle:
            for row, pacbra in enumerate(packets):
                for col, packet in enumerate(packets[:row]):
                    if self._oracle.is_not_zero(pacbra, packet):
                        # TODO: Handle multi-component packets
                        M[row, col] = self._quad.quadrature(pacbra, packet, operator=operator, component=0)
        else:
            for row, pacbra in enumerate(packets):
                for col, packet in enumerate(packets[:row]):
                    # TODO: Handle multi-component packets
                    M[row, col] = self._quad.quadrature(pacbra, packet, operator=operator, component=0)

        M = M + conjugate(transpose(M))

        # Diagonal Elements
        for d, packet in enumerate(packets):
            # TODO: Handle multi-component packets
            M[d, d] = self._quad.quadrature(packet, packet, operator=operator, component=0)

        c = lcket.get_coefficients()

        return dot(conjugate(transpose(c)), dot(M, c))


    def build_matrix(self, lcket, operator=None):
        r"""Delegates the computation of the matrix elements of :math:`\langle\Upsilon|f|\Upsilon\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        The matrix is computed without including the coefficients :math:`c_j`.

        :param lcket: The wavepacket :math:`\Upsilon` with :math:`J` summands :math:`\Psi_j`.
        :param operator: A matrix-valued function :math:`f(q, x): \mathbb{R} \times \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N}`.
        :return: A matrix of shape :math:`J \times J`.
        :type: An :py:class:`ndarray`.
        """
        J = lcket.get_number_packets()
        packets = lcket.get_wavepackets()

        M = zeros((J, J), dtype=complexfloating)

        # Elements below the diagonal
        if self._obey_oracle:
            for row, pacbra in enumerate(packets):
                for col, packet in enumerate(packets[:row]):
                    if self._oracle.is_not_zero(pacbra, packet):
                        # TODO: Handle multi-component packets
                        M[row, col] = self._quad.quadrature(pacbra, packet, operator=operator, component=0)
        else:
            for row, pacbra in enumerate(packets):
                for col, packet in enumerate(packets[:row]):
                    # TODO: Handle multi-component packets
                    M[row, col] = self._quad.quadrature(pacbra, packet, operator=operator, component=0)

        M = M + conjugate(transpose(M))

        # Diagonal Elements
        for d, packet in enumerate(packets):
            # TODO: Handle multi-component packets
            M[d, d] = self._quad.quadrature(packet, packet, operator=operator, component=0)

        return M
