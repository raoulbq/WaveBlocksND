"""The WaveBlocks Project

This file contains code for the delegation of the evaluation ofinner products
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

    def __init__(self, ip=None):
        r"""
        Note that although this class computes the homogeneous inner product
        :math:`\langle\Upsilon|f|\Upsilon\rangle` of a single linear combination
        :math:`\Upsilon` with itself, the delegate inner product class used
        for computing :math:`\langle\Psi|f|\Psi^\prime\rangle` has still to
        be of *inhomogeneous* type.
        """
        # Pure convenience to allow setting of quadrature instance in constructor
        if ip is not None:
            self.set_quadrature(ip)
        else:
            self._quad = None


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


    def quadrature(self, lcket, operator=None, component=None):
        r"""Delegates the evaluation of :math:`\langle\Upsilon|f|\Upsilon^\prime\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param lcket: The wavepacket :math:`\Upsilon^\prime` from the ket with :math:`J^\prime` summands :math:`\Psi_j^\prime`.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :return: A matrix of size :math:`J \times J^\prime`.
        """
        J = lcket.get_number_packets()
        packets = lcket.get_wavepackets()

        M = zeros((J, J), dtype=complexfloating)

        # Elements below the diagonal
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
        r"""Delegates the computation of the matrix elements :math:`\langle\Upsilon|f|\Upsilon^\prime\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        The matrix is computed without including the coefficients :math:`c_j` and :math:`c_j^\prime`.

        :param lcket: The wavepacket :math:`\Upsilon^\prime` from the ket with :math:`J^\prime` summands :math:`\Psi_j^\prime`.
        :param operator: A matrix-valued function :math:`f(q, x): \mathbb{R} \times \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :return: A matrix of size :math:`J \times J^\prime`.
        """
        J = lcket.get_number_packets()
        packets = lcket.get_wavepackets()

        M = zeros((J, J), dtype=complexfloating)

        # Elements below the diagonal
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
