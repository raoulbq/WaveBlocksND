"""The WaveBlocks Project

This file contains code for the delegation of the evaluation of inhomogeneous (or mixing)
inner products of two wavepackets. The class defined here can compute brakets, inner products
and expectation values and the matrix elements of an arbitrary operator.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, sum, cumsum

from InnerProduct import InnerProduct

__all__ = ["InhomogeneousInnerProduct"]


class InhomogeneousInnerProduct(InnerProduct):

    def __init__(self, quad=None):
        # Pure convenience to allow setting of quadrature instance in constructor
        if quad is not None:
            self.set_quadrature(quad)
        else:
            self._quad = None


    def __str__(self):
        return "Inhomogeneous inner product computed by " + str(self._quad)


    def get_description(self):
        r"""Return a description of this inner product object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "InhomogeneousInnerProduct"
        d["delegate"] = self._quad.get_description()
        return d


    def quadrature(self, pacbra, packet=None, operator=None, summed=False, component=None, diag_component=None, diagonal=False, eval_at_once=False):
        r"""Delegates the evaluation of :math:`\langle\Psi|f|\Psi^\prime\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param pacbra: The wavepacket :math:`\Psi` from the bra with :math:`N` components.
        :param packet: The wavepacket :math:`\Psi^\prime` from the ket with :math:`N^\prime` components.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :param summed: Whether to sum up the individual integrals :math:`\langle\Phi_i|f_{i,j}|\Phi^\prime_j\rangle`.
        :type summed: Boolean, default is ``False``.
        :param component: Request only the i-th component of the result. Remember that :math:`i \in [0, N \cdot N^\prime-1]`.
        :param diag_component: Request only the i-th component from the diagonal entries, here :math:`i \in [0, N^\prime-1]`.
                               Note that ``component`` takes precedence over ``diag_component`` if both are supplied. (Which is discouraged)
        :param diagonal: Only return the diagonal elements :math:`\langle\Phi_i|f_{i,i}|\Phi^\prime_i\rangle`.
                         This is useful for diagonal operators :math:`f`.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        :return: The value of the braket :math:`\langle\Psi|f|\Psi^\prime\rangle`. This is either a scalar value or
                 a list of :math:`N \cdot N^\prime` scalar elements depending on the value of ``summed``.
        """
        # Allow to ommit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        # TODO: Consider adding 'is_diagonal' flag to make computations cheaper if we know the operator is diagonal
        # TODO: Should raise Exceptions if pacbra and packet are incompatible wrt N, K etc

        self._quad.initialize_packet(pacbra, packet)
        self._quad.initialize_operator(operator, eval_at_once=eval_at_once)

        # Packets can have different number of components
        Nbra = pacbra.get_number_components()
        Nket = packet.get_number_components()

        # Avoid unnecessary computations of other components
        if component is not None:
            rows = [ component // Nket ]
            cols = [ component % Nket ]
        elif diag_component is not None:
            rows = [ diag_component ]
            cols = [ diag_component ]
        else:
            rows = xrange(Nbra)
            cols = xrange(Nket)

        self._quad.prepare(rows, cols)

        # Compute the quadrature
        result = []

        for row in rows:
            for col in cols:
                I = self._quad.perform_quadrature(row, col)
                result.append(I)

        if summed is True:
            result = sum(result)
        elif component is not None or diag_component is not None:
            # Do not return a list for quadrature of specific single components
            result = result[0]
        elif diagonal is True:
            # Only keep the diagonal elements
            res = []
            i = 1
            while i <= Nbra*Nket:
                res.append(result[i-1])
                i = i + 2*i
            result = res

        return result


    def build_matrix(self, pacbra, packet=None, operator=None, eval_at_once=False):
        r"""Delegates the computation of the matrix elements :math:`\langle\Psi|f|\Psi^\prime\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        The matrix is computed without including the coefficients :math:`c^i_k`.

        :param pacbra: The wavepacket :math:`\Psi` from the bra with :math:`N` components.
        :param packet: The wavepacket :math:`\Psi^\prime` from the ket with :math:`N^\prime` components.
        :param operator: A matrix-valued function :math:`f(q, x): \mathbb{R} \times \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        :return: A matrix of size :math:`\sum_i^N |\mathfrak{K}_i| \times \sum_j^{N^\prime} |\mathfrak{K}^\prime_j|`.
        """
        # Allow to ommit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        # TODO: Consider adding 'is_diagonal' flag to make computations cheaper if we know the operator is diagonal
        # TODO: Should raise Exceptions if pacbra and packet are incompatible wrt N, K etc

        self._quad.initialize_packet(pacbra, packet)
        self._quad.initialize_operator(operator, matrix=True, eval_at_once=eval_at_once)

        # Packets can have different number of components
        Nbra = pacbra.get_number_components()
        Nket = packet.get_number_components()
        # Packets can also have different basis size
        Kbra = [ bs.get_basis_size() for bs in pacbra.get_basis_shapes() ]
        Kket = [ bs.get_basis_size() for bs in packet.get_basis_shapes() ]
        # The partition scheme of the block vectors and block matrix
        partitionb = [0] + list(cumsum(Kbra))
        partitionk = [0] + list(cumsum(Kket))

        self._quad.prepare(range(Nbra), range(Nket))

        result = zeros((sum(Kbra),sum(Kket)), dtype=complexfloating)

        for row in xrange(Nbra):
            for col in xrange(Nket):
                M = self._quad.perform_build_matrix(row, col)
                # Put the result into the global storage
                result[partitionb[row]:partitionb[row+1], partitionk[col]:partitionk[col+1]] = M

        return result
