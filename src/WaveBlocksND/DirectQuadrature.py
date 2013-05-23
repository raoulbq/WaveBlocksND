"""The WaveBlocks Project

This file contains an intermediate API definition for evaluating inner products
and matrix elements by using standard quadrature rules.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze, transpose, conjugate, dot

from Quadrature import Quadrature

__all__ = ["DirectQuadrature"]


class DirectQuadrature(Quadrature):
    r"""
    """

    def get_description(self):
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        raise NotImplementedError("'DirectQuadrature' is an abstract interface.")


    def perform_quadrature(self, row, col):
        r"""Evaluates by numerical steepest descent the integral
        :math:`\langle \Phi_i | f | \Phi^\prime_j \rangle` for a polynomial
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A single complex floating point number.
        """
        if not self._QR.get_dimension() == self._packet.get_dimension():
            raise ValueError("Quadrature dimension does not match the wavepacket dimension")

        M = self.do_quadrature(row, col)
        # Include the coefficients as c^H M c
        cbra = self._pacbra.get_coefficients(component=row)
        cket = self._packet.get_coefficients(component=col)
        I = squeeze(dot(transpose(conjugate(cbra)), dot(M, cket)))
        return I


    def perform_build_matrix(self, row, col):
        r"""Computes by standard quadrature the matrix elements
        :math:`\langle\Phi_i | f |\Phi^\prime_j\rangle` for a general function
        :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}^\prime_j|`.
        """
        if not self._QR.get_dimension() == self._packet.get_dimension():
            raise ValueError("Quadrature dimension does not match the wavepacket dimension")

        M = self.do_quadrature(row, col)
        return M
