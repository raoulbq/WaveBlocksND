"""The WaveBlocks Project

This file contains the interface for general quadratures.
Do not confuse quadratures with quadrature rules! Quadrature rules
are structs containing just nodes and weights and some convenience
methods. Quadratures are classes that really can compute things
like inner products (brakets) etc.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""


class Quadrature(object):
    r"""This class is an abstract interface to quadratures in general.
    """

    def __init__(self):
        r"""General interface for quadratures.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def __str__(self):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def get_description(self):
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def set_qr(self, QR):
        r"""Set the :py:class:`QuadratureRule` subclass instance used for quadrature.

        :param QR: The new :py:class:`QuadratureRule` instance.
        """
        # TODO: Allow a list of QRs, one QR for each component of Psi
        self._QR = QR


    def get_qr(self):
        r"""Return the :py:class:`QuadratureRule` subclass instance used for quadrature.

        :return: The current instance of the quadrature rule.
        """
        return self._QR


    def transform_nodes(self):
        r"""Transform the quadrature nodes such that they fit the given wavepacket.
        Note that the arguments may vary through subclasses!

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def quadrature(self):
        r"""Performs the quadrature of :math:`\langle\Psi|f|\Psi\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        Note that the arguments may vary through subclasses!

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def build_matrix(self):
        r"""Calculate the matrix elements of :math:`\langle\Psi|f|\Psi\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        Note that the arguments may vary through subclasses!

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")
