"""The WaveBlocks Project


@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["Quadrature"]


class Quadrature(object):
    r"""This class is an abstract interface to quadratures in general.
    """

    def __init__(self):
        r"""General interface for quadratures.

        :raise: :py:class:`NotImplementedError` Abstract interface.
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


    def initialize_packet(self):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def initialize_operator(self):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def prepare(self, rows, cols):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def prepare_for_row(self, row, col):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def preprare_for_col(self, row, col):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def perform_quadrature(self, row, col):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def perform_build_matrix(self, row, col):
        raise NotImplementedError("'Quadrature' is an abstract interface.")
