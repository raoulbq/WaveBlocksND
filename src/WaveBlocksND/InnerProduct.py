"""The WaveBlocks Project

This file contains the interface for general quadratures.
Do not confuse quadratures with quadrature rules! Quadrature rules
are structs containing just nodes and weights and some convenience
methods. InnerProducts are classes that really can compute things
like inner products (brakets) etc.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["InnerProduct"]


class InnerProduct(object):
    r"""This class is an abstract interface to inner products in general.
    """

    def __init__(self):
        r"""General interface for quadratures.

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'InnerProduct' is an abstract interface.")


    def __str__(self):
        raise NotImplementedError("'InnerProduct' is an abstract interface.")


    def get_description(self):
        r"""Return a description of this inner product object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        raise NotImplementedError("'InnerProduct' is an abstract interface.")


    def set_delegate(self, delegate):
        r"""Set the :py:class:`Quadrature` subclass instance used for quadrature.

        :param delegate: The new :py:class:`Quadrature` instance.
        """
        # TODO: Allow a list of quads, one quad for each component of Psi
        self._delegate = delegate


    def get_delegate(self):
        r"""Return the :py:class:`Quadrature` subclass instance
        used for evaluation of this inner product.

        :return: The current instance of the quadrature.
        """
        return self._delegate


    def quadrature(self):
        r"""Performs the quadrature of :math:`\langle\Psi|f|\Psi\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        Note that the arguments may vary through subclasses!

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'InnerProduct' is an abstract interface.")


    def build_matrix(self):
        r"""Calculate the matrix elements of :math:`\langle\Psi|f|\Psi\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        Note that the arguments may vary through subclasses!

        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'InnerProduct' is an abstract interface.")
