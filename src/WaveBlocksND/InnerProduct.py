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

__all__ = ["InnerProduct", "InnerProductException"]


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
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        raise NotImplementedError("'InnerProduct' is an abstract interface.")


    # TODO: Rename 'quadrature', the delegate can be something other than a classical quadrature.
    def set_quadrature(self, ip):
        r"""Set the :py:class:`Quadrature` subclass instance used for quadrature.

        :param ip: The delegate inner product.
        :type ip: An instance of a :py:class:`Quadrature` subclass or
                  of another :py:class:`InnerProduct` subclass.
        """
        # TODO: Allow a list of ips, one quad for each component of Psi
        self._quad = ip


    # TODO: Rename 'quadrature', the delegate can be something other than a classical quadrature.
    def get_quadrature(self):
        r"""Return the :py:class:`Quadrature` subclass instance
        used for evaluation of this inner product.

        :return: The current instance of the quadrature.
        """
        return self._quad


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


class InnerProductException(Exception):
    r"""Exception to raise in case an inner product fails for whatever reason.
    """
    pass
