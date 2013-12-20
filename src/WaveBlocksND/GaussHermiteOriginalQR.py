"""The WaveBlocks Project

This file contains the class for Gauss-Hermite quadrature.
The quadrature is not transformed to exclude the exponential
weight factor.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from copy import deepcopy
from scipy.special.orthogonal import h_roots

from QuadratureRule import QuadratureRule

__all__ = ["GaussHermiteOriginalQR"]


class GaussHermiteOriginalQR(QuadratureRule):
    r"""This class implements a Gauss-Hermite quadrature rule.
    The quadrature is not transformed to exclude the exponential
    weight factor.
    """

    def __init__(self, order, options={}):
        r"""Initialize a new quadrature rule.

        :param order: The order :math:`k` of the Gauss-Hermite quadrature.
                      From theory we know that a Gauss quadrature rule
                      of order :math:`k` is exact for polynomials up to
                      degree :math:`2 k - 1`.

        :raise: :py:class:`ValueError` if the ``order`` is not 1 or above.
        """
        # The space dimension of the quadrature rule.
        self._dimension = 1

        # The order of the Gauss-Hermite quadrature.
        self._order = order

        # Qudrature has to have at least a single (node,weight) pair.
        if not self._order > 0:
            raise ValueError("Quadrature rule has to be of order 1 at least.")

        # Set the options
        self._options = options

        nodes, weights = h_roots(self._order)

        # The number of nodes in this quadrature rule
        self._number_nodes = nodes.size

        # The quadrature nodes \gamma.
        self._nodes = nodes.reshape((1,self._number_nodes))
        # The quadrature weights \omega.
        self._weights = weights.reshape((1,self._number_nodes))


    def __str__(self):
        return "Gauss-Hermite quadrature rule (untransformed) of order " + str(self._order)


    def get_description(self):
        r"""Return a description of this quadrature rule object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "GaussHermiteOriginalQR"
        d["dimension"] = self._dimension
        d["order"] = self._order
        d["options"] = deepcopy(self._options)
        return d


    def get_nodes(self):
        r"""Returns the quadrature nodes :math:`\gamma_i`.

        :return: An array containing the quadrature nodes :math:`\gamma_i`.
        """
        return self._nodes.copy()


    def get_weights(self):
        r"""Returns the quadrature weights :math:`\omega_i`.

        :return: An array containing the quadrature weights :math:`\omega_i`.
        """
        return self._weights.copy()
