"""The WaveBlocks Project

This file contains the class for trapezoidal quadrature rules.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2013 R. Bourquin
@license: Modified BSD License
"""

from copy import deepcopy
from numpy import linspace, abs, ones

from QuadratureRule import QuadratureRule

__all__ = ["TrapezoidalQR"]


class TrapezoidalQR(QuadratureRule):
    r"""This class implements a trapezoidal quadrature rule.
    """

    def __init__(self, left, right, number_nodes, options={}):
        r"""Initialize a new quadrature rule.

        :param number_nodes: The order :math:`R` of the trapezoidal quadrature rule.

        :raise: :py:class:`ValueError` if the ``number_nodes`` is not 1 or above.
        """
        # The space dimension of the quadrature rule.
        self._dimension = 1

        # The order R of the trapezoidal quadrature rule.
        self._number_nodes = number_nodes

        # Qudrature has to have at least a single (node,weight) pair.
        if self._number_nodes < 1:
            raise ValueError("Quadrature rule has to be of order 1 at least.")

        # Set the options
        self._options = options

        self._left = left
        self._right = right
        self._compute_qr()


    def __str__(self):
        return "Trapezoidal quadrature rule of number_nodes " + str(self._number_nodes)


    def get_description(self):
        r"""Return a description of this quadrature rule object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "TrapezoidalQR"
        d["dimension"] = self._dimension
        d["left"] = self._left
        d["right"] = self._right
        d["order"] = self._number_nodes
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


    def _compute_qr(self):
        nodes = linspace(self._left, self._right, self._number_nodes)
        # The number of nodes in this quadrature rule
        self._number_nodes = nodes.size
        dx = abs(self._right-self._left) / (1.0*self._number_nodes-1.0)
        weights = ones(nodes.shape)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights = dx * weights
        self._nodes = nodes.reshape((1, nodes.size))
        self._weights = weights.reshape((1, weights.size))
