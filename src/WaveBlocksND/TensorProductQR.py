"""The WaveBlocks Project

This file contains the class for constructing high dimensional
quadrature rules from one-dimensional ones by taking tensor products.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import operator as op
from numpy import vstack

from QuadratureRule import QuadratureRule
from Utils import meshgrid_nd


class TensorProductQR(QuadratureRule):
    r"""This class implements the construction of high dimensional
    quadrature rules from one-dimensional ones by taking tensor products.
    """

    def __init__(self, rules):
        r"""
        Initialize a :py:class:`TensorProductQR` instance.

        :param rules: A list of :py:class:`QuadratureRule` subclass instances. Their
                      nodes and weights will be used to compute the tensor product.
        """
        # The dimension of the quadrature rule.
        self._dimension = len(rules)

        # The individual quadrature rules.
        self._qrs = tuple(rules)

        # The order R of the tensor product quadrature.
        self._order = None
        # TODO: Compute the order from the orders of the input QRs

        # The number of nodes in this quadrature rule.
        self._number_nodes = reduce(op.mul, [ rule.get_number_nodes() for rule in rules ])

        # The quadrature nodes \gamma.
        nodes = meshgrid_nd([ rule.get_nodes() for rule in rules ])
        self._nodes = vstack([ node.flatten() for node in nodes ])

        # The quadrature weights \omega.
        weights = meshgrid_nd([ rule.get_weights() for rule in rules ])
        weights = reduce(lambda x,y: x*y, weights)
        self._weights = weights.flatten()


    def get_description(self):
        r"""Return a description of this quadrature rule object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "TensorProductQR"
        d["dimension"] = self._dimension
        d["qr_rules"] = [ qr.get_description() for qr in self._qrs ]
        return d


    def get_nodes(self, flat=True, split=False):
        r"""Returns the quadrature nodes :math:`\gamma_i`.

        :param flat: Dummy parameter for API compatibility with Grids.
        :type flat: Boolean, default is `True`.
        :param split: Dummy parameter for API compatibility with Grids.
        :type split: Boolean, default is `False`.
        :return: An ndarray containing the quadrature nodes :math:`\gamma_i`.
        """
        return self._nodes.copy()


    def get_weights(self):
        r"""Returns the quadrature weights :math:`\omega_i`.

        :return: An ndarray containing the quadrature weights :math:`\omega_i`.
        """
        return self._weights.copy()
