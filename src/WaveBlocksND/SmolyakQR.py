"""The WaveBlocks Project

This file contains the class for constructing high dimensional
quadrature rules from one-dimensional ones by using the
Smolyak construction.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from copy import deepcopy
from numpy import zeros, hstack, vsplit, squeeze, lexsort, integer
from numpy.linalg import norm
from scipy.special import binom

from QuadratureRule import QuadratureRule
from TensorProductQR import TensorProductQR

__all__ = ["SmolyakQR"]


class SmolyakQR(QuadratureRule):
    r"""This class implements the construction of high dimensional
    quadrature rules from one-dimensional ones by applying the
    Smolyak construction.
    """

    def __init__(self, dimension, level, rules, options={}):
        r"""
        Initialize a :py:class:`SmolyakQR` instance.

        :param dimension: The dimension :math:`D` of the final
                          quadrature rule.
        :param level: The level :math:`k` of the Smolyak construction.
                      From theory we know that a Smolyak rule of order
                      :math:`k` is exact up to :math:`2 k - 1` if the
                      individual rules :math:`Q_i` are exact up to
                      :math:`2 i - 1`.
        :param rules: A list of :py:class:`QuadratureRule` subclass
                      instances. Their nodes and weights will be used
                      in the Smolyak construction.

        .. note:: The ``rules`` object must implement simple indexing by
                  non-negative numbers and for a given index :math:`j` return
                  a univariate quadrature rule of order :math:`j`. Special attention
                  must be payed in case this object is mutable. We do not copy it.
        """
        # The dimension of the quadrature rule.
        self._dimension = dimension

        # The construction level
        self._level = level

        # The individual quadrature rules.
        self._rules = rules

        # The level of the Smolyak sparse grid quadrature
        self._order = level

        # Set the options
        self._options = options

        # The number of nodes in this quadrature rule.
        self._number_nodes = None

        # The quadrature nodes \gamma.
        self._nodes = None

        # The quadrature weights \omega.
        self._weights = None


    def __str__(self):
        s = "Sparse grid (Smolyak) quadrature rule consisting of:\n"
        l = ["  " + str(rule) + "\n" for k,rule in self._rules.iteritems() if k <= self._level]
        s += reduce(lambda x,y: x+y, l)
        return s


    def get_description(self):
        r"""Return a description of this quadrature rule object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "SmolyakQR"
        d["dimension"] = self._dimension
        d["qr_rules"] = [ qr.get_description() for qr in self._rules ]
        d["options"] = deepcopy(self._options)
        return d


    def get_nodes(self, flat=True, split=False):
        r"""Returns the quadrature nodes :math:`\gamma_i`.

        :param flat: Dummy parameter for API compatibility with Grids.
        :type flat: Boolean, default is ``True``.
        :param split: Dummy parameter for API compatibility with Grids.
        :type split: Boolean, default is ``False``.
        :return: An ndarray containing the quadrature nodes :math:`\gamma_i`.
        """
        if self._nodes is None:
            self.construct_rule()
        return self._nodes.copy()


    def get_weights(self):
        r"""Returns the quadrature weights :math:`\omega_i`.

        :return: An ndarray containing the quadrature weights :math:`\omega_i`.
        """
        if self._weights is None:
            self.construct_rule()
        return self._weights.copy()


    def enumerate_lattice_points(self, N, D=None):
        r"""This method enumerates all lattice points of a lattice
        :math:`\Lambda \subset \mathbb{N}^D` in :math:`D` dimensions
        having fixed :math:`l_1` norm :math:`N`.

        :param N: The :math:`l_1` norm of the lattice points.
        :param D: The dimension :math:`D` of the lattice.
        """
        if D is None:
            D = self._dimension

        k = zeros(D, dtype=integer)
        k[0] = N
        yield tuple(k)

        c = 1
        while k[D-1] < N:
            if c == D:
                for i in xrange(c-1,0,-1):
                    c = i
                    if not k[i-1] == 0:
                        break
            k[c-1] = k[c-1] - 1
            c += 1
            k[c-1] = N - sum(k[0:c-1])
            if c < D:
                k[c:D] = zeros(D-c, dtype=integer)

            yield tuple(k)


    def construct_rule(self, level=None, tolerance=1e-15):
        r"""Compute the quadrature nodes :math:`\{\gamma_i\}_i` and quadrature
        weights :math:`\{\omega_i\}_i`.

        :param level: The level :math:`k` of the Smolyak construction.
        :param tolerance: Tolerance for dropping identical
                          quadrature nodes.

        .. warning:: This method is very expensive and may take a long time
                     to finish. Also, the quadrature nodes may use large amounts
                     of memory depending on the dimension and level parameters.
        """
        D = self._dimension

        if level is None:
            k = self._level
        else:
            k = level

        if k > max(self._rules.keys()):
            raise ValueError("Not enough quadrature rules to build Smolyak grid of level "+str(k))

        allnodes = []
        allweights = []
        factors = []

        # Index Set
        for q in xrange(max(0, k-D), k):
            S = self.enumerate_lattice_points(q)
            for j,s in enumerate(S):
                # TODO: Only use non-negative nodes for the construction.
                #       The final rule is assumed to be symmetric.
                Q = TensorProductQR([self._rules[si+1] for si in s])
                allnodes.append(Q.get_nodes())
                allweights.append(Q.get_weights())
                factors.append((-1)**(k-1-q) * binom(D-1, k-1-q))

        # Sort
        allnodes = hstack(allnodes).reshape(D,-1)
        allweights = hstack([factor*weights for factor, weights in zip(factors,allweights)]).reshape(1,-1)

        I = lexsort(map(squeeze, vsplit(allnodes, D))[::-1])

        allnodes = allnodes[:,I].reshape(D,-1)
        allweights = allweights[:,I].reshape(1,-1)

        # Remove duplicates
        last = 0
        I = [last]

        no = norm(allnodes[:,:-1] - allnodes[:,1:], axis=0)

        for col in xrange(1, allnodes.shape[1]):
            if no[col-1] <= tolerance:
                allweights[0,last] += allweights[0,col]
                allweights[0,col] = 0
            else:
                last = col
                I.append(last)

        allnodes = allnodes[:,I]
        allweights = allweights[:,I]

        self._nodes = allnodes
        self._weights = allweights
        self._number_nodes = allnodes.shape[1]
