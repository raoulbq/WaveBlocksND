"""The WaveBlocks Project

This file contains the class for representing the hyperbolic cut
basis shape which is a special type of sparse basis set.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import eye, vstack, integer

from BasisShape import BasisShape
from HyperbolicCutShape import HyperbolicCutShape

__all__ = ["LimitedHyperbolicCutShape"]


class LimitedHyperbolicCutShape(BasisShape):
    r"""This class implements the hyperbolic cut basis shape which
    is a special type of sparse basis set. A basis shape is essentially
    all information and operations related to the set :math:`\mathfrak{K}`
    of multi-indices :math:`k`. The hyperbolic cut shape in :math:`D` dimensions
    with `sparsity` :math:`S` and limits :math:`K = (K_0,\ldots,K_{D-1})`
    is defined as the set

    .. math::
        \mathfrak{K}(D, S, K) := \{ (k_0, \ldots, k_{D-1}) |
                                    0 \leq k_d < K_d \forall d \in [0,\ldots,D-1]
                                    \land \prod_{d=0}^{D-1}(1+k_d) \leq S \}
    """

    def __init__(self, D, K, limits):
        r"""
        """
        # The dimension of K
        self._dimension = D

        # The sparsity parameter
        self._sparsity = K

        # The limits
        self._limits = tuple(limits)

        # The linear mapping k -> index for the basis
        iil = self._get_index_iterator_lex()
        self._lima = {k:index for index, k in enumerate(iil)}
        # And the inverse mapping
        self._lima_inv = {v:k for k, v in self._lima.iteritems()}

        # The basis size
        self._basissize = len(self._lima)


    def __str__(self):
        r""":return: A string describing the basis shape :math:`\mathfrak{K}`.
        """
        s = ("Hyperbolic cut basis shape of dimension "+str(self._dimension)+
             " and sparsity "+str(self._sparsity)+" limited at "+str(self._limits)+".")
        return s


    def __hash__(self):
        r"""Compute a unique hash for the basis shape. In the case of hyperbolic
        cut basis shapes :math:`\mathfrak{K}` the basis is fully specified by its
        dimension :math:`D` and the sparsity parameter :math:`K`.
        """
        return hash(("LimitedHyperbolicCutShape", self._dimension, self._sparsity, self._limits))


    def __getitem__(self, k):
        r"""Make map look ups.
        """
        if type(k) is tuple or type(k) is list:
            k = tuple(k)
            assert len(k) == self._dimension
            if k in self._lima:
                return self._lima[k]
        elif type(k) is int:
            if k in self._lima_inv:
                return self._lima_inv[k]
        else:
            raise IndexError("Wrong index type")


    def __contains__(self, k):
        r"""
        Checks if a given multi-index :math:`k` is part of the basis set :math:`\mathfrak{K}`.

        :param k: The multi-index :math:`k` we want to test.
        :type k: tuple
        """
        assert len(tuple(k)) == self._dimension
        return tuple(k) in self._lima


    def __iter__(self):
        r"""Implements iteration over the multi-indices :math:`k`
        of the basis set :math:`\mathfrak{K}`.

        Note: The order of iteration is NOT fixed. If you need a special
        iteration scheme, use :py:meth:`get_node_iterator`.
        """
        # TODO: Better remove this as it may cause unexpected behaviour?
        return iter(self._lima)


    def contains(self, k):
        r"""
        Checks if a given multi-index :math:`k` is part of the basis set :math:`\mathfrak{K}`.

        :param k: The multi-index :math:`k` we want to test.
        :type k: tuple
        """
        return tuple(k) in self._lima


    def get_description(self):
        r"""Return a description of this basis shape object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current basis shape. A description
        never contains any data.
        """
        d = {}
        d["type"] = "LimitedHyperbolicCutShape"
        d["dimension"] = self._dimension
        d["K"] = self._sparsity
        d["limits"] = self._limits
        return d


    def extend(self, tight=True):
        r"""Extend the basis shape such that (at least) all neighbours of all
        boundary nodes are included in the extended basis shape.

        :param tight: Whether to cut off the long tails.
        :type tight: Boolean, default is ``False``
        """
        D = self._dimension
        K = self._sparsity
        if D > 1:
            # This formula is more narrow than: K = 2**(D-1) * (K+1)
            # but works only for D >= 2
            new_sparsity = 2**(D-1) * K
        else:
            # Special casing K = 2**(D-1) * (K+1) for D = 1
            new_sparsity = K + 1

        if tight is True:
            new_limits = tuple([ l+1 for l in self._limits ])
            return LimitedHyperbolicCutShape(D, new_sparsity, new_limits)
        else:
            return HyperbolicCutShape(D, new_sparsity)


    def _get_index_iterator_lex(self):
        r"""
        """
        # The hyperbolic cut parameter
        sparsity = self._sparsity
        # Upper bounds in each dimension
        bounds = self._limits[::-1]

        def index_iterator_lex(S, bounds):
            # Initialize a counter
            z = [0 for i in xrange(self._dimension + 1)]

            while z[self._dimension] == 0:
                # Yield the current index vector
                yield tuple(reversed(z[:-1]))

                # Increment fastest varying bit
                z[0] += 1

                # Reset overflows
                for d in xrange(self._dimension):
                    K = reduce(lambda x,y: x*(y+1), z[:-1], 1)
                    if z[d] >= bounds[d] or K > S:
                        z[d] = 0
                        z[d+1] += 1

        return index_iterator_lex(sparsity, bounds)


    def _get_index_iterator_chain(self, direction=0):
        r"""
        """
        # The hyperbolic cut parameter
        sparsity = self._sparsity
        # Upper bounds in each dimension
        bounds = self._limits[::-1]

        def index_iterator_chain(S, bounds, d):
            D = self._dimension
            # The counter
            z = [ 0 for i in range(D + 1) ]

            # Iterate over all valid stencil points
            while z[D] == 0:
                yield tuple(reversed(z[:-1]))

                # Increase index in the dimension we build the chain
                z[D-d-1] += 1

                # Check if we are done with the current base point
                # If yes, move base point and start a new chain
                # Reset overflows
                for i in xrange(D-d-1, D):
                    K = reduce(lambda x,y: x*(y+1), z[(D-d-1):-1], 1)
                    if z[i] > bounds[i]-1 or K > S:
                        z[i] = 0
                        z[i+1] += 1

        return index_iterator_chain(sparsity, bounds, direction)


    def get_node_iterator(self, mode="lex", direction=None):
        r"""
        Returns an iterator to iterate over all basis elements :math:`k \in \mathfrak{K}`.

        :param mode: The mode by which we iterate over the indices. Default is ``lex``
                     for lexicographical order. Supported is also ``chain``, for
                     the chain-like mode, details see the manual.
        :type mode: string
        :param direction: If iterating in `chainmode` this specifies the direction
                          the chains go.
        :type direction: integer.
        """
        if mode == "lex":
            return self._get_index_iterator_lex()
        elif mode == "chain":
            if direction < self._dimension:
                return self._get_index_iterator_chain(direction=direction)
            else:
                raise ValueError("Can not build iterator for this direction.")
        # TODO: Consider boundary node only iterator
        else:
            raise ValueError("Unknown iterator mode: "+str(mode)+".")


    def get_limits(self):
        r"""Returns the upper limit :math:`K_d` for all directions :math:`d`.

        :return: A tuple of the maximum of the multi-index in each direction.
        """
        return tuple(self._limits)


    def get_neighbours(self, k, selection=None, direction=None):
        r"""
        Returns a list of all multi-indices that are neighbours of a given
        multi-index :math:`k`. A direct neighbour is defined as
        :math:`(k_0, \ldots, k_d \pm 1, \ldots, k_{D-1}) \forall d \in [0 \ldots D-1]`.

        :param k: The multi-index of which we want to get the neighbours.
        :type k: tuple
        :param selection:
        :type selection: string with fixed values ``forward``, ``backward`` or ``all``.
                         The values ``all`` is equivalent to the value ``None`` (default).
        :param direction: The direction :math:`0 \leq d < D` in which we want to find
                          the neighbours :math:`k \pm e_d`.
        :type direction: int
        :return: A list containing the pairs :math:`(d, k^\prime)`.
        """
        assert len(tuple(k)) == self._dimension

        # First build a list of potential neighbours
        I = eye(self._dimension, dtype=integer)
        ki = vstack(k)

        # Forward and backward direct neighbours
        nbfw = ki + I
        nbbw = ki - I

        # Keep only the valid ones
        nbh = []

        if direction is not None:
            directions = [ direction ]
        else:
            directions = xrange(self._dimension)

        for d in directions:
            nfw = tuple(nbfw[:,d])
            nbw = tuple(nbbw[:,d])

            # TODO: Try to simplify these nested if blocks
            if selection in ("backward", "all", None):
                if nbw in self:
                    nbh.append((d, nbw))

            if selection in ("forward", "all", None):
                if nfw in self:
                    nbh.append((d, nfw))

        return nbh
