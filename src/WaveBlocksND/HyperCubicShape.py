"""The WaveBlocks Project

This file contains the class for representing the hypercubic basis shape
which is the full dense basis set.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, eye, vstack, integer

from BasisShape import BasisShape

__all__ = ["HyperCubicShape"]


class HyperCubicShape(BasisShape):
    r"""This class implements the hypercubic basis shape
    which is the full dense basis set.
    A basis shape is essentially all information and operations
    related to the set :math:`\mathfrak{K}` of multi-indices :math:`k`.
    """

    def __init__(self, limits):
        r"""
        """
        # The dimension of K
        self._dimension = len(limits)

        # The limits Ki for each axis
        self._limits = tuple(limits)

        # TODO: Do we really want to store these maps or better compute data the fly

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
        s = ("Hypercubic basis shape of dimension "+str(self._dimension)+
             " and with limits "+str(self._limits)+".")
        return s


    def __hash__(self):
        r"""Compute a unique hash for the basis shape. In the case of hypercubic
        basis shapes :math:`\mathfrak{K}` the basis is fully specified by its
        maximal index :math:`K_i` along each direction :math:`i \in [0,\ldots,D-1]`.
        """
        return hash(("HyperCubicShape", self._limits))


    def __getitem__(self, k):
        r"""Make map lookups.
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

        :param k: The multi-index we want to test.
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

        :param k: The multi-index we want to test.
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
        d["type"] = "HyperCubicShape"
        d["limits"] = self._limits
        return d


    def extend(self):
        r"""Extend the basis shape such that (at least) all neighbours of all
        boundary nodes are included in the extended basis shape.
        """
        extended_limits = [ l+1 for l in self._limits ]
        return HyperCubicShape(extended_limits)


    def _get_index_iterator_lex(self):
        r"""
        """
        # Upper bounds in each dimension
        bounds = self._limits[::-1]

        def index_iterator_lex(bounds):
            # Initialize a counter
            z = [0 for i in xrange(self._dimension + 1)]

            while z[self._dimension] == 0:
                # Yield the current index vector
                yield tuple(reversed(z[:-1]))

                # Incremet fastest varying bit
                z[0] += 1

                # Reset overflows
                for d in xrange(self._dimension):
                    if z[d] >= bounds[d]:
                        z[d] = 0
                        z[d+1] += 1

        return index_iterator_lex(bounds)


    def _get_index_iterator_chain(self, direction=0):
        r"""
        """
        def index_iterator_chain(d):
            # Number of functions in each dimension
            bounds = self._limits[:]

            # The counter
            z = [ 0 for i in range(self._dimension + 1) ]

            # Iterate over all valid stencil points
            while not z[-1] > 0:
                # Otherwise we would yield k = (0,...,0) for limits = [1,...,1]
                if not z[d] > bounds[d]-2:
                    yield tuple(z[:-1])

                # Increase index in the dimension we build the chain
                z[d] += 1

                # Check if we are done with the current base point
                # If yes, move base point and start a new chain
                if z[d] > bounds[d]-2:
                    z[d] = 0
                    z[d-1] += 1

                    for i in reversed(range(d)):
                        if z[i] > bounds[i]-1:
                            z[i] = 0
                            z[i-1] += 1

        return index_iterator_chain(direction)


    def get_node_iterator(self, mode="lex", direction=None):
        r"""
        Returns an iterator to iterate over all basis elements :math:`k`.

        :param mode: The mode by which we iterate over the indices. Default is 'lex'
                     for lexicographical order. Supported is also 'chain', for
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
                raise ValueError("Can not build chain iterator for this direction.")
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


    def find_largest_index(self):
        r"""
        Find the index :math:`k \in \mathfrak{K}` with maximal distance
        :math:`\sum_{d=0}^D k_d^2` from the zero index. In case of
        multiple maxima the method returns the first one found.
        """
        return tuple(array(self._limits) - 1)
