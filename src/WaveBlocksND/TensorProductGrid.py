r"""The WaveBlocks Project

This file contains a class for representing
dense regular tensor product grids.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import operator
from numpy import mgrid, ogrid, atleast_1d, array, diff, squeeze, hstack, floating, complexfloating

from DenseGrid import DenseGrid

__all__ = ["TensorProductGrid"]


class TensorProductGrid(DenseGrid):
    r"""This class represents a dense tensor product grid. It can
    have an arbitrary dimension :math:`D`. The grid nodes are
    enclosed in a hypercubic bounding box. This box can have
    different limits :math:`min_i`, :math:`max_i` along each
    axis :math:`x_i`. In each of these intervalls we place
    :math:`N_i` grid nodes. Note that the point :math:`max_i`
    is not part of the grid. The grid interior is build as the
    tensor product of all the grid nodes along all the axes.
    """

    def __init__(self, limits, number_nodes):
        r"""Construct a tnesor product grid instace.

        :param limits: The grid domain limits along each axis.
        :type limits: A list of two-element tuples.
        :param number_nodes: The number of grid nodes along each axis.
        :type number_nodes: A list of positive integers.
        :return: A :py:class:`TensorProductGrid` instance.
        """
        assert len(limits) == len(number_nodes)

        # Regular grid spacing
        self._is_regular = True

        # The dimension of the grid
        self._dimension = len(limits)

        # The number of grid nodes along each axis
        self._number_nodes = number_nodes
        # format: [N_1, ..., N_D]

        # The limits of the bounding box of the grid
        self._limits = [ array(limit) for limit in limits ]
        # format: [(min_0,max_0), ..., (min_D,max_D)]

        # The extensions (edge length) of the bounding box
        self._extensions = hstack([ abs(diff(limit)) for limit in self._limits ])

        # Compute the grid spacings along each axis
        self._meshwidths = self._extensions / squeeze(array(self._number_nodes, dtype=floating))
        # format: [h_1, ..., h_D]

        # Cached values
        self._gridaxes = None
        self._gridnodes = None


    def get_limits(self, axes=None):
        r"""Returns the limits of the bounding box.

        :param axes: The axes for which we want to get the limits.
        :type axes: A single integer or a list of integers. If set
                    to ``None`` (default) we return the limits for all axes.
        :return: A list of :math:`(min_i, max_i)` ndarrays.
        """
        if axes is None:
            axes = range(self._dimension)

        return [ self._limits[i] for i in atleast_1d(axes) ]


    def get_extensions(self, axes=None):
        r"""Returns the extensions (length of the edges) of the bounding box.

        :param axes: The axes for which we want to get the extensions.
        :type axes: A single integer or a list of integers. If set
                    to ``None`` (default) we return the extensions for all axes.
        :return: A list of :math:`|max_i-min_i|` values.
        """
        if axes is None:
            axes = range(self._dimension)

        return [ self._extensions[i] for i in atleast_1d(axes) ]


    def get_meshwidths(self, axes=None):
        r"""Returns the meshwidths of the grid.

        :param axes: The axes for which we want to get the meshwidths.
        :type axes: A single integer or a list of integers. If set
                    to ``None`` (default) we return the data for all axes.
        :return: A list of :math:`h_i` values or a single value.
        """
        if axes is None:
            axes = xrange(self._dimension)

        return [ self._meshwidths[i] for i in atleast_1d(axes) ]


    def get_number_nodes(self, axes=None, overall=False):
        r"""Returns the number of grid nodes along a set of axes.

        :param axes: The axes for which we want to get the number of nodes.
        :type axes: A single integer or a list of integers. If set
                    to ``None`` (default) we return the data for all axes.
        :param overall: Compute the product :math:`\prod_i^D N_i` of the
                        number :math:`N_i` of grid nodes along each axis
                        :math:`i` specified.
        :type overall: Boolean, default is ``False``
        :return: A list of :math:`N_i` values or a single value :math:`N`.
        """
        if axes is None:
            axes = xrange(self._dimension)

        values = [ self._number_nodes[i] for i in atleast_1d(axes) ]

        if overall is True:
            return reduce(operator.mul, values)
        else:
            return values


    def _build_slicers(self):
        # Helper routine to build the necessary slicing
        # objects used for constructing the grid nodes.
        slicers = [ slice(lims[0], lims[1], step) for lims, step in zip(self._limits, self._meshwidths) ]
        return slicers


    def _compute_grid_axes(self):
        # Helper routine which computes the one-dimensional
        # grids along all axes and caches the result. Each
        # grid is a D-dimensional ndarray of correct shape.
        S = self._build_slicers()
        self._gridaxes = [ array(ax, dtype=complexfloating) for ax in ogrid[S] ]


    def _compute_grid_full(self):
        # Helper routine which computes the full set of
        # tensor product grid nodes and caches the result.
        # The result is a (D, product(N_i)) shaped ndarray.
        S = self._build_slicers()
        # TODO: Code is 4x slower withOUT complex floating
        grid = array(mgrid[S], dtype=complexfloating)
        self._gridnodes = grid.reshape((self._dimension, self.get_number_nodes(overall=True)))


    def get_axes(self, axes=None):
        r"""Returns the one-dimensional grids along the axes.

        :param axes: The axes for which we want to get the grid.
        :type axes: A single integer or a list of integers. If set
                    to ``None`` (default) we return the data for all axes.
        :return: A list of ndarrays, each having a shape of :math:`(1,...,N_i,...,1)`.
                 We return a list even if it contains just a single element.
        """
        if self._gridaxes is None:
            self._compute_grid_axes()

        if axes is None:
            axes = xrange(self._dimension)
        axes = atleast_1d(axes)

        return [ self._gridaxes[i] for i in axes ]


    def get_nodes(self, flat=True, split=False):
        r"""Returns all grid nodes of the full tensor product grid.

        :param flat: Whether to return the grid with a `hypercubic`
                     :math:`(D, N_1, ..., N_D)` shape or a `flat`
                     :math:`(D, \prod_i^D N_i)` shape.
        :type flat: Boolean, default is ``True``.
        :param split: Whether to return the different components, one for each
                      dimension inside a single ndarray or a list with ndarrays,
                      with one item per dimension.
        :type split: Boolean, default is ``False``.
        :return: Depends of the optional arguments.
        """
        if self._gridnodes is None:
            self._compute_grid_full()

        # All operations here only return views to the original grid data
        if flat is True:
            if split is False:
                return self._gridnodes
            else:
                return tuple([ self._gridnodes[i,:] for i in xrange(self._dimension) ])
        else:
            if split is False:
                return self._gridnodes.reshape([self._dimension] + self.get_number_nodes())
            else:
                return tuple([ self._gridnodes[i,:].reshape(self.get_number_nodes()) for i in xrange(self._dimension) ])
