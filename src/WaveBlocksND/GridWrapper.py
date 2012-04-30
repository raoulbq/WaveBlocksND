"""The WaveBlocks Project

This file contains a tiny wrapper to wrap
numpy ndarrays into Grid instances.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from Grid import Grid


class GridWrapper(Grid):
    r"""This class constructs a thin layer around an ``ndarray`` and wrappes
    it as :py:class:`Grid` subclass for API compatibility. The array must
    have a shape of :math:`(D, N)` with :math:`N` the overall number of nodes.

    Rather than using this class, one should try to elliminate the
    cases where it is used now.
    """

    def __init__(self, anarray):
        # Shape is (D, #nodes)
        self._data = anarray
        self._dimension = self._data.shape[0]


    def get_number_nodes(self, overall=False):
        r"""Returns the number of grid nodes.

        :param overall: Compute the product :math:`N = \prod_i^D N_i` of the
                        number :math:`N_i` of grid nodes along each dimension
                        :math:`i` specified.
        :type overall: Boolean, default is ``False``
        :return: A list of :math:`N_i` values or a single value :math:`N`.
        """
        if overall is False:
            return self._data.shape[1:]
        else:
            return reduce(lambda x,y: x*y, self._data.shape[1:])


    def get_nodes(self, flat=True, split=False):
        r"""Returns all grid nodes.

        :param flat: Whether to return the grid with a `hypercubic`
                     :math:`(D, N_1, ..., N_D)` shape or a `flat`
                     :math:`(D, \prod_i^D N_i)` shape. Note that the
                     hypercubic shape is not implemented!
        :type flat: Boolean, default is `True`.
        :param split: Whether to return the different components, one for each
                      dimension inside a single ndarray or a list with ndarrays,
                      with one item per dimension.
        :type split: Boolean, default is `False`.
        :return: Depends of the optional arguments.
        """
        if flat is False:
            raise NotImplementedError("Grid wrapping for hypercubic storageB")

        if split is True:
            return [ self._data[i,...] for i in xrange(self._data.shape[0]) ]
        else:
            return self._data
