"""The WaveBlocks Project

This file contains a tiny wrapper to wrap
numpy ndarrays into Grid instances.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from Grid import Grid


class GridWrapper(Grid):

    def __init__(self, anarray):
        self._data = anarray


    def get_dimension(self):
        return self._data.ndim


    def get_number_nodes(self, overall=True):
        if overall is False:
            return self._data.shape
        else:
            return reduce(lambda x,y: x*y, self._data.shape[1:])


    def get_nodes(self, split=True):
        if split is True:
            return [ self._data[i,...] for i in xrange(self._data.shape[0]) ]
        else:
            return self._data
