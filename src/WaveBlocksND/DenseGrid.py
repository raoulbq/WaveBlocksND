"""The WaveBlocks Project

This file contains the abstract class for representing dense grids.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from Grid import Grid

__all__ = ["DenseGrid"]


class DenseGrid(Grid):
    """This class is an abstract interface to dense grids in general.
    """

    def __init__(self):
        """
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'DenseGrid' is an abstract interface.")


    def is_regular(self):
        """Answers the question if the grid spacing is regular.
        (It can still be different along each axis!)
        """
        return self._is_regular
