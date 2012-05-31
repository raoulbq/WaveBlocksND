"""The WaveBlocks Project

IOM plugin providing functions for handling grid data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_grid(self, parameters, blockid=0):
    """Add storage for a tensor product grid.

    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the keys `number_nodes` and `dimension`.
    :param blockid: The ID of the data block to operate on.
    """
    # TODO: Consider storing axes, extensions etc too
    # TODO: What about grid types other than tensor product grids?
    self._srf[self._prefixb+str(blockid)].create_dataset("grid", [parameters["dimension"]] + list(parameters["number_nodes"]), np.floating)


def delete_grid(self, blockid=0):
    """Remove the stored grid.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/grid"]
    except KeyError:
        pass


def has_grid(self, blockid=0):
    """Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return "grid" in self._srf[self._prefixb+str(blockid)].keys()


def save_grid(self, gridnodes, blockid=0):
    """Save the grid nodes.

    :param blockid: The ID of the data block to operate on.
    """
    path = "/"+self._prefixb+str(blockid)+"/grid"
    self._srf[path][...] = np.real(gridnodes)


def load_grid(self, blockid=0):
    """Load the grid nodes.

    :param blockid: The ID of the data block to operate on.
    """
    path = "/"+self._prefixb+str(blockid)+"/grid"
    return np.squeeze(self._srf[path])
