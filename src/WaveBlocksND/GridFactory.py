"""The WaveBlocks Project

This file contains a simple factory for Grid instances. The exact
subtype of the instance is read from the parameter provider.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocksND import TensorProductGrid

__all__ = ["GridFactory"]


class GridFactory(object):
    """A factory for :py:class:`Grid` and subclass instances.
    """

    def __init__(self):
        pass


    def create_grid(self, parameters):
        """The method that creates a :py:class:`Grid` instance and decides
        which subclass to instantiate depending on the given parameters.

        :param parameters: A :py:class:`ParameterProvider` instance with all necessary parameters.
        :return: An adequate :py:class:`Grid` instance.
        """
        if parameters.has_key("grid_type"):
            grid_type = parameters["grid_type"]
        else:
            grid_type = "tensor_product"

        if grid_type == "tensor_product":

            limits = parameters["grid_limits"]
            number_nodes = parameters["grid_number_nodes"]

            # TODO: Improve for one|multiple values: limits = "D*(a,b)" || "[(a1,b1), (a2,b2), ...]"

            grid = TensorProductGrid(limits, number_nodes)

        return grid
