"""The WaveBlocks Project

This file contains a simple factory method for Grid instances.
The exact subtype of the instance is read from the description.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

def create_grid(description):
    """The method that creates a :py:class:`Grid` instance and decides
    which subclass to instantiate depending on the given description.

    :param description: A ``description`` (``dict`` or :py:class:`ParameterProvider` instance)
                        with all necessary parameters.
    :return: An adequate :py:class:`Grid` instance.
    """
    if description.has_key("grid_type"):
        grid_type = description["type"]
    else:
        grid_type = "tensor_product"

    if grid_type == "tensor_product":
        from WaveBlocksND import TensorProductGrid
        limits = description["limits"]
        number_nodes = description["number_nodes"]

        # TODO: Improve for one|multiple values: limits = "D*(a,b)" || "[(a1,b1), (a2,b2), ...]"

        grid = TensorProductGrid(limits, number_nodes)

    return grid
