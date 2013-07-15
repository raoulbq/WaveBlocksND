"""The WaveBlocks Project

This class abstracts compatibility conditions on nested inner products.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["InnerProductCompatibility"]


class InnerProductCompatibility(object):
    r"""This class abstracts compatibility conditions on nested inner products.
    """

    def get_kind(self):
        return None

    def require_kind(self):
        return None

    def compatible(self, ipouter, ipinner):
        r"""
        """
        inner = set(ipinner.get_kind())
        outer = set(ipouter.require_kind())

        if not len(outer.intersection(inner)) == 0:
            return True
        else:
            raise ValueError("Can not nest inner product with kind "+str(inner)+" into inner product which requires "+str(outer))
