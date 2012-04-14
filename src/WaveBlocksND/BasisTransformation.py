r"""The WaveBlocks Project

This file contains the abstract base class for basis transformations
between the canonical basis and the basis spanned by the eigenvectors
of the potential.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["BasisTransformation"]


class BasisTransformation(object):
    r"""This class is the interface definition for general basis transformation
    procedures. The transformation switches between the canonical basis of the
    potential :math:`V(x)` and its eigenbasis :math:`\Lambda(x)` consisting
    of the energy levels :math:`\lambda_i(x)` with :math:`i \in [0, \ldots, N]`.
    """

    def __init__(self, potential):
        r"""Create a new :py:class:`BasisTransformation` instance for a given potential
        matrix :math:`V(x)`.

        :param potential: The potential underlying the basis transformation.
        :type potential: A :py:class:`MatrixPotential` instance.
        """
        # Keep a reference to the potential
        self._potential = potential


    def transform_to_canonical(self, transformable):
        """Do nothing, implement an identity transformation.
        """
        return transformable


    def transform_to_eigen(self, transformable):
        """Do nothing, implement an identity transformation.
        """
        return transformable
