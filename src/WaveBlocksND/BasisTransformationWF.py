r"""The WaveBlocks Project

This file contains the class for basis transformations of wavefunctions
between the canonical basis and the basis spanned by the eigenvectors
of the potential.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import numpy

from BasisTransformation import BasisTransformation

__all__ = ["BasisTransformationWF"]


class BasisTransformationWF(BasisTransformation):
    r"""This class implements basis transformations of wavefunctions :math:`\psi(x)`
    between the canonical basis of and the basis :math:`\Lambda(x)` spanned
    by the eigenvectors :math:`\nu_i(x)` of the potential :math:`V(x)`.
    """

    def __init__(self, potential, grid=None):
        r"""Create a new :py:class:`BasisTransformation` instance for a given potential
        matrix :math:`V(x)`.

        :param potential: The potential underlying the basis transformation.
        :type potential: A :py:class:`MatrixPotential` instance.
        :param grid: The grid.
        :type grid: A :py:class:`Grid` subclass instance.
        """
        # Keep a reference to the potential
        self._potential = potential

        # Precompute eigenvectors is case it is necessary
        self._potential.calculate_eigenvectors()

        if grid is not None:
            self.set_grid(grid)


    def set_grid(self, grid):
        r"""Set the grid :math:`\Gamma` containing the nodes :math:`\gamma` on
        which the wavefunction :math:`\psi` was evaluated. The :math:`N` eigenvectors
        :math:`\nu_i` will be evaluated on the same grid nodes.

        :param grid: The grid
        :type grid: A :py:class:`Grid` subclass instance.
        """
        # Keep a reference to the grid
        self._grid = grid

        # Evaluate the eigenvectors of the potential
        self._ev = self._potential.evaluate_eigenvectors_at(grid)


    def transform_to_canonical(self, wavefunction):
        r"""Transform the evaluated wavefunction :math:`\psi(\Gamma)` given
        in the eigenbasis to the canonical basis.

        :param wavefunction: The wavefunction to transform.
        :type wavefunction: A :py:class:`WaveFunction` instance.
        :return: Another :py:class:`WaveFunction` instance containing the
                 transformed wavefunction :math:`\psi^\prime(\Gamma)`.
        """
        # No transformation for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if self._potential.get_number_components() == 1:
            return

        # Unpack the values from the wavefunction
        values = wavefunction.get_values()

        # Reshape to match eigenvectores
        oshapes = [ value.shape for value in values ]
        values = [ value.flatten() for value in values ]

        # The basis transformation, essentially a matrix multiplication
        N = wavefunction.get_number_components()
        result = []
        for i in xrange(N):
            tmp = numpy.zeros(values[0].shape, dtype=numpy.complexfloating)
            for j in xrange(N):
                tmp += self._ev[j][i,:] * values[j]
            result.append(tmp)

        # Reshape back
        result = [ item.reshape(shape) for item, shape in zip(result, oshapes) ]

        # Pack values back to wavefunction object
        wavefunction.set_values(result)


    def transform_to_eigen(self, wavefunction):
        r"""Transform the evaluated wavefunction :math:`\psi^\prime(\Gamma)` given
        in the canonical basis to the eigenbasis.

        :param wavefunction: The wavefunction to transform.
        :type wavefunction: A :py:class:`WaveFunction` instance.
        :return: Another :py:class:`WaveFunction` instance containing the
                 transformed wavefunction :math:`\psi(\Gamma)`.
        """
        # No transformation for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if self._potential.get_number_components() == 1:
            return

        # Unpack the values from the wavefunction
        values = wavefunction.get_values()

        # Reshape to match eigenvectores
        oshapes = [ value.shape for value in values ]
        values = [ value.flatten() for value in values ]

        # The basis transformation, essentially a matrix multiplication
        N = wavefunction.get_number_components()
        result = []
        for i in xrange(N):
            tmp = numpy.zeros(values[0].shape, dtype=numpy.complexfloating)
            for j in xrange(N):
                tmp += self._ev[i][j,:] * values[j]
            result.append(tmp)

        # Reshape back
        result = [ item.reshape(shape) for item, shape in zip(result, oshapes) ]

        # Pack values back to wavefunction object
        wavefunction.set_values(result)
