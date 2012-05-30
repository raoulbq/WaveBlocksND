"""The WaveBlocks Project

This file contains code for the representation of potentials :math:`V(x)`
that contain three or more energy levels :math:`\lambda_i`. (In principle
the code works also with 1 or 2 levels, but it is not used that way.)
The number of space dimensions can be arbitrary, :math:`x \in \mathbb{R}^D`.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sympy
import numpy
from scipy import linalg

from MatrixPotential import MatrixPotential

__all__ = ["MatrixPotentialMS"]


class MatrixPotentialMS(MatrixPotential):
    """This class represents a matrix potential :math:`V(x)`. The potential is
    given as an analytic :math:`N \times N` matrix expression. All methods use
    pure numerical techniques because symbolical calculations are unfeasible
    for 3 or more energy levels.
    """

    def __init__(self, expression, variables, **kwargs):
        """Create a new :py:class:`MatrixPotentialMS` instance for a given
        potential matrix :math:`V(x)`.

        :param expression:The mathematical expression representing the potential.
        :type expressiom: A `Sympy` matrix type.
        :param variables: The variables corresponding to the space dimensions.
        :type variables: A list of `Sympy` symbols.
        """
        # This class handles potentials with a single energy level.
        self._number_components = expression.shape[0]

        # The variables that represents position space. The order matters!
        self._variables = variables

        # The dimension of position space.
        self._dimension = len(variables)

        # Find active variables and constants
        self._variables = []
        self._constants = []
        self._isconstant = []

        for entry in expression:
            # Determine the variables and constants for each component
            local_vars = [ v for v in entry.atoms(sympy.Symbol) ]
            local_consts = [ c for c in variables if not c in local_vars ]
            self._variables.append(local_vars)
            self._constants.append(local_consts)

            # Does the potential depend on all space variables?
            # True:  V_i,j is constant along all axes
            # False: V_i,j is not constant
            # None:  V_i,j is constant along some axes
            if len(local_consts) == 0:
                self._isconstant.append(False)
            elif len(local_vars) == 0:
                self._isconstant.append(True)
            else:
                self._isconstant.append(None)

        # The the potential, symbolic expressions and evaluatable functions
        self._potential_s = expression
        self._potential_n = tuple([ sympy.lambdify(self._variables, entry, "numpy") for entry in self._potential_s ])

        # The cached eigenvalues evaluatable functions
        self._eigenvalues_n = None

        # The cached eigenvectors evaluatable functions
        self._eigenvectors_n = None

        # The cached exponential evaluatable functions
        self._exponential_n = None


    def evaluate_at(self, grid, entry=None, as_matrix=True):
        """Evaluate the potential :math:`V(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the potential at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param entry: The indices :math:`(i,j)` of the component :math:`V_{i,j}(x)`
                      we want to evaluate or `None` to evaluate all entries.
        :type entry: A python tuple of two integers.
        :param as_matrix: Dummy parameter which has no effect here.
        :return: A list containing :math:`N^2` numpy ndarrays of shape :math:`(N_1, ... ,N_D)`.
        """
        if entry is not None:
            (row, col) = entry
            items = [ row * self._number_components + col ]
        else:
            N = self._number_components
            items = [ row * N + col for row in xrange(N) for col in xrange(N) ]

        # Evaluate all entries specified
        result = []

        for item in items:
            if self._isconstant[item] is False:
                # We can use numpy broadcasting
                # TODO: This only works with tensor product grids
                result.append( self._potential_n[item](*grid.get_axes()) )
            elif self._isconstant[item] is None:
                # Make sure we work with (N_1, ..., N_D) shaped arrays
                result.append( self._potential_n[item](*grid.get_nodes(split=True, flat=False)) )
            elif self._isconstant[item] is True:
                values = self._potential_n[item](*grid.get_axes())
                result.append( values * numpy.ones(grid.get_number_nodes(), dtype=numpy.complexfloating) )

        # TODO: Consider unpacking single ndarray iff entry != None
        return tuple(result)


    def calculate_eigenvalues(self):
        """Calculate all the eigenvalues :math:`\lambda_i(x)` of the potential :math:`V(x)`.
        We can not do this by symbolic calculations, hence the function has an empty
        implementation. We compute the eigenvalues by numercial techniques in the corresponding
        `evaluate_eigenvalues_at` function.
        """
        pass


    def evaluate_eigenvalues_at(self, grid, entry=None, as_matrix=False):
        """Evaluate the eigenvalues :math:`\lambda_i(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvalues at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param entry: The indices :math:`(i,j)` of the component :math:`\Lambda_{i,j}(x)`
                      we want to evaluate or `None` to evaluate all entries. If :math:`j = i`
                      then we evaluate the eigenvalue :math:`\lambda_i(x)`.
        :type entry: A python tuple of two integers.
        :param as_matrix: Whether to include the off-diagonal zero entries of
                          :math:`\Lambda_{i,j}(x)` in the return value.
        :return: A list containing the numpy ndarrays, all of shape :math:`(N_1, ... ,N_D)`.
        """
        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        # Memory for storing temporary values
        tmppot = numpy.ndarray((n, N, N), dtype=numpy.complexfloating)
        tmpew = numpy.ndarray((n, N), dtype=numpy.complexfloating)

        # Evaluate potential
        values = self.evaluate_at(grid)
        values = [ value.flatten() for value in values ]

        # Fill in values
        for row in xrange(N):
            for col in xrange(N):
                tmppot[:, row, col] = values[N*row + col]

        # Calculate eigenvalues assuming hermitian matrix (eigvalsh for stability!)
        for i in xrange(n):
            ew = linalg.eigvalsh(tmppot[i,:,:])
            # Sorting the eigenvalues biggest first.
            # TODO: Sort will fail iff energy level cross!
            ew.sort()
            tmpew[i,:] = ew[::-1]

        # Split the data into different eigenvalues
        shape = grid.get_number_nodes()
        tmp = [ tmpew[:,index].reshape(shape) for index in xrange(N) ]

        if entry is not None:
            (row, col) = entry
            if row == col:
                result = tmp[row]
            else:
                result = numpy.zeros(tmp[row].shape, dtype=numpy.complexfloating)
        elif as_matrix is True:
            result = []
            for row in xrange(N):
                for col in xrange(N):
                    if row == col:
                        result.append(tmp[row])
                    else:
                        result.append(numpy.zeros(tmp[row].shape, dtype=numpy.complexfloating))
        else:
            result = tmp

        return result


    def calculate_eigenvectors(self):
        """Calculate all the eigenvectors :math:`\nu_i(x)` of the potential :math:`V(x)`.
        We can not do this by symbolic calculations, hence the function has an empty
        implementation. We compute the eigenvectors by numercial techniques in the corresponding
        `evaluate_eigenvectors_at` function.
        """
        pass


    def evaluate_eigenvectors_at(self, grid):
        """Evaluate the eigenvectors :math:`\nu_i(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvectors at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :return: A list containing the numpy ndarrays, all of shape :math:`(N_1, ... ,N_D, 1)`.
        """
        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        # Memory for storing temporary values
        tmppot = numpy.ndarray((n, N, N), dtype=numpy.complexfloating)
        tmpev = numpy.ndarray((n, N, N), dtype=numpy.complexfloating)

        # Evaluate potential
        values = self.evaluate_at(grid)
        values = [ value.flatten() for value in values ]

        # Fill in values
        for row in xrange(N):
            for col in xrange(N):
                tmppot[:, row, col] = values[N*row + col]

        # Calculate eigenvectors assuming hermitian matrix (eigh for stability!)
        for i in xrange(0, n):
            ew, ev = linalg.eigh(tmppot[i,:,:])
            # Sorting the eigenvectors in the same order as the eigenvalues.
            ind = numpy.argsort(ew)
            ind = ind[::-1]
            evs = ev[:,ind]
            tmpev[i,:,:] = evs

        # A trick due to G. Hagedorn to get continuous eigenvectors
        for i in xrange(1, n):
            for ev in xrange(0,N):
                if numpy.dot(tmpev[i,:,ev],tmpev[i-1,:,ev]) < 0:
                    tmpev[i,:,ev] *= -1

        result = tuple([ numpy.transpose(tmpev[:,:,index]) for index in xrange(N) ])
        return result


    def calculate_exponential(self, factor=1):
        """Calculate the matrix exponential :math:`\exp(\alpha V)`. In the case
        of this class the matrix is of size :math:`N \times N` thus the exponential
        can not be calculated analytically for a general matrix. We use numerical
        approximations to determine the matrix exponential.
        Note: This function is idempotent.

        :param factor: The prefactor :math:`\alpha` in the exponential.
        """
        # Store the factor for later numerical computations.
        self._factor = factor


    def evaluate_exponential_at(self, grid):
        """Evaluate the exponential of the potential matrix :math:`V(x)` on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the exponential at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :return: The numerical approximation of the matrix exponential at the given grid nodes.
                 A list contains the exponentials for all entries :math:`(i,j)`, each having the
                 same shape as the grid.
        """
        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        # Memory for storing temporary values
        tmp = numpy.ndarray((n,N,N), dtype=numpy.complexfloating)

        # Evaluate potential
        values = self.evaluate_at(grid)
        values = [ value.flatten() for value in values ]

        # Fill in values
        for row in xrange(N):
            for col in xrange(N):
                tmp[:, row, col] = self._factor * values[N*row + col]

        # Calculate exponential
        for i in xrange(n):
            tmp[i,:,:] = linalg.expm(tmp[i,:,:], 10)

        # Split the data into different components
        shape = grid.get_number_nodes()
        result = [ tmp[:,row,col].reshape(shape) for row in xrange(N) for col in xrange(N) ]
        return tuple(result)
