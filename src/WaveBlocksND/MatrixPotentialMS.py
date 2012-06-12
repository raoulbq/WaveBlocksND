r"""The WaveBlocks Project

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
from Grid import Grid
from GridWrapper import GridWrapper

__all__ = ["MatrixPotentialMS"]


class MatrixPotentialMS(MatrixPotential):
    r"""This class represents a matrix potential :math:`V(x)`. The potential is
    given as an analytic :math:`N \times N` matrix expression. All methods use
    pure numerical techniques because symbolical calculations are unfeasible
    for 3 or more energy levels.
    """

    def __init__(self, expression, variables, **kwargs):
        r"""Create a new :py:class:`MatrixPotentialMS` instance for a given
        potential matrix :math:`V(x)`.

        :param expression:The mathematical expression representing the potential.
        :type expressiom: A `Sympy` matrix type.
        :param variables: The variables corresponding to the space dimensions.
        :type variables: A list of `Sympy` symbols.
        """
        # The variables that represents position space. The order matters!
        self._variables = variables

        # The dimension of position space.
        self._dimension = len(variables)

        # This number of energy levels.
        assert expression.is_square
        # We handle the general NxN case here
        self._number_components = expression.shape[0]

        # The the potential, symbolic expressions and evaluatable functions
        self._potential_s = expression
        self._potential_n = tuple([ sympy.lambdify(self._variables, entry, "numpy") for entry in self._potential_s ])

        # The Jacobian and Hessian matrices of all entries of V
        self._JV_s = None
        self._JV_n = None
        self._HV_s = None
        self._HV_n = None


    def _grid_wrap(self, grid):
        # TODO: Consider additional input types for "nodes":
        #       list of numpy ndarrays, list of single python scalars
        if not isinstance(grid, Grid):
            if not type(grid) is numpy.ndarray:
                grid = numpy.atleast_2d(grid)
            grid = GridWrapper(grid)
        return grid


    def evaluate_at(self, grid, entry=None, as_matrix=True):
        r"""Evaluate the potential :math:`V(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the potential at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param entry: The indices :math:`(i,j)` of the component :math:`V_{i,j}(x)`
                      we want to evaluate or `None` to evaluate all entries.
        :type entry: A python tuple of two integers.
        :param as_matrix: Dummy parameter which has no effect here.
        :return: A list containing :math:`N^2` numpy ndarrays of shape :math:`(1, |\Gamma|)`.
        """
        grid = self._grid_wrap(grid)

        # Determine which entries to evaluate
        if entry is not None:
            (row, col) = entry
            entries = [ row * self._number_components + col ]
        else:
            N = self._number_components
            entries = [ row * N + col for row in xrange(N) for col in xrange(N) ]

        # Evaluate all entries specified
        result = []

        nodes = grid.get_nodes(split=True)

        for index in entries:
            # Evaluate the potential at the given nodes
            values = self._potential_n[index](*nodes)

            # Test for potential beeing constant
            if numpy.atleast_1d(values).shape == (1,):
                values = values * numpy.ones(grid.get_number_nodes(), dtype=numpy.complexfloating)

            # Put the result in correct shape (1, #gridnodes)
            N = grid.get_number_nodes(overall=True)
            result.append( values.reshape((1,N)) )

        # TODO: Consider unpacking single ndarray iff entry != None
        if entry is not None:
            result = result[0]

        return tuple(result)


    def calculate_eigenvalues(self):
        r"""Calculate all the eigenvalues :math:`\lambda_i(x)` of the potential :math:`V(x)`.
        We can not do this by symbolic calculations, hence the function has an empty
        implementation. We compute the eigenvalues by numercial techniques in the corresponding
        `evaluate_eigenvalues_at` function.
        """
        pass


    def evaluate_eigenvalues_at(self, grid, entry=None, as_matrix=False, sorted=True):
        r"""Evaluate the eigenvalues :math:`\lambda_i(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvalues at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param entry: The indices :math:`(i,j)` of the component :math:`\Lambda_{i,j}(x)`
                      we want to evaluate or `None` to evaluate all entries. If :math:`j = i`
                      then we evaluate the eigenvalue :math:`\lambda_i(x)`.
        :type entry: A python tuple of two integers.
        :param as_matrix: Whether to include the off-diagonal zero entries of
                          :math:`\Lambda_{i,j}(x)` in the return value.
        :return: A list containing the numpy ndarrays, all of shape :math:`(1, |\Gamma|)`.
        """
        grid = self._grid_wrap(grid)

        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        # Early return shortcut for off-diagonal entries
        if entry is not None:
            (row, col) = entry
            if row != col:
                return numpy.zeros((1, n), dtype=numpy.complexfloating)

        # Memory for storing temporary values
        tmppot = numpy.ndarray((n, N, N), dtype=numpy.complexfloating)
        tmpew = numpy.ndarray((n, N), dtype=numpy.complexfloating)

        # Evaluate potential
        values = self.evaluate_at(grid)

        # Fill in values
        for row in xrange(N):
            for col in xrange(N):
                tmppot[:, row, col] = values[N*row + col]

        # Calculate eigenvalues assuming hermitian matrix (eigvalsh for stability!)
        for i in xrange(n):
            ew = linalg.eigvalsh(tmppot[i,:,:])
            if sorted is True:
                # Sorting the eigenvalues, biggest first.
                # TODO: Sort will fail iff energy level cross!
                ew.sort()
                tmpew[i,:] = ew[::-1]
            else:
                # Do not sort
                tmpew[i,:] = ew[:]

        # Split the data into different eigenvalues
        tmp = [ tmpew[:,index].reshape((1,n)) for index in xrange(N) ]

        if entry is not None:
            (row, col) = entry
            result = tmp[row]
            # Offdiagonal case handled on top
        elif as_matrix is True:
            result = []
            for row in xrange(N):
                for col in xrange(N):
                    if row == col:
                        result.append(tmp[row])
                    else:
                        result.append(numpy.zeros(tmp[row].shape, dtype=numpy.complexfloating))
        else:
            result = tuple(tmp)

        return result


    def calculate_eigenvectors(self):
        r"""Calculate all the eigenvectors :math:`\nu_i(x)` of the potential :math:`V(x)`.
        We can not do this by symbolic calculations, hence the function has an empty
        implementation. We compute the eigenvectors by numercial techniques in the corresponding
        `evaluate_eigenvectors_at` function.
        """
        pass


    def evaluate_eigenvectors_at(self, grid, sorted=True):
        r"""Evaluate the eigenvectors :math:`\nu_i(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvectors at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :return: A list containing the :math:`N` numpy ndarrays, all of shape :math:`(D, |\Gamma|)`.
        """
        grid = self._grid_wrap(grid)

        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        # Memory for storing temporary values
        tmppot = numpy.ndarray((n, N, N), dtype=numpy.complexfloating)
        tmpev = numpy.ndarray((n, N, N), dtype=numpy.complexfloating)

        # Evaluate potential
        values = self.evaluate_at(grid)

        # Fill in values
        for row in xrange(N):
            for col in xrange(N):
                tmppot[:, row, col] = values[N*row + col]

        # Calculate eigenvectors assuming hermitian matrix (eigh for stability!)
        for i in xrange(0, n):
            ew, ev = linalg.eigh(tmppot[i,:,:])
            if sorted is True:
                # Sorting the eigenvectors in the same order as the eigenvalues.
                ind = numpy.argsort(ew)
                ind = ind[::-1]
                evs = ev[:,ind]
                tmpev[i,:,:] = evs
            else:
                # No sorting
                tmpev[i,:,:] = ev

        # A trick due to G. Hagedorn to get continuous eigenvectors
        # TODO: Not sure if it works in higher dimensions too! (Probably it does not)
        for i in xrange(1,n):
            for ev in xrange(0,N):
                if numpy.dot(tmpev[i,:,ev],tmpev[i-1,:,ev]) < 0:
                    tmpev[i,:,ev] *= -1

        return tuple([ numpy.transpose(tmpev[:,:,index]) for index in xrange(N) ])


    def calculate_exponential(self, factor=1):
        r"""Calculate the matrix exponential :math:`\exp(\alpha V)`. In the case
        of this class the matrix is of size :math:`N \times N` thus the exponential
        can not be calculated analytically for a general matrix. We use numerical
        approximations to determine the matrix exponential. We just store
        the prefactor :math:`\alpha` for use during numerical evaluation.

        :param factor: The prefactor :math:`\alpha` in the exponential.
        """
        # Store the factor for later numerical computations.
        self._factor = factor


    def evaluate_exponential_at(self, grid):
        r"""Evaluate the exponential of the potential matrix :math:`V(x)` on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the exponential at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :return: The numerical approximation of the matrix exponential at the given grid nodes.
                 A list contains the exponentials for all entries :math:`(i,j)`, each having
                 a shape of :math:`(1, |\Gamma|)`.
        """
        grid = self._grid_wrap(grid)

        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        # Memory for storing temporary values
        tmp = numpy.ndarray((n,N,N), dtype=numpy.complexfloating)

        # Evaluate potential
        values = self.evaluate_at(grid)

        # Fill in values
        for row in xrange(N):
            for col in xrange(N):
                tmp[:, row, col] = self._factor * values[N*row + col]

        # Calculate exponential
        for i in xrange(n):
            tmp[i,:,:] = linalg.expm(tmp[i,:,:], 10)

        # Split the data into different components
        return tuple([ tmp[:,row,col].reshape((1,n)) for row in xrange(N) for col in xrange(N) ])


    def _calculate_jacobian_of_matrix(self, entry=None):
        r"""Compute the Jacobian of the matrix elements :math:`V_{i,j}`.
        """
        if self._JV_s is not None:
            return

        self._JV_s = {}
        self._JV_n = {}

        for i, variable in enumerate(self._variables):
            self._JV_s[i] = tuple([ sympy.diff(entry, variable) for entry in self._potential_s ])

        for k, v in self._JV_s.iteritems():
            self._JV_n[k] = tuple([ sympy.lambdify(self._variables, entry, "numpy") for entry in v ])


    def _calculate_hessian_of_matrix(self, entry=None):
        r"""Compute the Hessian of the matrix elements :math:`V_{i,j}`.
        """
        if self._HV_s is not None:
            return

        self._HV_s = {}
        self._HV_n = {}

        for i, variable1 in enumerate(self._variables):
            for j, variable2 in enumerate(self._variables):
                self._HV_s[(i,j)] = tuple([ sympy.diff(sympy.diff(entry, variable1), variable2)  for entry in self._potential_s ])

        for key, val in self._HV_s.iteritems():
            self._HV_n[key] = tuple([ sympy.lambdify(self._variables, entry, "numpy") for entry in val ])


    def _evaluate_jacobian_of_matrix(self, variable, grid, entry=None):
        # Note: We assume grid is already of supertype Grid
        #issubclass(type(grid), Grid)
        n = grid.get_number_nodes(overall=True)
        N = self._number_components
        nodes = grid.get_nodes(split=True)

        dAdxk = numpy.zeros((n, N,N), dtype=numpy.complexfloating)
        for row in xrange(N):
            for col in xrange(N):
                dAdxk[:,row,col] = self._JV_n[variable][N*row+col](*nodes)

        return dAdxk


    def _evaluate_hessian_of_matrix(self, variables, grid, entry=None):
        # Note: We assume grid is already of supertype Grid
        #issubclass(type(grid), Grid)
        n = grid.get_number_nodes(overall=True)
        N = self._number_components
        nodes = grid.get_nodes(split=True)

        dAdxidxj = numpy.zeros((n, N,N), dtype=numpy.complexfloating)
        for row in xrange(N):
            for col in xrange(N):
                dAdxidxj[:,row,col] = self._HV_n[variables][N*row+col](*nodes)

        return dAdxidxj


    def calculate_jacobian(self):
        r"""Calculate the Jacobian matrix :math:`\nabla \lambda_i` of the potential's
        eigenvalues :math:`\Lambda(x)` with :math:`x \in \mathbb{R}^D`. For potentials
        which depend only one variable, this equals the first derivative and :math:`D=1`.
        Note that this function is idempotent.
        """
        self._calculate_jacobian_of_matrix()


    def evaluate_jacobian_at(self, grid, component=None):
        r"""Evaluate the list of Jacobian matrices :math:`\nabla \lambda_i(x)` at some grid
        nodes :math:`\Gamma` for one or all eigenvalues.

        :param grid: The grid nodes :math:`\Gamma` the Jacobian gets evaluated at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param component: The index :math:`i` of the eigenvalue :math:`\lambda_i`.
        :return: The value of the potential's Jacobian at the given nodes. The result
                 is a list of ``ndarray`` each of shape :math:`(D,1)` is we evaluate
                 at a single grid node or of shape :math:`(D,|\Gamma|)`
                 if we evaluate at multiple nodes simultaneously.
        """
        grid = self._grid_wrap(grid)

        D = self._dimension
        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        # For which eigenvalues do we need to do the computation
        if component is None:
            levels = xrange(N)
        else:
            levels = [component]

        nodes = grid.get_nodes(split=True)

        # Compute eigenvectors
        EV = self.evaluate_eigenvectors_at(grid, sorted=False)

        Jn = []
        # For each eigenvalue
        for l in levels:
            Jl = numpy.zeros((D, n), dtype=numpy.complexfloating)
            # For each variable xi
            for i in xrange(D):
                dAdxi = self._evaluate_jacobian_of_matrix(i, grid)
                # TODO: Adapt to non-real eigenvectors by conjugating first EV[l]
                Jl[i,:] = numpy.einsum("j...,...jk,k...", numpy.conjugate(EV[l]), dAdxi, EV[l])

            Jn.append(Jl)

        if component is not None:
            # Unpack single item
            return Jn[0]
        else:
            return tuple(Jn)


    def calculate_hessian(self):
        r"""Calculate the Hessian matrix :math:`\nabla^2 \lambda_i` of the potential's
        eigenvalues :math:`\Lambda(x)` with :math:`x \in \mathbb{R}^D`. For potentials
        which depend only one variable, this equals the second derivative and :math:`D=1`.
        Note that this function is idempotent.
        """
        self._calculate_hessian_of_matrix()


    def evaluate_hessian_at(self, grid, component=None):
        r"""Evaluate the list of Hessian matrices :math:`\nabla^2 \lambda_i(x)` at some grid
        nodes :math:`\Gamma` for one or all eigenvalues.

        :param grid: The grid nodes :math:`\Gamma` the Hessian gets evaluated at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param component: The index :math:`i` of the eigenvalue :math:`\lambda_i`.
        :return: The value of the potential's Hessian at the given nodes. The result
                 is an ``ndarray`` of shape :math:`(D,D)` is we evaluate at a single
                 grid node or of shape :math:`(D,D,|\Gamma|)` if we evaluate at multiple
                 nodes simultaneously.
        """
        grid = self._grid_wrap(grid)

        D = self._dimension
        N = self._number_components
        n = grid.get_number_nodes(overall=True)

        nodes = grid.get_nodes(split=True)

        # For which eigenvalues do we need to do the computation
        if component is None:
            levels = xrange(N)
        else:
            levels = [component]

        # Compute eigenvalues
        EW = self.evaluate_eigenvalues_at(grid, sorted=False)

        # Compute eigenvectors
        EV = self.evaluate_eigenvectors_at(grid, sorted=False)

        Hn = []
        # For each eigenvalue
        for l in levels:
            Hl = numpy.zeros((D,D, n), dtype=numpy.complexfloating)

            # For all variable pairs (xi, xj)
            for i in xrange(D):
                for j in xrange(D):
                    # First term
                    dAdxidxj = self._evaluate_hessian_of_matrix((i,j), grid)
                    # TODO: Adapt to non-real eigenvectors by conjugating first EV[l]
                    Hl[i,j] = numpy.einsum("j...,...jk,k...", numpy.conjugate(EV[l]), dAdxidxj, EV[l])

                    # Second terms
                    tmp = numpy.zeros((n,), dtype=numpy.complexfloating)
                    for k in xrange(N):
                        if k != l:
                            # TODO: Pull these out of the i/j loops?
                            dAdxi = self._evaluate_jacobian_of_matrix(i, grid)
                            dAdxj = self._evaluate_jacobian_of_matrix(j, grid)

                            # TODO: Adapt to non-real eigenvectors by conjugating first EV[l]
                            factor1 = numpy.einsum("j...,...jk,k...", numpy.conjugate(EV[l]), dAdxi, EV[k])
                            factor2 = numpy.einsum("j...,...jk,k...", numpy.conjugate(EV[l]), dAdxj, EV[k])
                            tmp = tmp + factor1*factor2 / (EW[l]-EW[k])

                    Hl[i,j,:] = Hl[i,j,:] + 2*tmp

            if n == 1:
                Hl = Hl.reshape((D,D))
            Hn.append(Hl)

        if component is not None:
            # Unpack single item
            return Hn[0]
        else:
            return tuple(Hn)


    def calculate_local_quadratic(self, diagonal_component=None):
        r"""Calculate the local quadratic approximation matrix :math:`U(x)` of the potential's
        eigenvalues in :math:`\Lambda(x)`. This function can be used for the homogeneous case
        and takes into account the leading component :math:`\chi \in [0,\ldots,N-1]`.
        If the parameter :math:`i` is not given, calculate the local quadratic approximation
        matrix :math:`U(x)` of all the potential's eigenvalues in :math:`\Lambda`. This case
        can be used for the inhomogeneous case.

        :param diagonal_component: Dummy parameter which has no effect here.
        """
        self._calculate_jacobian_of_matrix()
        self._calculate_hessian_of_matrix()


    def evaluate_local_quadratic_at(self, grid, diagonal_component=None):
        r"""Numerically evaluate the local quadratic approximation matrix :math:`U(x)` of
        the potential's eigenvalues in :math:`\Lambda(x)` at the given grid nodes :math:`\Gamma`.

        :param grid: The grid :math:`\Gamma` containing the nodes :math:`\gamma` we want to
                     evaluate the quadratic approximation at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param diagonal_component: Specifies the index :math:`i` of the eigenvalue :math:`\lambda_i`
                                   that gets expanded into a Taylor series :math:`u_i`.
        :return: A list of tuples or a single tuple. Each tuple :math:`(\lambda, J, H)` contains the
                 the evaluated eigenvalue :math:`\lambda_i(\Gamma)`, its Jacobian :math:`J(\Gamma)`
                 and its Hessian :math:`H(\Gamma)` in this order.
        """
        if diagonal_component is not None:
            V = self.evaluate_eigenvalues_at(grid, entry=(diagonal_component,diagonal_component))
            J = self.evaluate_jacobian_at(grid, component=diagonal_component)
            H = self.evaluate_hessian_at(grid, component=diagonal_component)
            result = (V, J, H)
        else:
            Vlist = self.evaluate_eigenvalues_at(grid)
            Jlist = self.evaluate_jacobian_at(grid)
            Hlist = self.evaluate_hessian_at(grid)
            result = [ (V, J, H) for V, J, H in zip(Vlist, Jlist, Hlist) ]

        return tuple(result)


    def calculate_local_remainder(self, diagonal_component=None):
        r"""Calculate the non-quadratic remainder matrix :math:`W(x) = V(x) - U(x)` of the
        quadratic approximation matrix :math:`U(x)` of the potential's eigenvalue matrix
        :math:`\Lambda(x)`. In the homogeneous case the matrix :math:`U` is given by
        :math:`U(x) = \text{diag}([u_i,\ldots,u_i])` where in the inhomogeneous case it
        is given by :math:`U(x) = \text{diag}([u_0,\ldots,u_{N-1}])`.

        :param diagonal_component: Specifies the index :math:`i` of the eigenvalue :math:`\lambda_i`
                                   that gets expanded into a Taylor series :math:`u_i`. If set to
                                   ``None`` the inhomogeneous case is computed.
        :type diagonal_component: Integer or ``None`` (default)
        """
        self._calculate_jacobian_of_matrix()
        self._calculate_hessian_of_matrix()


    def evaluate_local_remainder_at(self, grid, position, diagonal_component=None, entry=None):
        r"""Numerically evaluate the non-quadratic remainder :math:`W(x)` of the quadratic
        approximation :math:`U(x)` of the potential's eigenvalue :math:`\Lambda(x)` at the
        given nodes :math:`\Gamma`.

         Warning: do not set the ``diagonal_component`` and the ``entry`` parameter both to ``None``.

        :param grid: The grid nodes :math:`\Gamma` the remainder :math:`W` gets evaluated at.
        :param position: The point :math:`q \in \mathbb{R}^D` where the Taylor series is computed.
        :param diagonal_component: Specifies the index :math:`i` of the eigenvalue :math:`\lambda_i`
                                   that gets expanded into a Taylor series :math:`u_i` and whose
                                   remainder matrix :math:`W(x) = V(x) - \text{diag}([u_i,\ldots,u_i])`
                                   we evaluate. If set to ``None`` the inhomogeneous case given by
                                   :math:`W(x) = V(x) - \text{diag}([u_0,\ldots,u_{N-1}])` is computed.
        :type diagonal_component: Integer or ``None`` (default)
        :param entry: The entry :math:`\left(i,j\right)` of the remainder matrix :math:`W`
                      that is evaluated.
        :type entry: A python tuple of two integers.
        :return: A list with :math:`N^2` ``ndarray`` elements or a single ``ndarray``. Each
                 containing the values of :math:`W_{i,j}(\Gamma)`. Each array is of shape
                 :math:`(1,|\Gamma|)`.
        """
        grid = self._grid_wrap(grid)
        nodes = grid.get_nodes()
        N = self._number_components

        if entry is not None:
            rows = [entry[0]]
            cols = [entry[1]]
        else:
            rows = xrange(N)
            cols = xrange(N)

        W = []

        if diagonal_component is not None:
            # Homogeneous case
            V = self.evaluate_at(grid)
            L, J, H = self.evaluate_local_quadratic_at(position, diagonal_component=diagonal_component)

            # Compute quadratic approximation
            # L(q) + J(q)*(G-q) + 1/2*(G-q)T*H(q)*(G-q)
            df = nodes - position
            U = L + numpy.einsum("i...,i...", J, df) + 0.5*numpy.einsum("j...,jk...,k...", df, H, df)

            # Compute the remainder W = V - U
            for row in rows:
                for col in cols:
                    if row == col:
                        W.append(V[row*N+col] - U)
                    else:
                        W.append(V[row*N+col])
        else:
            # Inhomogeneous case
            V = self.evaluate_at(grid)
            # Compute the remainder W = V - U
            for row in rows:
                L, J, H = self.evaluate_local_quadratic_at(position, diagonal_component=row)
                # Compute quadratic approximation
                # L(q) + J(q)*(G-q) + 1/2*(G-q)T*H(q)*(G-q)
                df = nodes - position
                U = L + numpy.einsum("i...,i...", J, df) + 0.5*numpy.einsum("j...,jk...,k...", df, H, df)

                for col in cols:

                    if row == col:
                        W.append(V[row*N+col] - U)
                    else:
                        W.append(V[row*N+col])

        if entry is not None:
            # Unpack single item
            return W[0]
        else:
            return tuple(W)
