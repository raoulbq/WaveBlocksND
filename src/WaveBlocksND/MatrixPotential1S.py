r"""The WaveBlocks Project

This file contains code for the representation of potentials :math:`V(x)`
that contain only a single energy level :math:`\lambda_0`. The number
of space dimensions can be arbitrary, :math:`x \in \mathbb{R}^D`.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sympy
import numpy

from MatrixPotential import MatrixPotential
from Grid import Grid
from GridWrapper import GridWrapper

__all__ = ["MatrixPotential1S"]


class MatrixPotential1S(MatrixPotential):
    r"""This class represents a scalar potential :math:`V(x)`. The potential is
    given as an analytic :math:`1 \times 1` matrix expression. Some symbolic
    calculations with the potential are supported.
    """

    def __init__(self, expression, variables):
        r"""Create a new :py:class:`MatrixPotential1S` instance for a given
        potential matrix :math:`V(x)`.

        :param expression:The mathematical expression representing the potential.
        :type expressiom: A `Sympy` matrix type.
        :param variables: The variables corresponding to the space dimensions.
        :type variables: A list of `Sympy` symbols.
        """
        # This class handles potentials with a single energy level.
        self._number_components = 1

        # The variables that represents position space. The order matters!
        self._all_variables = variables

        # The dimension of position space.
        self._dimension = len(variables)

        # The the potential, symbolic expressions and evaluatable functions
        assert expression.shape == (1,1)

        self._potential_s = expression
        self._potential_n = sympy.lambdify(self._all_variables, self._potential_s[0,0], "numpy")

        # The cached eigenvalues, symbolic expressions and evaluatable functions
        self._eigenvalues_s = None
        self._eigenvalues_n = None

        # The cached eigenvectors, symbolic expressions and evaluatable functions
        self._eigenvectors_s = None
        self._eigenvectors_n = None

        # The cached exponential, symbolic expressions and evaluatable functions
        self._exponential_s = None
        self._exponential_n = None

        # The cached Jacobian of the eigenvalues, symbolic expressions and evaluatable functions
        self._jacobian_s = None
        self._jacobian_n = None

        # The cached Hessian of the eigenvalues, symbolic expressions and evaluatable functions
        self._hessian_s = None
        self._hessian_n = None


    def _grid_wrap(self, grid):
        # TODO: Consider additional input types for "nodes":
        #       list of numpy ndarrays, list of single python scalars
        if not isinstance(grid, Grid):
            if not type(grid) is numpy.ndarray:
                grid = numpy.atleast_2d(grid)
            grid = GridWrapper(grid)
        return grid


    def evaluate_at(self, grid, entry=None, as_matrix=False):
        r"""Evaluate the potential :math:`V(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the potential at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param entry: The indices :math:`(i,j)` of the component :math:`V_{i,j}(x)`
                      we want to evaluate or `None` to evaluate all entries.
                      This has no effect here as we only have a single entry :math:`V_{0,0}`.
        :type entry: A python tuple of two integers.
        :param as_matrix: Dummy parameter which has no effect here.
        :return: A list containing a single numpy ``ndarray`` of shape :math:`(1,|\Gamma|)`.
        """
        grid = self._grid_wrap(grid)

        # Evaluate the potential at the given nodes
        values = self._potential_n(*grid.get_nodes(split=True))

        # Test for potential beeing constant
        if numpy.atleast_1d(values).shape == (1,):
            values = values * numpy.ones(grid.get_number_nodes(), dtype=numpy.complexfloating)

        # Put the result in correct shape (1, #gridnodes)
        N = grid.get_number_nodes(overall=True)
        result = [ values.reshape((1,N)) ]

        # TODO: Consider unpacking single ndarray iff entry != None
        if entry is not None:
            result = result[0]

        return result


    def calculate_eigenvalues(self):
        r"""Calculate the eigenvalue :math:`\lambda_0(x)` of the potential :math:`V(x)`.
        In the scalar case this is just equal to the matrix entry :math:`V_{0,0}(x)`.
        Note: This function is idempotent and the eigenvalues are memoized for later reuse.
        """
        if self._eigenvalues_s is not None:
            return

        self._eigenvalues_s = self._potential_s
        self._eigenvalues_n = self._potential_n


    def evaluate_eigenvalues_at(self, grid, entry=None, as_matrix=False):
        r"""Evaluate the eigenvalue :math:`\lambda_0(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvalue at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param entry: The indices :math:`(i,j)` of the component :math:`\Lambda_{i,j}(x)`
                      we want to evaluate or `None` to evaluate all entries. If :math:`j = i`
                      then we evaluate the eigenvalue :math:`\lambda_i(x)`. This has no
                      effect here as we only have a single entry :math:`\lambda_0`.
        :type entry: A python tuple of two integers.
        :param as_matrix: Dummy parameter which has no effect here.
        :return: A list containing a single numpy ndarray of shape :math:`(N_1, ... ,N_D)`.
        """
        # This is the correct solution but we will never use the values
        #if self._eigenvalues_n is None:
        #    self.calculate_eigenvalues()

        # Just evaluate the potential
        return self.evaluate_at(grid, entry=entry, as_matrix=as_matrix)


    def calculate_eigenvectors(self):
        r"""Calculate the eigenvector :math:`\nu_0(x)` of the potential :math:`V(x)`.
        In the scalar case this is just the value :math:`1`.
        Note: This function is idempotent and the eigenvectors are memoized for later reuse.
        """
        # This is the correct solution but we will never use the values
        #self._eigenvectors_s = sympy.Matrix([[1]])
        #self._eigenvectors_n = sympy.lambdify(self._variables, self._eigenvectors_s, "numpy"))
        pass


    def evaluate_eigenvectors_at(self, grid, entry=None):
        r"""Evaluate the eigenvector :math:`\nu_0(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvector at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :param entry: The index :math:`i` of the eigenvector :math:`\nu_i(x)`
                      we want to evaluate or `None` to evaluate all eigenvectors.
                      This has no effect here as we only have a single entry :math:`\nu_0`.
        :type entry: A singly python  integer.
        :return: A list containing the numpy ndarrays, all of shape :math:`(1, |\Gamma|)`.
        """
        # This is the correct solution but we will never use the values
        #if self._eigenvectors_n is None:
        #    self.calculate_eigenvectors()
        # TODO: Rethink about the 'entry' parameter here. Do we need it?
        grid = self._grid_wrap(grid)
        #shape = [1] + list(grid.get_number_nodes())
        shape = (1, grid.get_number_nodes(overall=True))
        return tuple([ numpy.ones(shape, dtype=numpy.complexfloating) ])


    def calculate_exponential(self, factor=1):
        r"""Calculate the matrix exponential :math:`\exp(\alpha V)`. In the
        case of this class the matrix is of size :math:`1 \times 1` thus
        the exponential simplifies to the scalar exponential function.
        Note: This function is idempotent.

        :param factor: The prefactor :math:`\alpha` in the exponential.
        """
        self._exponential_s = sympy.exp(factor*self._potential_s[0,0])
        self._exponential_n = sympy.lambdify(self._all_variables, self._exponential_s, "numpy")


    def evaluate_exponential_at(self, grid, entry=None):
        r"""Evaluate the exponential of the potential matrix :math:`V(x)` on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the exponential at.
        :type grid: A :py:class:`Grid` instance. (Numpy arrays are not directly supported yet.)
        :return: The numerical approximation of the matrix exponential at the given grid nodes.
        """
        grid = self._grid_wrap(grid)

        if self._exponential_n is None:
            self.calculate_exponential()

        # Evaluate the exponential at the given nodes
        values = self._exponential_n(*grid.get_nodes(split=True))

        # Test for potential beeing constant
        if numpy.atleast_1d(values).shape == (1,):
            values = values * numpy.ones(grid.get_number_nodes(), dtype=numpy.complexfloating)

        # Put the result in correct shape (1, #gridnodes)
        N = grid.get_number_nodes(overall=True)
        result = [ values.reshape((1,N)) ]

        # TODO: Consider unpacking single ndarray iff entry != None
        if entry is not None:
            result = result[0]

        return result


    def calculate_jacobian(self):
        r"""Calculate the Jacobian matrix :math:`\nabla V` of the potential :math:`V(x)`
        with :math:`x \in \mathbb{R}^D`. For potentials which depend only one variable,
        this equals the first derivative and :math:`D=1`. Note that this function is idempotent.
        """
        if self._jacobian_s is None:
            # TODO: Add symbolic simplification
            self._jacobian_s = self._potential_s.jacobian(self._all_variables).T
            self._jacobian_n = tuple([ sympy.lambdify(self._all_variables, entry, "numpy") for entry in self._jacobian_s ])


    def evaluate_jacobian_at(self, grid, component=None):
        r"""Evaluate the potential's Jacobian :math:`\nabla V(x)` at some grid
        nodes :math:`\Gamma`.

        :param grid: The grid nodes :math:`\Gamma` the Jacobian gets evaluated at.
        :param component: Dummy parameter that has no effect here.
        :return: The value of the potential's Jacobian at the given nodes. The result
                 is an ``ndarray`` of shape :math:`(D,1)` is we evaluate at a single
                 grid node or of shape :math:`(D,|\Gamma|)`
                 if we evaluate at multiple nodes simultaneously.
        """
        # TODO: Rethink about the 'component' parameter here. Do we need it?
        grid = self._grid_wrap(grid)
        nodes = grid.get_nodes(split=True)

        D = self._dimension
        N = grid.get_number_nodes(overall=True)

        J = numpy.zeros((D,N))

        for row in xrange(D):
            J[row, :] = self._jacobian_n[row](*nodes)

        return J


    def calculate_hessian(self):
        r"""Calculate the Hessian matrix :math:`\nabla^2 V` of the potential :math:`V(x)`
        with :math:`x \in \mathbb{R}^D`. For potentials which depend only one variable,
        this equals the second derivative and :math:`D=1`. Note that this function is idempotent.
        """
        if self._hessian_s is None:
            # TODO: Add symbolic simplification
            self._hessian_s = sympy.hessian(self._potential_s[0,0], self._all_variables)
            self._hessian_n = tuple([ sympy.lambdify(self._all_variables, entry, "numpy") for entry in self._hessian_s ])


    def evaluate_hessian_at(self, grid, component=None):
        r"""Evaluate the potential's Hessian :math:`\nabla^2 V(x)` at some grid
        nodes :math:`\Gamma`.

        :param grid: The grid nodes :math:`\Gamma` the Hessian gets evaluated at.
        :param component: Dummy parameter that has no effect here.
        :return: The value of the potential's Hessian at the given nodes. The result
                 is an ``ndarray`` of shape :math:`(D,D)` is we evaluate at a single
                 grid node or of shape :math:`(|\Gamma|,D,D)` if we evaluate at multiple
                 nodes simultaneously.
        """
        # TODO: Rethink about the 'component' parameter here. Do we need it?
        grid = self._grid_wrap(grid)
        nodes = grid.get_nodes(split=True)

        D = self._dimension
        N = grid.get_number_nodes(overall=True)

        H = numpy.zeros((N,D,D))

        for row in xrange(D):
            for col in xrange(D):
                H[:, row, col] = self._hessian_n[row*D+col](*nodes)

        # 'squeeze' would be dangerous here, make sure it works in the 1D case too
        if N == 1:
            H = H[0,:,:]

        return H


    def calculate_local_quadratic(self, diagonal_component=None):
        r"""Calculate the local quadratic approximation :math:`U(x)` of the potential's
        eigenvalue :math:`\lambda(x)`. Note that this function is idempotent.

        :param diagonal_component: Dummy parameter that has no effect here.
        """
        # Calculation already done at some earlier time?
        self.calculate_eigenvalues()
        self.calculate_jacobian()
        self.calculate_hessian()

        # Construct function to evaluate the taylor approximation at point q at the given nodes
        self._taylor_eigen_s = [ (0, self._eigenvalues_s), (1, self._jacobian_s), (2, self._hessian_s) ]
        self._taylor_eigen_n = [ (0, self._eigenvalues_n), (1, self._jacobian_n), (2, self._hessian_n) ]


    def evaluate_local_quadratic_at(self, grid, diagonal_component=None):
        r"""Numerically evaluate the local quadratic approximation :math:`U(x)` of
        the potential's eigenvalue :math:`\lambda(x)` at the given grid nodes :math:`\Gamma`.
        This function is used for the homogeneous case.

        :param grid: The grid nodes :math:`\Gamma` the quadratic approximation gets evaluated at.
        :param diagonal_component: Dummy parameter that has no effect here.
        :return: A list containing the values :math:`V(\Gamma)`, :math:`\nabla V(\Gamma)` and
                 :math:`\nabla^2 V(\Gamma)`.
        """
        grid = self._grid_wrap(grid)

        # TODO: Relate this to the _taylor_eigen_{s,n} data
        V = self.evaluate_eigenvalues_at(grid, entry=(diagonal_component,diagonal_component))
        J = self.evaluate_jacobian_at(grid)
        H = self.evaluate_hessian_at(grid)

        return tuple([V, J, H])


    def calculate_local_remainder(self, diagonal_component=None):
        r"""Calculate the non-quadratic remainder :math:`W(x) = V(x) - U(x)` of the quadratic
        Taylor approximation :math:`U(x)` of the potential's eigenvalue :math:`\lambda(x)`.
        Note that this function is idempotent.

        :param diagonal_component: Dummy parameter that has no effect here.
        """
        # Calculation already done at some earlier time?
        self.calculate_eigenvalues()
        self.calculate_jacobian()
        self.calculate_hessian()

        # Point q where the taylor series is computed
        # This is a column vector q = (q1, ... ,qD)
        qs = [ sympy.Symbol("q"+str(i)) for i,v in enumerate(self._all_variables) ]

        pairs = [ (xi,qi) for xi,qi in zip(self._all_variables, qs) ]

        V = self._eigenvalues_s.subs(pairs)
        J = self._jacobian_s.subs(pairs)
        H = self._hessian_s.subs(pairs)

        # Symbolic expression for the quadratic Taylor expansion term
        xmq = sympy.Matrix([ (xi-qi) for xi,qi in zip(self._all_variables, qs) ])
        quadratic = V + J.T*xmq + sympy.Rational(1,2)*xmq.T*H*xmq
        try:
            quadratic = quadratic.applyfunc(sympy.simplify)
        except:
            pass

        # Symbolic expression for the Taylor expansion remainder term
        remainder = self._potential_s - quadratic
        try:
            remainder = remainder.applyfunc(sympy.simplify)
        except:
            pass
        self._remainder_s = remainder

        # Construct functions to evaluate the approximation at point q at the given nodes
        # The variable ordering in lambdify is [x1, ..., xD, q1, ...., qD]
        self._remainder_n = tuple([ sympy.lambdify(self._all_variables + qs, self._remainder_s[0,0], "numpy") ])


    def evaluate_local_remainder_at(self, grid, position, diagonal_component=None, entry=None):
        r"""Numerically evaluate the non-quadratic remainder :math:`W(x)` of the quadratic
        approximation :math:`U(x)` of the potential's eigenvalue :math:`\lambda(x)` at the
        given nodes :math:`\Gamma`.

        :param grid: The grid nodes :math:`\Gamma` the remainder :math:`W` gets evaluated at.
        :param position: The point :math:`q \in \mathbb{R}^D` where the Taylor series is computed.
        :param diagonal_component: Dummy parameter that has no effect here.
        :keyword entry: Dummy parameter that has no effect here.
        :return: A list with a single entry consisting of an ``ndarray`` containing the
                 values of :math:`W(\Gamma)`. The array is of shape :math:`(1,|\Gamma|)`.
        """
        grid = self._grid_wrap(grid)

        # Evaluate the remainder at the given nodes
        args = grid.get_nodes(split=True) + numpy.vsplit(position, position.shape[0])
        values = self._remainder_n[0](*args)

        # Test for potential beeing constant
        if numpy.atleast_1d(values).shape == (1,):
            values = values * numpy.ones(grid.get_number_nodes(), dtype=numpy.complexfloating)

        # Put the result in correct shape (1, #gridnodes)
        N = grid.get_number_nodes(overall=True)
        result = [ values.reshape((1,N)) ]

        # TODO: Consider unpacking single ndarray iff entry != None
        if entry is not None:
            result = result[0]

        return result
