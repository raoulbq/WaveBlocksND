r"""The WaveBlocks Project

This file contains the abstract base class for the representation of matrix-valued
potentials :math:`V(x)` in an arbitrary number :math:`D` of space dimensions, i.e.
we have :math:`x \in \mathbb{R}^D`. The potential can model an arbitrary but fixed
number :math:`N` of energy levels :math:`\lambda_i`.

The :py:class:`MatrixPotential` defines the interface every subclass must support
to represent a potential. Additionally it implements some of the common methods.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["MatrixPotential"]


class MatrixPotential(object):
    r"""This class represents a potential :math:`V(x)` with :math:`x \in \mathbb{R}^D`.
    The potential is given as an analytic expression. Some calculations with the
    potential are supported. For example calculation of eigenvalues :math:`\lambda_i(x)`
    and eigenvectors :math:`\nu_i(x)` and numerical evaluation on a grid :math:`\Gamma`.
    """

    def __init__(self):
        r"""Create a new :py:class:`MatrixPotential` instance for a given potential
        matrix :math:`V(x)`.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'MatrixPotential' is an abstract base class.")


    def get_dimension(self):
        r"""Return the dimension :math:`D` of the potential :math:`V(x)`.
        The dimension is equal to the number of free variables :math:`x_i`
        where :math:`x := (x_1, x_2, ..., x_D)`.
        """
        return self._dimension


    def get_number_components(self):
        r"""Return the number :math:`N` of components the potential :math:`V(x)`
        supports. This is equivalent to the number of energy levels :math:`\lambda_i(x)`.
        """
        return self._number_components


    def evaluate_at(self, grid, entry=None):
        r"""Evaluate the potential :math:`V(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the potential at.
        :param entry: The indices :math:`(i,j)` of the component :math:`V_{i,j}(x)`
                      we want to evaluate or `None` to evaluate all entries.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_at(...)")


    def calculate_eigenvalues(self):
        r"""Calculate the eigenvalue :math:`\lambda_0(x)` of the potential :math:`V(x)`.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_eigenvalues(...)")


    def evaluate_eigenvalues_at(self, grid, entry=None):
        r"""Evaluate the eigenvalues :math:`\Lambda(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvalues at.
        :param entry: The index :math:`i` of the eigenvalue :math:`\lambda_i(x)`
                      we want to evaluate or `None` to evaluate all eigenvalues.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_eigenvalues_at(...)")


    def calculate_eigenvectors(self):
        r"""Calculate the eigenvectors :math:`\nu_i(x)` of the potential :math:`V(x)`.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_eigenvectors(...)")


    def evaluate_eigenvectors_at(self, grid, entry=None):
        r"""Evaluate the eigenvectors :math:`\nu_i(x)` elementwise on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the eigenvectors at.
        :param entry: The index :math:`i` of the eigenvector :math:`\nu_i(x)`
                      we want to evaluate or `None` to evaluate all eigenvectors.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_eigenvectors_at(...)")


    def calculate_exponential(self, factor=1):
        r"""Calculate the matrix exponential :math:`\exp(\alpha V)`.

        :param factor: The prefactor :math:`\alpha` in the exponential.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_exponential(...)")


    def evaluate_exponential_at(self, grid):
        r"""Evaluate the exponential of the potential matrix :math:`V(x)` on a grid :math:`\Gamma`.

        :param grid: The grid containing the nodes :math:`\gamma_i` we want
                     to evaluate the exponential at.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_exponential_at(...)")
