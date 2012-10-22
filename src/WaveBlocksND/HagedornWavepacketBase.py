"""The WaveBlocks Project

This file contains the basic interface for general wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import vstack, vsplit, cumsum, zeros, array, complexfloating, pi, dot, sum, atleast_2d
from scipy import exp, sqrt, conjugate
from scipy.linalg import det, inv, norm

from Wavepacket import Wavepacket
from Grid import Grid
from GridWrapper import GridWrapper

__all__ = ["HagedornWavepacketBase"]


class HagedornWavepacketBase(Wavepacket):
    r"""This class implements the abstract :py:class:`Wavepacket` interface
    and contains code common to all types of Hagedorn wavepackets.
    """

    def __init__(self, parameters):
        r"""Initialize a wavepacket object that represents :math:`\Psi`.
        """
        raise NotImplementedError("'HagedornWavepacketBase' should not be instantiated.")


    # We can handle basis shapes here as the logic is the same for
    # homogeneous and inhomogeneous Hagedorn wavepackets.

    def _resize_coefficient_storage(self, component, bs_old, bs_new):
        r"""
        """
        bso = bs_old.get_basis_size()
        bsn = bs_new.get_basis_size()

        # Find the intersection of K and K'
        # Optimization: iterate over smaller set
        if bso <= bsn:
            insec = [ k for k in bs_old if k in bs_new ]
        elif bso > bsn:
            insec = [ k for k in bs_new if k in bs_old ]
        # TODO: Consider making this part of the BasisShape interface
        # TODO: Consider implementing set operations for basis shapes

        # Construct the index mapping
        i = array([ bs_old[k] for k in insec ])
        j = array([ bs_new[k] for k in insec ])

        # Copy over the data
        cnew = zeros((bsn,1), dtype=complexfloating)
        cnew[j] = self._coefficients[component][i]
        self._coefficients[component] = cnew


    def get_basis_shape(self, component=None):
        r"""Retrieve the basis shapes :math:`\mathcal{K}_i` for each component :math:`i`.

        :param component: The component :math:`i` whose basis shape we request. (Default is
                          ``None`` which means to return the basis shapes for all components.
        :type component: int
        :return: The basis shape for an individual component or a list with all shapes.
        """
        if component is not None:
            return self._basis_shapes[component]
        else:
            return tuple(self._basis_shapes)


    def set_basis_shape(self, basis_shape, component=None):
        r"""Set the basis shape :math:`\mathcal{K}` of a given component or for all components.

        :param basis_shape: The basis shape for an individual component or a list with all :math:`N` shapes.
        :type basis_shape: A subclass of :py:class:`BasisShape`.
        :param component: The component :math:`i` whose basis shape we want to set. (Default is
                          ``None`` which means to set the basis shapes for all components.
        :type component: int
        """
        if component is not None:
            # Check for valid input basis shape
            if not component in range(self._number_components):
                raise ValueError("Invalid component index " + str(component))

            # Adapt the coefficient storage vectors
            self._resize_coefficient_storage(component, self._basis_shapes[component], basis_shape)
            # Set the new basis shape for the given component
            self._basis_shapes[component] = basis_shape
        else:
            # Check for valid input basis shape
            if not len(basis_shape) == self._number_components:
                raise ValueError("Number of basis shape(s) does not match to number of components.")

            for index, bsnew in enumerate(basis_shape):
                # Adapt the coefficient storage vectors
                self._resize_coefficient_storage(index, self._basis_shapes[index], bsnew)
                # Set the new basis shape for the given component
                self._basis_shapes[index] = bsnew

        # And update the caches information
        self._basis_sizes = [ bs.get_basis_size() for bs in self._basis_shapes ]


    # We can handle coefficient set manipulation here as the logic is
    # the same for homogeneous and inhomogeneous Hagedorn wavepackets.


    def set_coefficient(self, component, index, value):
        r"""Set a single coefficient :math:`c^i_k` of the specified component :math:`\Phi_i`
        of :math:\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` we want to update.
        :type components: int
        :param index: The multi-index :math:`k` of the coefficient :math:`c^i_k` we want to update.
        :type index: A tuple of :math:`D` integers.
        :param value: The new value of the coefficient :math:`c^i_k`.
        :raise ValueError: For invalid indices :math:`i` or :math:`k`.
        """
        if component > self._number_components-1 or component < 0:
            raise ValueError("There is no component with index "+str(component)+".")
        if not index in self._basis_shapes[component]:
            raise ValueError("There is no basis function with multi-index "+str(index)+".")

        # Apply linear order mapping here
        key = self._basis_shapes[component][index]
        self._coefficients[component][key] = value


    def get_coefficient(self, component, index):
        r"""Retrieve a single coefficient :math:`c^i_k` of the specified component :math:`\Phi_i`
        of :math:\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` we want to update.
        :type components: int
        :param index: The multi-index :math:`k` of the coefficient :math:`c^i_k` we want to update.
        :type index: A tuple of :math:`D` integers.
        :return: A single complex number.
        """
        if component > self._number_components-1 or component < 0:
            raise ValueError("There is no component with index "+str(component)+".")
        if not index in self._basis_shapes[component]:
            raise ValueError("There is no basis function with multi-index "+str(index)+".")

        # Apply linear order mapping here
        key = self._basis_shapes[component][index]
        return self._coefficients[component][key]


    def set_coefficients(self, values, component=None):
        r"""Update all the coefficients :math:`c` of :math:`\Psi` or update
        the coefficients :math:`c^i` of the components :math:`\Phi_i` only.

        Note: this method copies the data arrays.

        :param values: The new values of the coefficients :math:`c^i` of :math:`\Phi_i`.
        :type values: An ndarray of suitable shape or a list of ndarrays.
        :param component: The index :math:`i` of the component we want to update with new coefficients.
        :type component: int (Default is ``None`` meaning all)
        :raise ValueError: For invalid component indices :math:`i`.
        """
        if component is None:
            if len(values) != self._number_components:
                raise ValueError("Too less or too many data provided.")

            for index, value in enumerate(values):
                bs = self._basis_sizes[index]
                self._coefficients[index] = value.copy().reshape((bs,1))
        else:
            if component > self._number_components-1 or component < 0:
                raise ValueError("There is no component with index "+str(component)+".")

            bs = self._basis_sizes[component]
            self._coefficients[component] = value.copy().reshape((bs,1))


    def get_coefficients(self, component=None):
        r"""Returns the coefficients :math:`c^i` for some component :math:`\Phi_i` of
        :math:`\Psi` or all the coefficients :math:`c` of all components.

        Note: this method copies the data arrays.

        :param component: The index :math:`i` of the component we want to retrieve.
        :type component: int (Default is ``None`` meaning all)
        :return: A single ndarray with the coefficients of the given component or
                 a list containing the ndarrays for each component. Each ndarray
                 is two-dimensional with a shape of :math:`(|\mathcal{K}_i|, 1)`.
        """
        if component is None:
            return [ item.copy() for item in self._coefficients ]
        else:
            if component > self._number_components-1 or component < 0:
                raise ValueError("There is no component with index "+str(component)+".")

            return self._coefficients[component].copy()


    def get_coefficient_vector(self):
        r"""Retrieve the coefficients for all components :math:`\Phi_i` simultaneously.

        :return: The coefficients :math:`c^i` of all components
                 :math:`\Phi_i` stacked into a single long column vector.
        """
        return vstack(self._coefficients)


    def set_coefficient_vector(self, vector):
        """Set the coefficients for all components :math:`\Phi_i` simultaneously.

        Note: This function does *NOT* copy the input data! This is for efficiency
        as this routine is used in the innermost loops.

        :param vector: The coefficients of all components as a single long column vector.
        :type vector: A two-dimensional ndarray of appropriate shape.
        """
        # Compute the partition of the block-vector from the basis sizes
        partition = cumsum(self._basis_sizes)[:-1]

        # Split the block-vector with the given partition and assign
        self._coefficients = vsplit(vector, partition)


    def get_eps(self):
        r"""Retrieve the semi-classical scaling parameter :math:`\varepsilon` of the wavepacket.

        :return: The value of :math:`\varepsilon`.
        """
        return self._eps


    # We can evaluate the ground state basis function phi_0 on a set of nodes
    # the same way for homogeneous and inhomogeneous Hagedorn wavepackets.


    def _grid_wrap(self, grid):
        # TODO: Consider additional input types for "nodes":
        #       list of numpy ndarrays, list of single python scalars
        if not isinstance(grid, Grid):
            grid = atleast_2d(grid)
            grid = grid.reshape(self._dimension, -1)
            grid = GridWrapper(grid)
        return grid


    def _evaluate_phi0(self, Pi, nodes, prefactor=False, root=sqrt):
        r"""Evaluate the lowest order basis function :math:`\phi_0` on a
        grid :math:`\Gamma` of nodes.

        :param Pi: The parameter set :math:`\Pi`.
        :param nodes: The nodes we evaluate :math:`\phi_0` at.
        :type nodes: An ndarray of shape ``(D, |\Gamma|)``.
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: bool, default is ``False``.
        :param root: The function used to compute the square root in the prefactor.
                     Defaults to the ``sqrt`` function of ``numpy`` but can be any
                     callable object and especially an instance of :py:class:`ContinuousSqrt`.
        :return: An ndarray of shape ``(|\Gamma|)``.
        """
        d = self._dimension
        eps = self._eps
        q, p, Q, P, S = Pi

        # TODO: Use LU instead of inv(...)
        df = nodes - q
        pr1 = sum(df * dot(dot(P,inv(Q)), df), axis=0)
        pr2 = sum(p * df, axis=0)

        exponent = 1.0j / eps**2 * (0.5 * pr1 + pr2)

        # The problematic prefactor cancels in inner products
        if prefactor is True:
            prefactor = (pi*eps**2)**(-d*0.25) / root(det(Q))
        else:
            prefactor = (pi*eps**2)**(-d*0.25)

        return prefactor * exp(exponent)


    def evaluate_basis_at(self, grid, component, prefactor=False):
        r"""Evaluate the basis functions :math:`\phi_k` recursively at the given nodes :math:`\gamma`.

        :param grid: The grid :math:\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:meth:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: bool, default is ``False``.
        :return: A two-dimensional ndarray :math:`H` of shape :math:`(|\mathcal{K}_i|, |\Gamma|)` where
                 the entry :math:`H[\mu(k), i]` is the value of :math:`\phi_k(\gamma_i)`.
        """
        D = self._dimension

        bas = self._basis_shapes[component]
        bs = self._basis_sizes[component]

        # The grid
        grid = self._grid_wrap(grid)
        nodes = grid.get_nodes()
        nn = grid.get_number_nodes(overall=True)

        # Allocate the storage array
        phi = zeros((bs, nn), dtype=complexfloating)

        # Precompute some constants
        Pi = self.get_parameters(component=component)
        q, p, Q, P, S = Pi

        Qinv = inv(Q)
        Qbar = conjugate(Q)
        QQ = dot(Qinv, Qbar)

        # Compute the ground state phi_0 via direct evaluation
        mu0 = bas[tuple(D*[0])]
        phi[mu0,:] = self._evaluate_phi0(Pi, nodes, prefactor=False)

        # Compute all higher order states phi_k via recursion
        for d in xrange(D):
            # Iterator for all valid index vectors k
            indices = bas.get_node_iterator(mode="chain", direction=d)

            for k in indices:
                # Current index vector
                ki = vstack(k)

                # Access predecessors
                phim = zeros((D, nn), dtype=complexfloating)

                for j, kpj in bas.get_neighbours(k, selection="backward"):
                    mukpj = bas[kpj]
                    phim[j,:] = phi[mukpj,:]

                # Compute 3-term recursion
                p1 = (nodes - q) * phi[bas[k],:]
                p2 = sqrt(ki) * phim

                t1 = sqrt(2.0/self._eps**2) * dot(Qinv[d,:], p1)
                t2 = dot(QQ[d,:], p2)

                # Find multi-index where to store the result
                kped = bas.get_neighbours(k, selection="forward", direction=d)

                # Did we find this k?
                if len(kped) > 0:
                    kped = kped[0]

                    # Store computed value
                    phi[bas[kped[1]],:] = (t1 - t2) / sqrt(ki[d] + 1.0)

        if prefactor is True:
            phi = phi / self._get_sqrt(component)(det(Q))

        return phi


    def slim_recursion(self, grid, component, prefactor=False):
        r"""Evaluate the Hagedorn wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.
        This routine is a slim version compared to the full basis evaluation. At every moment
        we store only the data we really need to compute the next step until we hit the highest
        order basis functions.

        :param grid: The grid :math:\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:meth:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: bool, default is ``False``.
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i`
                 at the nodes :math:`\gamma`.

        Note that this function does not include the global phase :math:`\exp(\frac{i S}{\varepsilon^2})`.
        """
        D = self._dimension

        # Precompute some constants
        Pi = self.get_parameters(component=component)
        q, p, Q, P, S = Pi

        Qinv = inv(Q)
        Qbar = conjugate(Q)
        QQ = dot(Qinv, Qbar)

        # The basis shape
        bas = self._basis_shapes[component]
        Z = tuple(D*[0])

        # Book keeping
        todo = []
        newtodo = [Z]
        olddelete = []
        delete = []
        tmp = {}

        # The grid nodes
        grid = self._grid_wrap(grid)
        nn = grid.get_number_nodes(overall=True)
        nodes = grid.get_nodes()

        # Evaluate phi0
        tmp[Z] = self._evaluate_phi0(Pi, nodes, prefactor=False)
        psi = self._coefficients[component][bas[Z], 0] * tmp[Z]

        # Iterate for higher order states
        while len(newtodo) != 0:
            # Delete results that never will be used again
            for d in olddelete:
                del tmp[d]

            # Exchange queus
            todo = newtodo
            newtodo = []
            olddelete = delete
            delete = []

            # Compute new results
            for k in todo:
                # Center stencil at node k
                ki = vstack(k)

                # Access predecessors
                phim = zeros((D, nn), dtype=complexfloating)
                for j, kpj in bas.get_neighbours(k, selection="backward"):
                    phim[j,:] = tmp[kpj]

                # Compute the neighbours
                for d, n in bas.get_neighbours(k, selection="forward"):
                    if not n in tmp.keys():
                        # Compute 3-term recursion
                        p1 = (nodes - q) * tmp[k]
                        p2 = sqrt(ki) * phim

                        t1 = sqrt(2.0/self._eps**2) * dot(Qinv[d,:], p1)
                        t2 = dot(QQ[d,:], p2)

                        # Store computed value
                        tmp[n] = (t1 - t2) / sqrt(ki[d] + 1.0)
                        # And update the result
                        psi = psi + self._coefficients[component][bas[n], 0] * tmp[n]

                        newtodo.append(n)
                delete.append(k)

        if prefactor is True:
            psi = psi / self._sqrt(det(Q))

        return psi


    def evaluate_at(self, grid, component=None, prefactor=False):
        r"""Evaluate the Hagedorn wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.

        :param grid: The grid :math:\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:meth:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
                          (Defaults to ``None`` for evaluating all components.)
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: bool, default is ``False``.
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i` at the nodes :math:`\gamma`.
        """
        Pis = self.get_parameters(component=component, aslist=True)

        if component is not None:
            phase = exp(1.0j * Pis[component][4] / self._eps**2)
            #basis = self.evaluate_basis_at(grid, component=component, prefactor=prefactor)
            #values = phase * sum(self._coefficients[component] * basis, axis=0)
            values = phase * self.slim_recursion(grid, component, prefactor=prefactor)

        else:
            values = []

            for component in xrange(self._number_components):
                # Note: This is very inefficient! We may evaluate the same basis functions multiple
                #       times. But as long as we don't know that the basis shapes are true subsets
                #       of the largest one, we can not evaluate just all functions in this
                #       maximal set.

                # TODO: Find more efficient way to do this

                phase = exp(1.0j * Pis[component][4] / self._eps**2)
                #basis = self.evaluate_basis_at(grid, component=index, prefactor=prefactor)
                #values.append( phase * sum(self._coefficients[index] * basis, axis=0) )
                values.append( phase * self.slim_recursion(grid, component, prefactor=prefactor) )

        return values


    # We can compute the norms the same way for homogeneous and inhomogeneous Hagedorn wavepackets.


    def norm(self, component=None, summed=False):
        r"""Calculate the :math:`L^2` norm :math:`\langle\Psi|\Psi\rangle` of the wavepacket :math:`\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` whose norm is calculated.
                          The default value is ``None`` which means to compute the norms of all :math:`N` components.
        :type component: int or ``None``.
        :param summed: Whether to sum up the norms :math:`\langle\Phi_i|\Phi_i\rangle` of the
                       individual components :math:`\Phi_i`.
        :type summed: Boolean, default is ``False``.
        :type summed: Boolean, default is ``False``.
        :return: The norm of :math:`\Psi` or the norm of :math:`\Phi_i` or a list with the :math:`N`
                 norms of all components. Depending on the values of ``component`` and ``summed``.
        """
        if component is not None:
            result = norm(self._coefficients[component])
        else:
            result = [ norm(item) for item in self._coefficients ]

            if summed is True:
                result = reduce(lambda x,y: x+conjugate(y)*y, result, 0.0)
                result = sqrt(result)

        return result


    # A wavepacket knows how to compute inner products
    # TODO: Consider moving this inside the general codata framework
    # TODO: Rethink if wavepackets should contain a QR


    def set_quadrature(self, quadrature):
        """Set the :py:class:`Quadrature` subclass instance used for computing
        inner products and evaluating brakets.

        :param quadrature: The new :py:class:`Quadrature` subclass instance.
        """
        self._QE = quadrature


    def get_quadrature(self):
        """Return the :py:class:`Quadrature` subclass instance used computing
        inner rpoducts and evaluating brakets.

        :return: The current :py:class:`Quadrature` subclass instance.
        """
        return self._QE
