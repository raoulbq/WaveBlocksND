"""The WaveBlocks Project



@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, eye, integer, complexfloating, atleast_2d, concatenate, hstack, vstack, squeeze
from numpy import pi, dot, einsum, conjugate
from scipy import exp, sqrt
from scipy.linalg import det, inv


from LinearCombinationOfWavepackets import LinearCombinationOfWavepackets
from HagedornWavepacket import HagedornWavepacket
from Grid import Grid
from GridWrapper import GridWrapper


__all__ = ["LinearCombinationOfHAWPs"]


class LinearCombinationOfHAWPs(LinearCombinationOfWavepackets):
    r"""This class represents linear combinations
    of compatible Hagedorn wavepackets.
    """

    def __init__(self, dimension, number_components, number_packets=0):
        r"""Initialize a new linear combination of Hagedorn wavepackets. This
        object represents :math:`\Upsilon := \sum_{j=0}^{J-1} c_j \Psi_j`.
        All :math:`J` wavepackets :math:`\Psi_j` have the same number :math:`N`
        components and are defined in the :math:`D` dimensional space.

        :param dimension: The space dimension :math:`D` the packets have.
        :param ncomponents: The number :math:`N` of components the packets have.
        :return: An instance of :py:class:`LinearCombinationOfHAWPs`.
        """
        self._dimension = dimension
        self._number_components = number_components
        self._number_packets = number_packets

        # Basis shapes
        self._basis_shapes_hashes = []
        self._basis_shapes = {}

        # Coefficients of individual packets
        self._wp_coefficients = zeros((number_packets,0), dtype=complexfloating)
        self._basis_sizes = []

        # Default parameters of harmonic oscillator eigenstates
        q = zeros((self._number_packets, self._dimension), dtype=complexfloating)
        p = zeros((self._number_packets, self._dimension), dtype=complexfloating)
        Q = ones((self._number_packets,1,1)) * eye(self._dimension, dtype=complexfloating)
        P = 1.0j * ones((self._number_packets,1,1)) * eye(self._dimension, dtype=complexfloating)
        S = zeros((self._number_packets,1), dtype=complexfloating)

        # Parameters
        self._Pis = [q, p, Q, P, S]

        # Coefficients of linear combination
        self._lc_coefficients = zeros((number_packets,1), dtype=integer)

        # TODO: Handle multi-component packets
        assert number_components == 1


    def _resize_coefficient_storage(self, newsize):
        r"""
        """
        curlen, cursize = self._wp_coefficients.shape
        diff = newsize - cursize
        if diff > 0:
            z = zeros((curlen,diff), dtype=complexfloating)
            self._wp_coefficients = hstack([self._wp_coefficients, z])
        elif diff < 0:
            self._wp_coefficients[:,:newsize]


    def add_wavepacket(self, packet, coefficient):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        if not packet.get_dimension() == self._dimension:
            raise ValueError("Number of dimensions does not match.")
        if not packet.get_number_components() == self._number_components:
            raise ValueError("Number of components does not match.")
        # Note: we do not test that the varepsilon parameter matches.

        self._number_packets = self._number_packets + 1
        # Epsilon
        self._eps = packet.get_eps()
        # Store the shape
        K = packet.get_basis_shapes(component=0)
        self._basis_shapes_hashes.append(hash(K))
        self._basis_shapes[hash(K)] = K
        self._basis_sizes.append(K.get_basis_size())
        # Store the coefficients
        bs = K.get_basis_size()
        c = packet.get_coefficients(component=0)
        self._resize_coefficient_storage(bs)
        self._wp_coefficients = vstack([self._wp_coefficients, c.reshape((1,-1))])
        # Store the parameter set
        D = self._dimension
        qs, ps, Qs, Ps, Ss = self._Pis
        q, p, Q, P, S = packet.get_parameters(component=0)
        self._Pis[0] = concatenate([qs, q.reshape((1,D))], axis=0)
        self._Pis[1] = concatenate([ps, p.reshape((1,D))], axis=0)
        self._Pis[2] = concatenate([Qs, Q.reshape((1,D,D))], axis=0)
        self._Pis[3] = concatenate([Ps, P.reshape((1,D,D))], axis=0)
        self._Pis[4] = concatenate([Ss, S], axis=0)
        # Store the linear combination coefficient
        self._lc_coefficients = vstack([self._lc_coefficients, atleast_2d(coefficient)])


    def add_wavepackets(self, packetlist, coefficients):
        r"""
        """
        for j, packet in enumerate(packetlist):
            self.add_wavepacket(packet, coefficients[j])


    # def remove_wavepacket(self, index):
    #     r"""
    #     """
    #     raise NotImplementedError()


    def get_wavepacket(self, packetindex):
        r"""
        """
        if packetindex > self._number_packets-1 or packetindex < 0:
            raise ValueError("There is no packet with index "+str(packetindex)+".")

        HAWP = HagedornWavepacket(self._dimension, self._number_components, self._eps)
        K = self._basis_shapes[self._basis_shapes_hashes[packetindex]]
        HAWP.set_basis_shapes([K])
        q = self._Pis[0][packetindex,:]
        p = self._Pis[1][packetindex,:]
        Q = self._Pis[2][packetindex,:,:]
        P = self._Pis[3][packetindex,:,:]
        S = self._Pis[4][packetindex,:,:]
        HAWP.set_parameters([q,p,Q,P,S])
        cj = self._wp_coefficients[packetindex,:]
        HAWP.set_coefficients(cj, component=0)
        return HAWP


    def get_wavepackets(self):
        r"""

        Be aware of inconsistencies arising if you update the linear
        combination while using this generator!
        """
        return (self.get_wavepacket(j) for j in xrange(self._number_wavepackets))


    # def set_wavepackets(self, packetlist):
    #     r"""
    #     :raise: :py:class:`NotImplementedError` Abstract interface.
    #     """
    #     raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


    def get_basis_shapes(self, packetindex=None):
        r"""Retrieve the basis shapes :math:`\mathfrak{K}_i` for each component :math:`i`.

        :param component: The component :math:`i` whose basis shape we request. (Default is
                          ``None`` which means to return the basis shapes for all components.
        :type component: int
        :return: The basis shape for an individual component or a list with all shapes.
        """
        if packetindex is not None:
            return self._basis_shapes[self._basis_shapes_hashes[packetindex]]
        else:
            return tuple([self._basis_shapes[self._basis_shapes_hashes[j]] for j in xrange(self._number_packets)])


    # def set_basis_shapes(self, basis_shape, component=None):
    #     r"""Set the basis shape :math:`\mathfrak{K}` of a given component or for all components.

    #     :param basis_shape: The basis shape for an individual component or a list with all :math:`N` shapes.
    #     :type basis_shape: A subclass of :py:class:`BasisShape`.
    #     :param component: The component :math:`i` whose basis shape we want to set. (Default is
    #                       ``None`` which means to set the basis shapes for all components.
    #     :type component: int
    #     """
    #     if component is not None:
    #         # Check for valid input basis shape
    #         if not component in range(self._number_components):
    #             raise ValueError("Invalid component index " + str(component))

    #         # Adapt the coefficient storage vectors
    #         self._resize_coefficient_storage(component, self._basis_shapes[component], basis_shape)
    #         # Set the new basis shape for the given component
    #         self._basis_shapes[component] = basis_shape
    #     else:
    #         # Check for valid input basis shape
    #         if not len(basis_shape) == self._number_components:
    #             raise ValueError("Number of basis shape(s) does not match to number of components.")

    #         for index, bsnew in enumerate(basis_shape):
    #             # Adapt the coefficient storage vectors
    #             self._resize_coefficient_storage(index, self._basis_shapes[index], bsnew)
    #             # Set the new basis shape for the given component
    #             self._basis_shapes[index] = bsnew

    #     # And update the caches information
    #     self._basis_sizes = [ bs.get_basis_size() for bs in self._basis_shapes ]


    def set_wavepacket_coefficient(self, packetindex, index, value):
        r"""Set a single coefficient :math:`c^i_k` of the specified component :math:`\Phi_i`
        of :math:`\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` we want to update.
        :type components: int
        :param index: The multi-index :math:`k` of the coefficient :math:`c^i_k` we want to update.
        :type index: A tuple of :math:`D` integers.
        :param value: The new value of the coefficient :math:`c^i_k`.
        :raise: :py:class:`ValueError` For invalid indices :math:`i` or :math:`k`.
        """
        if packetindex > self._number_packets-1 or packetindex < 0:
            raise ValueError("There is no packet with index "+str(packetindex)+".")

        basis_shape = self._basis_shapes[self._basis_shapes_hashes[packetindex]]

        if not index in basis_shape:
            raise ValueError("There is no basis function with multi-index "+str(index)+".")

        # Apply linear order mapping here
        key = basis_shape[index]
        self._wp_coefficients[packetindex][key] = value


    def get_wavepacket_coefficient(self, packetindex, index):
        r"""Retrieve a single coefficient :math:`c^i_k` of the specified component :math:`\Phi_i`
        of :math:`\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` we want to update.
        :type components: int
        :param index: The multi-index :math:`k` of the coefficient :math:`c^i_k` we want to update.
        :type index: A tuple of :math:`D` integers.
        :return: A single complex number.
        :raise: :py:class:`ValueError` For invalid indices :math:`i` or :math:`k`.
        """
        if packetindex > self._number_packets-1 or packetindex < 0:
            raise ValueError("There is no packet with index "+str(packetindex)+".")

        basis_shape = self._basis_shapes[self._basis_shapes_hashes[packetindex]]

        if not index in basis_shape:
            raise ValueError("There is no basis function with multi-index "+str(index)+".")

        # Apply linear order mapping here
        key = basis_shape[index]
        return self._wp_coefficients[packetindex][key]


    def set_wavepacket_coefficients(self, packetindex, coefficients):
        r"""Retrieve a single coefficient :math:`c^i_k` of the specified component :math:`\Phi_i`
        of :math:`\Psi`.

        Warning: make sure the coefficients and basis shapes stay in sync!

        :param component: The index :math:`i` of the component :math:`\Phi_i` we want to update.
        :type components: int
        :param index: The multi-index :math:`k` of the coefficient :math:`c^i_k` we want to update.
        :type index: A tuple of :math:`D` integers.
        :return: A single complex number.
        :raise: :py:class:`ValueError` For invalid indices :math:`i` or :math:`k`.
        """
        if packetindex is not None:
            if packetindex > self._number_packets-1 or packetindex < 0:
                raise ValueError("There is no packet with index "+str(packetindex)+".")

            bs = self._basis_sizes[packetindex]
            self._wp_coefficients[packetindex,:bs] = squeeze(coefficients)
        else:
            curshape = self._wp_coefficients.shape
            self._wp_coefficients = coefficients.reshape(curshape)


    def get_wavepacket_coefficients(self, packetindex=None):
        r"""Retrieve a single coefficient :math:`c^i_k` of the specified component :math:`\Phi_i`
        of :math:`\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` we want to update.
        :type components: int
        :param index: The multi-index :math:`k` of the coefficient :math:`c^i_k` we want to update.
        :type index: A tuple of :math:`D` integers.
        :return: A single complex number.
        :raise: :py:class:`ValueError` For invalid indices :math:`i` or :math:`k`.
        """
        if packetindex is not None:
            if packetindex > self._number_packets-1 or packetindex < 0:
                raise ValueError("There is no packet with index "+str(packetindex)+".")

            bs = self._basis_sizes[packetindex]
            return self._wp_coefficients[packetindex,:bs].copy()
        else:
            return self._wp_coefficients.copy()


    def get_parameters(self, packetindex=None, key=("q","p","Q","P","S")):
        r"""Get the Hagedorn parameter set :math:`\Pi` of the wavepacket :math:`\Psi_j`.

        :param packetindex: The index :math:`0 \leq j < J` of the packet whose parameter
                      set :math:`Pi` we want.
        :return: The Hagedorn parameter set :math:`\Pi = (q, p, Q, P, S)` in this order.
        """
        if packetindex is None:
            packetindex = slice(None)

        Pilist = []
        for k in key:
            if k == "q":
                Pilist.append(self._Pis[0][packetindex])
            elif k == "p":
                Pilist.append(self._Pis[1][packetindex])
            elif k == "Q":
                Pilist.append(self._Pis[2][packetindex])
            elif k == "P":
                Pilist.append(self._Pis[3][packetindex])
            elif k == "S":
                Pilist.append(self._Pis[4][packetindex])
            else:
                raise KeyError("Invalid parameter key: "+str(key))

        return Pilist


    def get_eps(self):
        r"""Retrieve the semi-classical scaling parameter :math:`\varepsilon` of the wavepacket.

        :return: The value of :math:`\varepsilon`.
        """
        return self._eps


    def get_coefficient(self, packetindex):
        r"""Get the coefficient :math:`c_j` of the wavepacket :math:`\Psi_j`.

        :param packetindex: The index :math:`0 \leq j < J` of the coefficient to retrieve.
        :return: The coefficient :math:`c_j`.
        """
        return self._lc_coefficients[packetindex]


    def set_coefficient(self, packetindex, coefficient):
        r"""Set the coefficient :math:`c_j` of the wavepacket :math:`\Psi_j`.

        :param packetindex: The index :math:`0 \leq j < J` of the coefficient to retrieve.
        :param coefficient: The coefficient :math:`c_j`.
        """
        self._lc_coefficients[packetindex] = coefficient


    def get_coefficients(self):
        r"""Get the vector with all coefficients :math:`c_j` of all wavepackets :math:`\Psi_j`.

        :return: The vector :math:`c` of all coefficients :math:`c_j`. The vector is of
                 shape :math:`(J, 1)`.
        :type: An :py:class:`ndarray`
        """
        return self._lc_coefficients.copy()


    def set_coefficients(self, coefficients):
        r"""Update all the coefficients :math:`c` of :math:`\Upsilon`.

        :param coefficients: The vector :math:`c`.
        :type coefficients: An :py:class:`ndarray`
        """
        if not coefficients.size == self._number_packets:
            raise ValueError("Wrong number of new coefficients.")

        self._lc_coefficients = coefficients.copy().reshape((-1,1))


    def _grid_wrap(self, grid):
        # TODO: Consider additional input types for "nodes":
        #       list of numpy ndarrays, list of single python scalars
        if not isinstance(grid, Grid):
            grid = atleast_2d(grid)
            grid = grid.reshape(self._dimension, -1)
            grid = GridWrapper(grid)
        return grid


    def _evaluate_phi0(self, nodes, packetindex, prefactor=False):
        r"""Evaluate the lowest order basis function :math:`\phi_0` on a
        grid :math:`\Gamma` of nodes.

        :param Pi: The parameter set :math:`\Pi`.
        :param nodes: The nodes we evaluate :math:`\phi_0` at.
        :type nodes: An ndarray of shape ``(D, |\Gamma|)``.
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: Boolean, default is ``False``.
        :param root: The function used to compute the square root in the prefactor.
                     Defaults to the ``sqrt`` function of ``numpy`` but can be any
                     callable object and especially an instance of :py:class:`ContinuousSqrt`.
        :return: An ndarray of shape ``(|\Gamma|)``.
        """
        D = self._dimension
        eps = self._eps

        # The current parameters
        q = self._Pis[0][packetindex,:].reshape((D,1))
        p = self._Pis[1][packetindex,:].reshape((D,1))
        Q = self._Pis[2][packetindex,:,:].reshape((D,D))
        P = self._Pis[3][packetindex,:,:].reshape((D,D))

        # TODO: Maybe use LU instead of inv(...)
        df = nodes - q
        pr1 = einsum("ik,ij,jk->k", df, dot(P,inv(Q)), df)
        pr2 = einsum("ij,ik", p, df)
        exponent = 1.0j / eps**2 * (0.5 * pr1 + pr2)

        # The problematic prefactor cancels in inner products
        if prefactor is True:
            prefactor = (pi*eps**2)**(-D*0.25) / sqrt(det(Q))
        else:
            prefactor = (pi*eps**2)**(-D*0.25)

        return prefactor * exp(exponent)


    def slim_recursion(self, grid, packetindex, prefactor=False):
        r"""Evaluate the Hagedorn wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.
        This routine is a slim version compared to the full basis evaluation. At every moment
        we store only the data we really need to compute the next step until we hit the highest
        order basis functions.

        :param grid: The grid :math:`\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:meth:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: Boolean, default is ``False``.
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i`
                 at the nodes :math:`\gamma`.

        Note that this function does not include the global phase :math:`\exp(\frac{i S}{\varepsilon^2})`.
        """
        D = self._dimension

        # The current parameters
        q = self._Pis[0][packetindex,:].reshape((D,1))
        #p = self._Pis[1][packetindex,:].reshape((D,1))
        Q = self._Pis[2][packetindex,:,:].reshape((D,D))
        #P = self._Pis[3][packetindex,:,:].reshape((D,D))

        # Precompute some constants
        Qinv = inv(Q)
        Qbar = conjugate(Q)
        QQ = dot(Qinv, Qbar)

        # The basis shape
        bas = self._basis_shapes[self._basis_shapes_hashes[packetindex]]
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
        tmp[Z] = self._evaluate_phi0(nodes, packetindex, prefactor=False)

        psi = self._wp_coefficients[packetindex,bas[Z]] * tmp[Z]

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
                        psi = psi + self._wp_coefficients[packetindex,bas[n]] * tmp[n]

                        newtodo.append(n)
                delete.append(k)

        if prefactor is True:
            psi = psi / sqrt(det(Q))

        return psi


    def evaluate_at(self, grid, packetindex=None, prefactor=False):
        r"""Evaluate the Hagedorn wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.

        :param grid: The grid :math:`\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:meth:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
                          (Defaults to ``None`` for evaluating all components.)
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: Boolean, default is ``False``.
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i` at the nodes :math:`\gamma`.
        """
        eps = self._eps

        if packetindex is not None:
            S = self._Pis[4][packetindex,:]
            phase = exp(1.0j * S / eps**2)
            values = phase * self.slim_recursion(grid, packetindex, prefactor=prefactor)
        else:
            grid = self._grid_wrap(grid)
            values = zeros((grid.get_number_nodes(overall=True),))
            # Evaluate each packet individually
            for j in xrange(self._number_packets):
                S = self._Pis[4][j,:]
                phase = exp(1.0j * S / eps**2)
                values = values + phase * self.slim_recursion(grid, j, prefactor=prefactor)

        return values
