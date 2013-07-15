"""The WaveBlocks Project



@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, eye, integer, complexfloating, atleast_2d, concatenate, hstack, vstack, squeeze


from LinearCombinationOfWavepackets import LinearCombinationOfWavepackets
from HagedornWavepacket import HagedornWavepacket

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
        self._coefficients = zeros((number_packets,0), dtype=complexfloating)
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
        curlen, cursize = self._coefficients.shape
        diff = newsize - cursize
        print(diff)
        if diff > 0:
            z = zeros((curlen,diff), dtype=complexfloating)
            self._coefficients = hstack([self._coefficients, z])
        elif diff < 0:
            self._coefficients[:,:newsize]


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
        self._coefficients = vstack([self._coefficients, c.reshape((1,-1))])
        # Store the parameter set
        D = self._dimension
        qs, ps, Qs, Ps, Ss = self._Pis
        q, p, Q, P, S = packet.get_parameters(component=0)
        concatenate([qs, q.reshape((1,D))], axis=0)
        concatenate([ps, p.reshape((1,D))], axis=0)
        concatenate([Qs, Q.reshape((1,D,D))], axis=0)
        concatenate([Ps, P.reshape((1,D,D))], axis=0)
        concatenate([Ss, S.reshape((1,1))], axis=0)
        # Store the linear combination coefficient
        self._lc_coefficients = vstack([self._lc_coefficients, atleast_2d(coefficient)])


    def add_wavepackets(self, packetlist, coefficients):
        r"""
        """
        for j, packet in enumerate(packetlist):
            self.add_wavepacket(packet, coefficients[j])


    def remove_wavepacket(self, index):
        r"""
        :raise: :py:class:`NotImplementedError` Abstract interface.
        """
        raise NotImplementedError("'LinearCombinationOfWavepackets' is an abstract interface.")


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
        cj = self._coefficients[packetindex,:]
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


    def set_coefficient(self, packetindex, index, value):
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
        key = basis_shape[packetindex]
        self._coefficients[packetindex][key] = value


    def get_coefficient(self, packetindex, index):
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
        return self._coefficients[packetindex][key]


    # def set_coefficients(self, values, component=None):
    #     r"""Update all the coefficients :math:`c` of :math:`\Psi` or update
    #     the coefficients :math:`c^i` of the components :math:`\Phi_i` only.

    #     Note: this method copies the data arrays.

    #     :param values: The new values of the coefficients :math:`c^i` of :math:`\Phi_i`.
    #     :type values: An ndarray of suitable shape or a list of ndarrays.
    #     :param component: The index :math:`i` of the component we want to update with new coefficients.
    #     :type component: int (Default is ``None`` meaning all)
    #     :raise: :py:class:`ValueError` For invalid component indices :math:`i`.
    #     """
    #     if component is None:
    #         if len(values) != self._number_components:
    #             raise ValueError("Too less or too many data provided.")

    #         for index, value in enumerate(values):
    #             bs = self._basis_sizes[index]
    #             self._coefficients[index] = value.copy().reshape((bs,1))
    #     else:
    #         if component > self._number_components-1 or component < 0:
    #             raise ValueError("There is no component with index "+str(component)+".")

    #         bs = self._basis_sizes[component]
    #         self._coefficients[component] = values.copy().reshape((bs,1))


    def get_coefficients(self, packetindex):
        r"""Returns the coefficients :math:`c^i` for some component :math:`\Phi_i` of
        :math:`\Psi` or all the coefficients :math:`c` of all components.

        :param component: The index :math:`i` of the component we want to retrieve.
        :type component: int (Default is ``None`` meaning all)
        :return: A single ndarray with the coefficients of the given component or
                 a list containing the ndarrays for each component. Each ndarray
                 is two-dimensional with a shape of :math:`(|\mathfrak{K}_i|, 1)`.
        :raise: :py:class:`ValueError` For invalid component indices :math:`i`.
        """
        if packetindex > self._number_packets-1 or packetindex < 0:
            raise ValueError("There is no packet with index "+str(packetindex)+".")

        bs = self._basis_sizes[packetindex]
        return self._coefficients[packetindex,:bs]


    def get_coefficient_vector(self, packetindex=None):
        r"""Retrieve the coefficients for all components :math:`\Phi_i` simultaneously.

        :param component: The component :math:`i` whose coefficients we request. (Default is
                          ``None`` which means to return the coefficients for all components.
        :type component: int
        :return: The coefficients :math:`c^i` of all components
                 :math:`\Phi_i` stacked into a single long column vector.
        """
        if packetindex > self._number_packets-1 or packetindex < 0:
            raise ValueError("There is no packet with index "+str(packetindex)+".")

        if packetindex is None:
            packetindex = slice(None)

        return self._coefficients[packetindex][:]


    def set_coefficient_vector(self, packetindex, vector):
        """Set the coefficients for all components :math:`\Phi_i` simultaneously.

        :param vector: The coefficients of all components as a single long column vector.
        :type vector: A two-dimensional ndarray of appropriate shape.
        """
        if packetindex > self._number_packets-1 or packetindex < 0:
            raise ValueError("There is no packet with index "+str(packetindex)+".")

        # TODO: Validate the size
        self._coefficients[packetindex] = squeeze(vector)


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
            #elif k == "adQ":
            #    Pilist.append(array(self._get_sqrt(component).get(), dtype=complexfloating))
            else:
                raise KeyError("Invalid parameter key: "+str(key))

        return Pilist


    def get_eps(self):
        r"""Retrieve the semi-classical scaling parameter :math:`\varepsilon` of the wavepacket.

        :return: The value of :math:`\varepsilon`.
        """
        return self._eps
