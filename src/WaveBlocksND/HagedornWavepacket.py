"""The WaveBlocks Project

This file contains the class which represents a homogeneous Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
from numpy import zeros, complexfloating, array, sum, transpose, arange, vstack, eye, prod, atleast_2d
from scipy import pi, sqrt, exp, conj, dot
from scipy.linalg import norm, inv, det

from HagedornWavepacketBase import HagedornWavepacketBase
from HyperCubicShape import HyperCubicShape
from Grid import Grid
import GlobalDefaults as GD


class HagedornWavepacket(HagedornWavepacketBase):
    r"""This class represents homogeneous vector valued Hagedorn wavepackets
    :math:`\Psi` with :math:`N` components in :math:`D` space dimensions.
    """

    def __init__(self, parameters):
        r"""
        """
        # TODO: Simpler way to initialize wavepackets. Maybe use a builder?

        self._dimension = parameters["dimension"]
        self._number_components = parameters["ncomponents"]

        self._eps = parameters["eps"]

        # The basis shapes K_i
        self._basis_shapes = []
        # The coefficients c^i
        self._coefficients = []

        for d in xrange(self._number_components):
            # Default basis shapes for all components
            bs = HyperCubicShape( self._dimension*[1] )
            self._basis_shapes.append(bs)

            # A Gaussian
            self._coefficients.append(zeros((bs.get_basis_size(),1), dtype=complexfloating))

        # Cache basis sizes
        self._basis_sizes = [ bs.get_basis_size() for bs in self._basis_shapes ]

        # Default parameters of harmonic oscillator eigenstates
        q = zeros((self._dimension, 1))
        p = zeros((self._dimension, 1))
        Q = eye(self._dimension)
        P = 1.0j * eye(self._dimension)
        S = 0.0

        # The parameter set Pi
        self._Pis = [q, p, Q, P, S]

        # No quadrature set
        self._QE = None


    def __str__(self):
        r""":return: A string describing the Hagedorn wavepacket :math:`\Psi`.
        """
        s = ("Homogeneous Hagedorn wavepacket with "+str(self._number_components)
             +" component(s) in "+str(self._dimension)+" space dimension(s)\n")
        return s


    def get_description(self):
        r"""Return a description of this wavepacket object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "HagedornWavepacket"
        d["dimension"] = self._dimension
        d["ncomponents"] = self._number_components
        d["eps"] = self._eps
        if self._QE is not None:
            d["quadrature"] = self._QE.get_description()
        return d


    def clone(self, keepid=False):
        # Parameters of this packet
        params = {"dimension":   self._dimension,
                  "ncomponents": self._number_components,
                  "eps":         self._eps}

        # Create a new Packet
        other = HagedornWavepacket(params)
        # If we wish to keep the packet ID
        if keepid is True:
            other.set_id(self.get_id())
        # And copy over all (private) data
        # Basis shapes are immutable, no issues with sharing same instance
        other.set_basis_shape(self.get_basis_shape())
        other.set_parameters(self.get_parameters())
        other.set_coefficients(self.get_coefficients())

        return other


    def get_parameters(self, component=None, aslist=False):
        r"""Get the Hagedorn parameter set :math:`\Pi` of the wavepacket :math:`\Psi`.

        :param component: Dummy parameter for API compatibility with the inhomogeneous packets.
        :param aslist: Return a list of :math:`N` parameter tuples. This is for API compatibility
                       with inhomogeneous packets.
        :return: The Hagedorn parameter set :math:`\Pi = (q, p, Q, P, S)` in this order.
        """
        if aslist is True:
            return self._number_components * [ self._Pis ]
        return self._Pis[:]


    def set_parameters(self, Pi, component=None):
        r"""Set the Hagedorn parameters :math:`\Pi` of the wavepacket :math:`\Psi`.

        :param Pi: The Hagedorn parameter set :math:`\Pi = (q, p, Q, P, S)` in this order.
        :param component: Dummy parameter for API compatibility with the inhomogeneous packets.
        """
        self._Pis = [ atleast_2d(array(item, dtype=complexfloating)) for item in Pi ]


    def evaluate_basis_at(self, grid, component, prefactor=False):
        r"""Evaluate the basis functions :math:`\phi_k` recursively at the given nodes :math:`\gamma`.

        :param grid: The grid :math:\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:method:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
                          We need this to choose the correct basis shape.
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: bool, default is ``False``.
        :return: A two-dimensional ndarray :math:`H` of shape :math:`(|\mathcal{K}_i|, |\Gamma|)` where
                 the entry :math:`H[\mu(k), i]` is the value of :math:`\phi_k(\gamma_i)`.
        """
        D = self._dimension

        bas = self._basis_shapes[component]
        bs = self._basis_sizes[component]

        # TODO: Consider putting this into the Grid class as 2nd level API
        # Allow ndarrays for the 'grid' argument
        if isinstance(grid, Grid):
            # The overall number of nodes
            nn = grid.get_number_nodes(overall=True)
            # The grid nodes
            nodes = grid.get_nodes()
        else:
            # The overall number of nodes
            nn = prod(grid.shape[1:])
            # The grid nodes
            nodes = grid

        # Allocate the storage array
        phi = zeros((bs, nn), dtype=complexfloating)

        # Precompute some constants
        q, p, Q, P, S = self._Pis

        Qinv = inv(Q)
        Qbar = conj(Q)
        QQ = dot(Qinv, Qbar)

        # Compute the ground state phi_0 via direct evaluation
        mu0 = bas[tuple(D*[0])]
        phi[mu0,:] = self._evaluate_phi0(self._Pis, nodes, prefactor=False)

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
            # TODO: Use continuous sqrt function
            phi = phi / sqrt(det(Q))

        return phi


    def evaluate_at(self, grid, component=None, prefactor=False):
        r"""Evaluate the Hagedorn wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.

        :param grid: The grid :math:\Gamma` containing the nodes :math:`\gamma`.
        :type grid: A class having a :py:method:`get_nodes(...)` method.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
                          (Defaults to ``None`` for evaluating all components.)
        :param prefactor: Whether to include a factor of :math:`\frac{1}{\sqrt{\det(Q)}}`.
        :type prefactor: bool, default is ``False``.
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i` at the nodes :math:`\gamma`.
        """
        # The global phase part
        phase = exp(1.0j * self._Pis[4] / self._eps**2)

        if component is not None:
            basis = self.evaluate_basis_at(grid, component, prefactor=prefactor)
            values = phase * sum(self._coefficients[component] * basis, axis=0)

        else:
            values = []

            for component in xrange(self._number_components):
                # Note: This is very inefficient! We may evaluate the same basis functions multiple
                #       times. But as long as we don't know that the basis shapes are true subsets
                #       of the largest one, we can not evaluate just all functions in this
                #       maximal set.

                # TODO: Find more efficient way to do this

                basis = self.evaluate_basis_at(grid, component, prefactor=prefactor)
                values.append( phase * sum(self._coefficients[component] * basis, axis=0) )

        return values
