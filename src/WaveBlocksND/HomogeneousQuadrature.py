"""The WaveBlocks Project

This file contains code for the homogeneous quadrature of Hagedorn wavepackets.
The class defined here can compute brakets, inner products and expectation
values and the matrix elements of an arbitrary operator.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, complexfloating, sum, cumsum, squeeze, conjugate, dot, outer, ndarray
from scipy.linalg import sqrtm, inv, svd, diagsvd

from Quadrature import Quadrature


class HomogeneousQuadrature(Quadrature):

    def __init__(self, QR=None):
        if QR is not None:
            self.set_qr(QR)
        else:
            self._QR = None


    def __str__(self):
        return "Homogeneous quadrature using a " + str(self._QR)


    def get_description(self):
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "HomogeneousQuadrature"
        d["qr"] = self._QR.get_description()
        return d


    def transform_nodes(self, Pi, eps, QR=None):
        r"""Transform the quadrature nodes :math:`\gamma` such that they
        fit the given wavepacket :math:`\Phi\left[\Pi\right]`.

        :param Pi: The parameter set :math:`\Pi` of the wavepacket.
        :param eps: The value of :math:`\varepsilon` of the wavepacket.
        :param QR: An optional quadrature rule :math:`\Gamma = (\gamma, \omega)` providing
                   the nodes. If not given the internal quadrature rule will be used.
        :return: A two-dimensional ndarray of shape :math:`(D, |\Gamma|)` where
                 :math:`|\Gamma|` denotes the total number of quadrature nodes.
        """
        if QR is None:
            QR = self._QR

        q, p, Q, P, S = Pi

        # Compute the affine transformation
        Q0 = inv(dot(Q, conjugate(Q.T)))
        Qs = inv(sqrtm(Q0))

        # TODO: Avoid sqrtm and inverse computation, use svd
        # Untested code:
        # D = Q.shape[0]
        # U, S, Vh = svd(inv(Q))
        # Sinv = 1.0 / S
        # Sinv = diagsvd(Sinv, D, D)
        # Qs = dot(conjugate(Vh.T), dot(Sinv, Vh))

        # And transform the nodes
        nodes = q + eps * dot(Qs, QR.get_nodes())
        return nodes.copy()


    def quadrature(self, packet, operator=None, summed=False, component=None, diag_component=None):
        r"""Performs the quadrature of :math:`\langle\Psi|f|\Psi\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param packet: The wavepacket :math:`\Psi`.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N}`.
        :param summed: Whether to sum up the individual integrals :math:`\langle\Phi_i|f_{i,j}|\Phi_j\rangle`.
        :type summed: bool, default is ``False``.
        :param component: Request only the i-th component of the result. Remember that :math:`i \in [0, N^2-1]`.
        :param diag_component: Request only the i-th component from the diagonal entries, here :math:`i \in [0, N-1]`.
                               Note that ``component`` takes precedence over ``diag_component`` if both are supplied. (Which is discouraged)
        :return: The value of the braket :math:`\langle\Psi|f|\Psi\rangle`. This is either a scalar value or
                 a list of :math:`N^2` scalar elements depending on the value of ``summed``.
        """
        # TODO: Consider adding 'is_diagonal' flag to make computations cheaper if we know the operator is diagonal
        D = packet.get_dimension()
        eps = packet.get_eps()

        nodes = self.transform_nodes(packet.get_parameters(), eps)
        weights = self._QR.get_weights()

        N = packet.get_number_components()
        K = [ bs.get_basis_size() for bs in packet.get_basis_shape() ]

        coeffs = packet.get_coefficients()

        # Avoid unnecessary computations of other components
        if component is not None:
            rows = [ component // N ]
            cols = [ component % N ]
        elif diag_component is not None:
            rows = [ diag_component ]
            cols = [ diag_component ]
        else:
            rows = xrange(N)
            cols = xrange(N)

        # TODO: Make this more efficient, only compute values needed at each (r,c) step.
        #       For this, 'operator' must support the 'component=(r,c)' option.
        if operator is None:
            # Operator is None is interpreted as identity transformation
            operator = lambda nodes, entry=None: ones(nodes.shape[1]) if entry[0] == entry[1] else zeros(nodes.shape[1])
            values = tuple([ operator(nodes, entry=(r,c)) for r in xrange(N) for c in xrange(N) ])
        else:
            values = tuple( operator(nodes) )

        # Recheck what we got
        assert type(values) is tuple
        assert len(values) == N**2

        # Evaluate only the bases we need:
        bases = [ None for n in xrange(N) ]

        for row in rows:
            if bases[row] is None:
                bases[row] = packet.evaluate_basis_at(nodes, component=row, prefactor=False)

        for col in cols:
            if bases[col] is None:
                bases[col] = packet.evaluate_basis_at(nodes, component=col, prefactor=False)

        # Compute the quadrature
        result = []

        for row in rows:
            for col in cols:
                M = zeros((K[row],K[col]), dtype=complexfloating)

                factor = squeeze(eps**D * weights * values[row*N + col])

                # Summing up matrices over all quadrature nodes
                for r in xrange(self._QR.get_number_nodes()):
                    M += factor[r] * outer(conjugate(bases[row][:,r]), bases[col][:,r])

                # And include the coefficients as conj(c).T*M*c
                result.append(dot(conjugate(coeffs[row]).T, dot(M, coeffs[col])))

        if summed is True:
            result = sum(result)
        elif component is not None:
            # Do not return a list for quadrature of specific single components
            result = result[0]

        return result


    def build_matrix(self, packet, operator=None):
        r"""Calculate the matrix elements of :math:`\langle\Psi|f|\Psi\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        The matrix is computed without including the coefficients :math:`c^i_k`.

        :param packet: The wavepacket :math:`\Psi`.
        :param operator: A matrix-valued function :math:`f(q, x): \mathbb{R} \times \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N}`.
        :return: A square matrix of size :math:`\sum_i^N |\mathcal{K}_i| \times \sum_j^N |\mathcal{K}_j|`.
        """
        # TODO: Consider adding 'is_diagonal' flag to make computations cheaper if we know the operator is diagonal
        D = packet.get_dimension()
        eps = packet.get_eps()

        nodes = self.transform_nodes(packet.get_parameters(), eps)
        weights = self._QR.get_weights()

        N = packet.get_number_components()
        K = [ bs.get_basis_size() for bs in packet.get_basis_shape() ]

        bases = [ packet.evaluate_basis_at(nodes, component=n, prefactor=False) for n in xrange(N) ]

        # The partition scheme of the block vectors and block matrix
        partition = [0] + list(cumsum(K))

        # TODO: Make this more efficient, only compute values needed at each (r,c) step.
        #       For this, 'operator' must support the 'entry=(r,c)' option.
        if operator is None:
            # Operator is None is interpreted as identity transformation
            operator = lambda nodes, entry=None: ones(nodes.shape[1]) if entry[0] == entry[1] else zeros(nodes.shape[1])
            values = tuple([ operator(nodes, entry=(r,c)) for r in xrange(N) for c in xrange(N) ])
        else:
            # TODO: operator should be only f(nodes) but we can not fix this currently
            q, p, Q, P, S = packet.get_parameters()
            values = tuple( operator(nodes, q) )

        # Recheck what we got
        assert type(values) is tuple
        assert len(values) == N**2

        # Compute the matrix elements
        result = zeros((sum(K),sum(K)), dtype=complexfloating)

        for row in xrange(N):
            for col in xrange(N):
                M = zeros((K[row],K[col]), dtype=complexfloating)

                factor = squeeze(eps**D * weights * values[row*N + col])

                # Sum up matrices over all quadrature nodes
                for r in xrange(self._QR.get_number_nodes()):
                    M += factor[r] * outer(conjugate(bases[row][:,r]), bases[col][:,r])

                # Put the result into the global storage
                result[partition[row]:partition[row+1], partition[col]:partition[col+1]] = M

        return result
