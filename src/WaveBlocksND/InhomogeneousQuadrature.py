"""The WaveBlocks Project

This file contains code for the inhomogeneous (or mixing) quadrature of two
wavepackets. The class defined here can compute brakets, inner products and
expectation values and the matrix elements of an arbitrary operator.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, complexfloating, sum, cumsum, squeeze, imag, conjugate, outer, dot, ndarray
from scipy import exp
from scipy.linalg import sqrtm, inv, det

from Quadrature import Quadrature


class InhomogeneousQuadrature(Quadrature):

    def __init__(self, QR=None):
        if QR is not None:
            self.set_qr(QR)
        else:
            self._QR = None


    def __str__(self):
        return "Inhomogeneous quadrature using a " + str(self._QR)


    def get_description(self):
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "InhomogeneousQuadrature"
        d["qr"] = self._QR.get_description()
        return d


    def transform_nodes(self, Pibra, Piket, eps, QR=None):
        r"""Transform the quadrature nodes :math:`\gamma` such that they
        fit the given wavepackets :math:`\Phi\left[\Pi_i\right]` and
        :math:`\Phi^\prime\left[\Pi_j\right]` best.

        :param Pibra: The parameter set :math:`\Pi_i` from the bra part wavepacket.
        :param Piket: The parameter set :math:`\Pi_j` from the ket part wavepacket.
        :param eps: The value of :math:`\varepsilon` of the wavepacket.
        :param QR: An optional quadrature rule :math:`\Gamma = (\gamma, \omega)` providing
                   the nodes. If not given the internal quadrature rule will be used.
        :return: A two-dimensional ndarray of shape :math:`(D, |\Gamma|)` where
                 :math:`|\Gamma|` denotes the total number of quadrature nodes.
        """
        if QR is None:
            QR = self._QR

        # Mix the parameters to compute the affine transformation
        q0, Qs = self.mix_parameters(Pibra, Piket)

        # And transform the nodes
        nodes = q0 + eps * dot(Qs, QR.get_nodes())
        return nodes.copy()


    def mix_parameters(self, Pibra, Piket):
        r"""Mix the two parameter sets :math:`\Pi_i` and :math:`\Pi_j`
        from the bra and the ket wavepackets :math:`\Phi\left[\Pi_i\right]`
        and :math:`\Phi^\prime\left[\Pi_j\right]`.

        :param Pibra: The parameter set :math:`\Pi_i` from the bra part wavepacket.
        :param Piket: The parameter set :math:`\Pi_j` from the ket part wavepacket.
        :return: The mixed parameters :math:`q_0` and :math:`Q_S`. (See the theory for details.)
        """
        # <Pibra | ... | Piket>
        (qr, pr, Qr, Pr, Sr) = Pibra
        (qc, pc, Qc, Pc, Sc) = Piket

        # Mix the parameters
        Gr = dot(Pr, inv(Qr))
        Gc = dot(Pc, inv(Qc))

        r = imag(Gc - conjugate(Gr.T))
        s = imag(dot(Gc, qc) - dot(conjugate(Gr.T), qr))

        q0 = dot(inv(r), s)
        Q0 = 0.5 * r

        # Here we can not avoid the matrix root by using svd
        Qs = inv(sqrtm(Q0))

        return (q0, Qs)


    def quadrature(self, pacbra, packet=None, operator=None, summed=False, component=None, diag_component=None):
        r"""Performs the quadrature of :math:`\langle\Psi|f|\Psi^\prime\rangle` for a general
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param pacbra: The wavepacket :math:`\Psi` from the bra with :math:`N` components.
        :param packet: The wavepacket :math:`\Psi^\prime` from the ket with :math:`N^\prime` components.
        :param operator: A matrix-valued function :math:`f(x): \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :param summed: Whether to sum up the individual integrals :math:`\langle\Phi_i|f_{i,j}|\Phi^\prime_j\rangle`.
        :type summed: bool, default is ``False``.

        :param component: Request only the i-th component of the result. Remember that :math:`i \in [0, N \cdot N^\prime-1]`.
        :param diag_component: Request only the i-th component from the diagonal entries, here :math:`i \in [0, N^\prime-1]`.
                               Note that ``component`` takes precedence over ``diag_component`` if both are supplied. (Which is discouraged)
        :return: The value of the braket :math:`\langle\Psi|f|\Psi^\prime\rangle`. This is either a scalar value or
                 a list of :math:`N \cdot N^\prime` scalar elements depending on the value of ``summed``.
        """
        # Allow to ommit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        # TODO: Consider adding 'is_diagonal' flag to make computations cheaper if we know the operator is diagonal
        # TODO: Should raise Exceptions if pacbra and packet are incompatible wrt N, K etc
        D = packet.get_dimension()
        eps = packet.get_eps()

        weights = self._QR.get_weights()

        # Packets can have different number of components
        Nbra = pacbra.get_number_components()
        Nket = packet.get_number_components()

        # Packets can also have different basis sizes
        Kbra = [ bs.get_basis_size() for bs in pacbra.get_basis_shape() ]
        Kket = [ bs.get_basis_size() for bs in packet.get_basis_shape() ]

        Pibra = pacbra.get_parameters(aslist=True)
        Piket = packet.get_parameters(aslist=True)

        coeffbra = pacbra.get_coefficients()
        coeffket = packet.get_coefficients()

        # Avoid unnecessary computations of other components
        if component is not None:
            rows = [ component // Nket ]
            cols = [ component % Nket ]
        elif diag_component is not None:
            rows = [ diag_component ]
            cols = [ diag_component ]
        else:
            rows = xrange(Nbra)
            cols = xrange(Nket)

        # Operator is None is interpreted as identity transformation
        if operator is None:
            operator = lambda nodes, entry=None: ones(nodes.shape[1]) if entry[0] == entry[1] else zeros(nodes.shape[1])

        # Compute the quadrature
        result = []

        for row in rows:
            for col in cols:
                M = zeros((Kbra[row],Kket[col]), dtype=complexfloating)

                # Transform nodes and evaluate bases
                nodes = self.transform_nodes(Pibra[row], Piket[col], eps)
                basisr = pacbra.evaluate_basis_at(nodes, component=row, prefactor=True)
                basisc = packet.evaluate_basis_at(nodes, component=col, prefactor=True)

                # Operator should support the component notation for efficiency
                values = operator(nodes, entry=(row,col))

                # Recheck what we got
                assert type(values) is ndarray
                assert values.shape == (1,self._QR.get_number_nodes())

                Pimix = self.mix_parameters(Pibra[row], Piket[col])
                factor = squeeze(eps**D * values * weights * det(Pimix[1]))

                # Summing up matrices over all quadrature nodes
                for r in xrange(self._QR.get_number_nodes()):
                    M += factor[r] * outer(conjugate(basisr[:,r]), basisc[:,r])

                # Compute global phase difference
                phase = exp(1.0j/eps**2 * (Piket[col][4]-conjugate(Pibra[row][4])))

                # And include the coefficients as conj(c).T*M*c
                result.append(phase * dot(conjugate(coeffbra[row]).T, dot(M, coeffket[col])))

        if summed is True:
            result = sum(result)
        elif component is not None:
            # Do not return a list for quadrature of specific single components
            result = result[0]

        return result


    def build_matrix(self, pacbra, packet=None, operator=None):
        r"""Calculate the matrix elements of :math:`\langle\Psi|f|\Psi^\prime\rangle`
        for a general function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.
        The matrix is computed without including the coefficients :math:`c^i_k`.

        :param pacbra: The wavepacket :math:`\Psi` from the bra with :math:`N` components.
        :param packet: The wavepacket :math:`\Psi^\prime` from the ket with :math:`N^\prime` components.
        :param operator: A matrix-valued function :math:`f(q, x): \mathbb{R} \times \mathbb{R}^D \rightarrow \mathbb{R}^{N \times N^\prime}`.
        :return: A matrix of size :math:`\sum_i^N |\mathcal{K}_i| \times \sum_j^{N^\prime} |\mathcal{K}^\prime_j|`.
        """
        # Allow to ommit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        # TODO: Consider adding 'is_diagonal' flag to make computations cheaper if we know the operator is diagonal
        # TODO: Should raise Exceptions if pacbra and packet are incompatible wrt N, K etc
        D = packet.get_dimension()
        eps = packet.get_eps()

        weights = self._QR.get_weights()

        # Packets can have different number of components
        Nbra = pacbra.get_number_components()
        Nket = packet.get_number_components()
        # Packets can also have different basis size
        Kbra = [ bs.get_basis_size() for bs in pacbra.get_basis_shape() ]
        Kket = [ bs.get_basis_size() for bs in packet.get_basis_shape() ]
        # The partition scheme of the block vectors and block matrix
        partitionb = [0] + list(cumsum(Kbra))
        partitionk = [0] + list(cumsum(Kket))

        Pibra = pacbra.get_parameters(aslist=True)
        Piket = packet.get_parameters(aslist=True)

        result = zeros((sum(Kbra),sum(Kket)), dtype=complexfloating)

        # Operator is None is interpreted as identity transformation
        if operator is None:
            operator = lambda nodes, dummy, entry=None: ones(nodes.shape[1]) if entry[0] == entry[1] else zeros(nodes.shape[1])

        for row in xrange(Nbra):
            for col in xrange(Nket):
                M = zeros((Kbra[row],Kket[col]), dtype=complexfloating)

                # Transform nodes and evaluate bases
                nodes = self.transform_nodes(Pibra[row], Piket[col], eps)
                basisr = pacbra.evaluate_basis_at(nodes, component=row, prefactor=True)
                basisc = packet.evaluate_basis_at(nodes, component=col, prefactor=True)

                Pimix = self.mix_parameters(Pibra[row], Piket[col])
                # Operator should support the component notation for efficiency
                # TODO: operator should be only f(nodes) but we can not fix this currently
                values = operator(nodes, Pimix[0], entry=(row,col))

                # Recheck what we got
                assert type(values) is ndarray
                assert values.shape == (1,self._QR.get_number_nodes())

                factor = squeeze(eps**D * values * weights * det(Pimix[1]))

                # Sum up matrices over all quadrature nodes
                for r in xrange(self._QR.get_number_nodes()):
                    M += factor[r] * outer(conjugate(basisr[:,r]), basisc[:,r])

                # Compute global phase difference
                phase = exp(1.0j/eps**2 * (Piket[col][4]-conjugate(Pibra[row][4])))

                result[partitionb[row]:partitionb[row+1], partitionk[col]:partitionk[col+1]] = phase * M

        return result
