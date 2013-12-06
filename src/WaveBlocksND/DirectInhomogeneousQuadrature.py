"""The WaveBlocks Project

This file contains code for evaluating inner products
and matrix elements by using standard quadrature rules.
Here we handle the inhomogeneous case.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, squeeze, imag, conjugate, dot, ndarray, einsum
from scipy import exp
from scipy.linalg import sqrtm, inv, det

from DirectQuadrature import DirectQuadrature

__all__ = ["DirectInhomogeneousQuadrature"]


class DirectInhomogeneousQuadrature(DirectQuadrature):
    r"""
    """

    def __init__(self, QR=None):
        # Pure convenience to allow setting of quadrature rule in constructor
        if QR is not None:
            self.set_qr(QR)
        else:
            self._QR = None


    def __str__(self):
        return "Inhomogeneous direct quadrature using a " + str(self._QR)


    def get_description(self):
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "DirectInhomogeneousQuadrature"
        d["qr"] = self._QR.get_description()
        return d


    def initialize_packet(self, pacbra, packet=None):
        r"""Provide the wavepacket parts of the inner product to evaluate.
        Since the quadrature is inhomogeneous different wavepackets can be
        used for the 'bra' as well as the 'ket' part.

        :param pacbra: The packet that is used for the 'bra' part.
        :param packet: The packet that is used for the 'ket' part.
        """
        # Allow to omit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        self._pacbra = pacbra
        self._packet = packet

        # Adapt the quadrature nodes and weights
        self._weights = self._QR.get_weights()

        # Force a call of 'preprare'
        self._coeffbra = None
        self._coeffket = None


    def initialize_operator(self, operator=None, matrix=False, eval_at_once=False):
        r"""Provide the operator part of the inner product to evaluate.
        This function initializes the operator used for quadratures
        and for building matrices.

        :param operator: The operator of the inner product.
                         If ``None`` a suitable identity is used.
        :param matrix: Set this to ``True`` (Default is ``False``) in case
                       we want to compute the matrix elements.
                       For nasty technical reasons we can not yet unify
                       the operator call syntax.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        """
        # TODO: Make this more efficient, only compute values needed at each (r,c) step.
        #       For this, 'operator' must support the 'component=(r,c)' option.
        # Operator is None is interpreted as identity transformation
        if operator is None:
            self._operator = lambda nodes, dummy, entry=None: ones((1,nodes.shape[1])) if entry[0] == entry[1] else zeros((1,nodes.shape[1]))
        else:
            if matrix is False:
                self._operator = lambda nodes, dummy, entry=None: operator(nodes, entry=entry)
            else:
                self._operator = operator
        self._eval_at_once = eval_at_once


    def prepare(self, rows, cols):
        r"""Precompute some values needed for evaluating the quadrature
        :math:`\langle \Phi_i | f(x) | \Phi^\prime_j \rangle` or the corresponding
        matrix over the basis functions of :math:`\Phi_i` and :math:`\Phi^\prime_j`.

        :param rows: A list of all :math:`i` with :math:`0 \leq i \leq N`
                     selecting the :math:`\Phi_i` for which we precompute values.
        :param cols: A list of all :math:`j` with :math:`0 \leq j \leq N`
                     selecting the :math:`\Phi^\prime_j` for which we precompute values.
        """
        # Coefficients
        self._coeffbra = self._pacbra.get_coefficients()
        self._coeffket = self._packet.get_coefficients()


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

        if QR["transform"] is not None and QR["transform"] is False:
            return QR.get_nodes()

        # Mix the parameters to compute the affine transformation
        q0, Qs = self.mix_parameters(Pibra, Piket)

        # And transform the nodes
        nodes = q0 + eps * dot(Qs, QR.get_nodes())
        return nodes.copy()


    def do_quadrature(self, row, col):
        r"""Evaluates by standard quadrature the integral
        :math:`\langle \Phi_i | f | \Phi^\prime_j \rangle` for a polynomial
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}^\prime_j|`.
        """
        D = self._packet.get_dimension()
        N = self._packet.get_number_components()
        eps = self._packet.get_eps()
        # Mix wavepacket parameters
        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        Pimix = self.mix_parameters(Pibra, Piket)
        # Transform nodes and evaluate bases
        nodes = self.transform_nodes(Pibra, Piket, eps)
        basisr = self._pacbra.evaluate_basis_at(nodes, component=row, prefactor=True)
        basisc = self._packet.evaluate_basis_at(nodes, component=col, prefactor=True)
        # Operator should support the component notation for efficiency
        if self._eval_at_once is True:
            # TODO: Sure, this is inefficient, but we can not do better right now.
            values = self._operator(nodes, Pimix[0])[row*N+col]
        else:
            values = self._operator(nodes, Pimix[0], entry=(row,col))
        # Recheck what we got
        assert type(values) is ndarray
        assert values.shape == (1,self._QR.get_number_nodes())
        # Main part of the integrand
        factor = (eps**D * values * self._weights * det(Pimix[1])).reshape((-1,))
        # Sum up matrices over all quadrature nodes
        M = einsum("k,ik,jk", factor, conjugate(basisr), basisc)
        # Compute global phase difference
        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))
        return phase * M
