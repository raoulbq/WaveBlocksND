"""The WaveBlocks Project

This file contains code for evaluating inner products
and matrix elements by using standard quadrature rules.
Here we handle the homogeneous case.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, squeeze, conjugate, dot, einsum
from scipy.linalg import sqrtm, inv #, svd, diagsvd

from DirectQuadrature import DirectQuadrature
from InnerProductCompatibility import InnerProductCompatibility

__all__ = ["DirectHomogeneousQuadrature"]


class DirectHomogeneousQuadrature(DirectQuadrature, InnerProductCompatibility):
    r"""
    """

    def __init__(self, QR=None):
        # Pure convenience to allow setting of quadrature rule in constructor
        self.set_qr(QR)


    def __str__(self):
        return "Homogeneous direct quadrature using a " + str(self._QR)


    def get_description(self):
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "DirectHomogeneousQuadrature"
        d["qr"] = self._QR.get_description()
        return d


    def get_kind(self):
        return ("homogeneous",)


    def initialize_packet(self, packet):
        r"""Provide the wavepacket part of the inner product to evaluate.
        Since the quadrature is homogeneous the same wavepacket is used
        for the 'bra' as well as the 'ket' part.

        :param packet: The packet that is used for the 'bra' and 'ket' part.
        """
        self._packet = packet
        self._pacbra = packet

        # Adapt the quadrature nodes and weights
        eps = self._packet.get_eps()
        self._nodes = self.transform_nodes(self._packet.get_parameters(), eps)
        self._weights = self._QR.get_weights()

        # Force a call of 'preprare'
        self._values = None
        self._bases = None
        self._coeffs = None


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
        :math:`\langle \Phi_i | f(x) | \Phi_j \rangle` or the corresponding
        matrix over the basis functions of :math:`\Phi_i` and :math:`\Phi_j`.

        :param rows: A list of all :math:`i` with :math:`0 \leq i \leq N`
                     selecting the :math:`\Phi_i` for which we precompute values.
        :param cols: A list of all :math:`j` with :math:`0 \leq j \leq N`
                     selecting the :math:`\Phi_j` for which we precompute values.
        """
        # Evaluate only the bases we need
        N  = self._packet.get_number_components()
        bases = [ None for n in xrange(N) ]

        for row in rows:
            if bases[row] is None:
                bases[row] = self._packet.evaluate_basis_at(self._nodes, component=row, prefactor=False)

        for col in cols:
            if bases[col] is None:
                bases[col] = self._packet.evaluate_basis_at(self._nodes, component=col, prefactor=False)

        self._bases = bases

        # Operator
        q, p, Q, P, S = self._packet.get_parameters()
        if self._eval_at_once is True:
            self._values = tuple(self._operator(self._nodes, q))
        else:
            self._values = tuple([ self._operator(self._nodes, q, entry=(r,c)) for r in xrange(N) for c in xrange(N) ])
        # Recheck what we got
        assert type(self._values) is tuple
        assert len(self._values) == N**2

        # Coefficients
        self._coeffs = self._packet.get_coefficients()


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

        if QR["transform"] is not None and QR["transform"] is False:
            return QR.get_nodes()

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


    def do_quadrature(self, row, col):
        r"""Evaluates by standard quadrature the integral
        :math:`\langle \Phi_i | f | \Phi_j \rangle` for a polynomial
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi_j` of :math:`\Psi`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}_j|`.
        """
        D = self._packet.get_dimension()
        eps = self._packet.get_eps()
        N  = self._packet.get_number_components()
        # Main part of the integrand
        factor = (eps**D * self._weights * self._values[row*N + col]).reshape((-1,))
        # Sum up matrices over all quadrature nodes
        M = einsum("k,ik,jk", factor, conjugate(self._bases[row]), self._bases[col])
        return M
