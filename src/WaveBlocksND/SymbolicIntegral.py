"""The WaveBlocks Project

Use a symbolic exact formula for computing the inner product
between two semi-classical wavepackets. The formula is
constructed explicitely for the inhomogeneous case.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze, conjugate, sqrt, ones, zeros, complexfloating, arange, isnan
from scipy import exp
from scipy.misc import factorial
from scipy.special import binom
#from scipy.special.orthogonal import eval_hermite

from InnerProduct import InnerProductException
from Quadrature import Quadrature

__all__ = ["SymbolicIntegral"]


class SymbolicIntegral(Quadrature):
    r"""
    """

    def __init__(self, doraise=False, *unused, **kunused):
        r"""
        """
        self._doraise = doraise
        # Drop any argument, we do not need a qr instance.


    def __str__(self):
        return "Inhomogeneous inner product computed using a symbolic exact formula."


    def get_description(self):
        r"""Return a description of this integral object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "SymbolicIntegral"
        return d


    def initialize_packet(self, pacbra, packet=None):
        r"""Provide the wavepacket parts of the inner product to evaluate.
        Since the formula is for the inhomogeneous case explicitly, different
        wavepackets can be used for the 'bra' as well as the 'ket' part.

        :param pacbra: The packet that is used for the 'bra' part.
        :param packet: The packet that is used for the 'ket' part.
        :raises: :py:class:`ValueError` if the dimension of :math:`\Psi` is not 1.
        """
        # Allow to ommit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        if not pacbra.get_dimension() == 1:
            raise ValueError("The 'SymbolicIntegral' applies in the 1D case only.")

        self._pacbra = pacbra
        self._packet = packet


    def initialize_operator(self, operator=None, matrix=False, eval_at_once=False):
        r"""Provide the operator part of the inner product to evaluate.
        This function initializes the operator used for quadratures
        and for building matrices.

        Note that the symbolic solution can not handle operators at all.

        :param operator: The operator of the inner product.
                         If ``None`` a suitable identity is used.
        :param matrix: Set this to ``True`` (Default is ``False``) in case
                       we want to compute the matrix elements.
                       For nasty technical reasons we can not yet unify
                       the operator call syntax.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
                             Since we do not support operators at all, it has no effect.
        :type eval_at_once: Boolean, default is ``False``.):
        """
        # Operator is None is interpreted as identity transformation
        if operator is None:
            self._operator = lambda nodes, dummy, entry=None: ones((1,nodes.shape[1])) if entry[0] == entry[1] else zeros((1,nodes.shape[1]))
        else:
            raise ValueError("The 'SymbolicIntegral' can not handle operators.")


    def prepare(self, rows, cols):
        r"""Precompute some values needed for evaluating the integral
        :math:`\langle \Phi_i | \Phi^\prime_j \rangle` or the corresponding
        matrix over the basis functions of :math:`\Phi_i` and :math:`\Phi^\prime_j`.
        Note that this function does nothing in the current implementation.

        :param rows: A list of all :math:`i` with :math:`0 \leq i \leq N`
                     selecting the :math:`\Phi_i` for which we precompute values.
        :param cols: A list of all :math:`j` with :math:`0 \leq j \leq N`
                     selecting the :math:`\Phi^\prime_j` for which we precompute values.
        """
        pass


    def _evaluate_hermite(self, N, x):
        r"""Evaluate the first `N` Hermite polynomials at once by exploiting
        the recursion relation.

        :param N: The maximal order :math:`N` for which we evaluate :math:`H_N`.
        :param x: The argument :math:`x` of the Hermite polynomial :math:`H_n(x)`.
        :return: A list :math:`[H_0, H_1, \ldots, H_N]` of all Hermite polynomials
                 up to order :math:`N` inclusive.
        """
        H = {}
        H[-1] = 0.0
        H[0] = 1.0
        for n in xrange(N+1):
            H[n+1] = 2.0*x*H[n] - 2.0*n*H[n-1]

        H.pop(-1)
        return H


    def exact_result_ground(self, Pibra, Piket, eps):
        r"""Compute the overlap integral :math:`\langle \phi_0 | \phi_0 \rangle` of
        the groundstate :math:`\phi_0` by using the symbolic formula:

        .. math::
            \langle \phi_0 | \phi_0 \rangle =
            \sqrt{\frac{-2 i}{Q_2 \overline{P_1} - P_2 \overline{Q_1}}} \cdot
              \exp \Biggl(
                \frac{i}{2 \varepsilon^2}
                \frac{Q_2 \overline{Q_1} \left(p_2-p_1\right)^2 + P_2 \overline{P_1} \left(q_2-q_1\right)^2}
                      {\left(Q_2 \overline{P_1} - P_2 \overline{Q_1}\right)}
              \\
              -\frac{i}{\varepsilon^2}
              \frac{\left(q_2-q_1\right) \left( Q_2 \overline{P_1} p_2 - P_2 \overline{Q_1} p_1\right)}
                   {\left(Q_2 \overline{P_1} - P_2 \overline{Q_1}\right)}
              \Biggr)

        Note that this is an internal method and usually there is no
        reason to call it from outside.

        :param Pibra: The parameter set :math:`\Pi = \{q_1,p_1,Q_1,P_1\}` of the bra :math:`\langle \phi_0 |`.
        :param Piket: The parameter set :math:`\Pi^\prime = \{q_2,p_2,Q_2,P_2\}` of the ket :math:`| \phi_0 \rangle`.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon`.
        :return: The value of the integral :math:`\langle \phi_0 | \phi_0 \rangle`.
        """
        q1, p1, Q1, P1 = Pibra
        q2, p2, Q2, P2 = Piket
        hbar = eps**2
        X = Q2*conjugate(P1) - P2*conjugate(Q1)
        I = sqrt(-2.0j/X) * exp( 1.0j/(2*hbar) * (Q2*conjugate(Q1)*(p2 - p1)**2 + P2*conjugate(P1)*(q2 - q1)**2) / X
                                -1.0j/hbar *     ((q2 - q1)*(Q2*conjugate(P1)*p2 - P2*conjugate(Q1)*p1)) / X
                               )
        return I


    def exact_result_higher(self, Pibra, Piket, eps, k, l):
        r"""Compute the overlap integral :math:`\langle \phi_k | \phi_l \rangle` of
        two states :math:`\phi_k` and :math:`\phi_l` by using the symbolic formula:

        .. math::
            \langle \phi_k | \phi_l \rangle =
            \frac{1}{\sqrt{k!l!}} 2^{-\frac{k+l}{2}} \langle \phi_0 | \phi_0 \rangle \cdot
            \left(i \overline{ P_1} Q_2 - i \overline{Q_1} P_2\right)^{-\frac{k+l}{2}} \cdot \\
            \sum_{j=0}^{\min\left(k,l\right)}
              \Biggl(
                \binom{k}{j} \binom{l}{j} j! 4^j
                \left(i Q_2  P_1 - i Q_1  P_2\right)^{\frac{k-j}{2}}
                \left(i \overline{Q_2 P_1} - i\overline{Q_1 P_2}\right)^{\frac{l-j}{2}}
                \\
                \cdot H_{k-j}\left(-\frac{1}{\varepsilon}
                              \frac{Q_2\left(p_1-p_2\right)-P_2\left(q_1-q_2\right)}
                                   {\sqrt{Q_2 P_1 - Q_1 P_2}\sqrt{\overline{P_1}Q_2-\overline{Q_1} P_2}}\right)
                \\
                \cdot H_{l-j}\left(\frac{1}{\varepsilon}
                             \frac{\overline{ P_1}\left(q_1-q_2\right)-\overline{Q_1}\left(p_1-p_2\right)}
                                  {\sqrt{\overline{Q_2 P_1}-\overline{Q_1 P_2}}\sqrt{\overline{ P_1}Q_2-\overline{Q_1} P_2}}\right)
              \Biggr)

        Note that this is an internal method and usually there is no
        reason to call it from outside.

        :param Pibra: The parameter set :math:`\Pi = \{q_1,p_1,Q_1,P_1\}` of the bra :math:`\langle \phi_k |`.
        :param Piket: The parameter set :math:`\Pi^\prime = \{q_2,p_2,Q_2,P_2\}` of the ket :math:`| \phi_l \rangle`.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon`.
        :param k: Index :math:`k` of the wavepacket basis function :math:`\phi_k`.
        :param l: Index :math:`l` of the wavepacket basis function :math:`\phi_l`.
        :return: The value of the integral :math:`\langle \phi_k | \phi_l \rangle`.
        """
        # < phi_k[Pi1] | phi_l[Pi2] >
        q1, p1, Q1, P1 = Pibra
        q2, p2, Q2, P2 = Piket

        # If both parameter sets are identical, we are back in the
        # homogeneous case where the phi are orthonormal.
        # This early returns just serves to avoid introducing NaN
        # elements in arg1 and arg2 further below. It is allowed to
        # compare floats on exact equality because the inhomogeneous
        # formula works even if the floats differ only by tiny amounts.
        if q1 == q2 and p1 == p2 and Q1 == Q2 and P1 == P2:
            return 1.0 if k == l else 0.0

        # TODO: Note that the formula can still fail if Q1 = Q2 and P1 = P2
        #       but q1 \neq q2 and p1 \neq p2.
        pf = (self._f[(k,l)] * 2**(-(k+l)/2.0) * self._I0 *
              (1.0j*conjugate(P1)*Q2-1.0j*conjugate(Q1)*P2)**(-(k+l)/2.0))

        S = 0.0j
        for j in xrange(0, min(k,l)+1):
            S = S + (self._bk[k,j] * self._bl[l,j] * self._jf[j] *
                     self._pfk[k-j] * self._pfl[l-j] * self._Hk[k-j] * self._Hl[l-j])

        Ikl = pf * S
        return squeeze(Ikl)


    def _cache_factors(self, Pibra, Piket, Kbra, Kket, eps):
        r"""Cache some summands to speed up the computation of the sum.

        :param Pibra: The parameter set :math:`\Pi` of the bra :math:`\langle \Phi |`.
        :param Piket: The parameter set :math:`\Pi^\prime` of the ket :math:`| \Phi^\prime \rangle`.
        :param Kbra: The basis shape :math:`\mathfrak{K}` of the bra :math:`\langle \Phi |`.
        :type Kbra: A :py:class:`BasisShape` instance.
        :param Kket: The basis shape :math:`\mathfrak{K}^\prime` of the ket :math:`| \Phi^\prime \rangle`.
        :type Kket: A :py:class:`BasisShape` instance.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon`.
        """
        q1, p1, Q1, P1 = Pibra
        q2, p2, Q2, P2 = Piket

        # If both parameter sets are identical, we are back in the homogeneous case.
        if q1 == q2 and p1 == p2 and Q1 == Q2 and P1 == P2:
            self._Hk = None
            self._Hl = None

        # We have k in [0, 1, ..., K-1] where |K| is the basis size
        # hence K-1 is the maximal index.
        K = Kbra.get_basis_size()
        L = Kket.get_basis_size()

        mikl = min(K,L)
        makl = max(K,L)

        # Factorials
        f = factorial(arange(makl))
        self._f = 1.0 / sqrt(f[:K].reshape(-1,1) * f[:L].reshape(1,-1))

        # These prefactors depend only on j
        self._jf = f[:mikl] * 4**arange(mikl)

        # Binomials depend on k or l and j
        ik = arange(K).reshape(-1,1)
        il = arange(L).reshape(-1,1)
        ij = arange(mikl).reshape(1,-1)
        self._bk = binom(ik, ij)
        self._bl = binom(il, ij)

        # Note: formula currently fails for non-inhomogeneous case
        #       because of divisions by zero in the two args below.
        argk = ((1.0j*Q2*(p1-p2) - 1.0j*P2*(q1-q2)) /
                (sqrt(1.0j*Q2*P1 - 1.0j*Q1*P2) *
                 sqrt(1.0j*conjugate(P1)*Q2 - 1.0j*conjugate(Q1)*P2)))

        argl = ((1.0j*conjugate(P1)*(q1-q2) - 1.0j*conjugate(Q1)*(p1-p2)) /
                (sqrt(1.0j*conjugate(Q2*P1) - 1.0j*conjugate(Q1*P2)) *
                 sqrt(1.0j*conjugate(P1)*Q2 - 1.0j*conjugate(Q1)*P2)))

        # TODO: Better test for failure?
        if self._doraise and (isnan(squeeze(argk)) or isnan(squeeze(argl))):
            raise InnerProductException("Symbolic formula failed due to Q_k = Q_l and P_k = P_l.")

        # The parameter j varies in the range [0, 1, ..., min(K-1,L-1)]
        # hence we have that k-j can be in [K-1, K-2, ..., K-1-min(K-1,L-1)]
        # and similar for l-j we have [L-1, L-2, ..., L-1-min(K-1,L-1)]
        # where both K-1-min(K-1,L-1) and L-1-min(K-1,L-1) are non-negative.
        self._Hk = self._evaluate_hermite(K-1, -1.0/eps * argk)
        self._Hl = self._evaluate_hermite(L-1,  1.0/eps * argl)

        self._pfk = zeros((K,), dtype=complexfloating)
        self._pfl = zeros((L,), dtype=complexfloating)
        self._pfk[K-1-ij] = (1.0j*Q2*P1 - 1.0j*Q1*P2)**((K-1-ij)/2.0)
        self._pfl[L-1-ij] = (1.0j*conjugate(Q2*P1) - 1.0j*conjugate(Q1*P2))**((L-1-ij)/2.0)

        # And the groundstate value
        self._I0 = self.exact_result_ground(Pibra, Piket, eps)


    def perform_quadrature(self, row, col):
        r"""Evaluates the integral :math:`\langle \Phi_i | \Phi^\prime_j \rangle`
        by an exact symbolic formula.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A single complex floating point number.
        """
        eps = self._packet.get_eps()

        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        cbra = self._pacbra.get_coefficient_vector(component=row)
        cket = self._packet.get_coefficient_vector(component=col)
        Kbra = self._pacbra.get_basis_shapes(component=row)
        Kket = self._packet.get_basis_shapes(component=col)

        self._cache_factors(Pibra[:4], Piket[:4], Kbra, Kket, eps)

        result = 0.0j

        for r in Kbra:
            for c in Kket:
                cr = cbra[Kbra[r],0]
                cc = cket[Kket[c],0]
                i = self.exact_result_higher(Pibra[:4], Piket[:4], eps, r[0], c[0])
                result = result + conjugate(cr) * cc * i

        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))
        return phase * result


    def perform_build_matrix(self, row, col):
        r"""Computes the matrix elements :math:`\langle\Phi_i |\Phi^\prime_j\rangle`
        by an exact symbolic formula.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}^\prime_j|`.
        """
        eps = self._packet.get_eps()

        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        Kbra = self._pacbra.get_basis_shapes(component=row)
        Kket = self._packet.get_basis_shapes(component=col)

        self._cache_factors(Pibra[:4], Piket[:4], Kbra, Kket, eps)

        M = zeros((Kbra.get_basis_size(),Kket.get_basis_size()), dtype=complexfloating)

        for r in Kbra:
            for c in Kket:
                M[Kbra[r], Kket[c]] = self.exact_result_higher(Pibra[:4], Piket[:4], eps, r[0], c[0])

        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))
        return phase * M
