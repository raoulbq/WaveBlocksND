"""The WaveBlocks Project

Use a symbolic exact formula for computing the inner product
between two semi-classical wavepackets. The formula is built
for Gaussian integrals and takes into account only the ground
states phi_0 of the 'bra' and the 'ket'.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze, conjugate, sqrt, ones, zeros, complexfloating, pi, dot, transpose
from scipy.linalg import inv, det
from scipy import exp

from Quadrature import Quadrature
from InnerProductCompatibility import InnerProductCompatibility

__all__ = ["GaussianIntegral"]


class GaussianIntegral(Quadrature, InnerProductCompatibility):
    r"""
    """

    def __init__(self, *unused, **kunused):
        r"""
        """
        # Drop any argument, we do not need a qr instance.


    def __str__(self):
        return "Inhomogeneous inner product computed using a Gaussian integral formula."


    def get_description(self):
        r"""Return a description of this integral object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "GaussianIntegral"
        return d


    def get_kind(self):
        return ("homogeneous", "inhomogeneous")


    def initialize_packet(self, pacbra, packet=None):
        r"""Provide the wavepacket parts of the inner product to evaluate.
        Since the formula is for the inhomogeneous case explicitly, different
        wavepackets can be used for the 'bra' as well as the 'ket' part.

        :param pacbra: The packet that is used for the 'bra' part.
        :param packet: The packet that is used for the 'ket' part.
        """
        # Allow to ommit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        self._pacbra = pacbra
        self._packet = packet


    def initialize_operator(self, operator=None, matrix=False, eval_at_once=False):
        r"""Provide the operator part of the inner product to evaluate.
        This function initializes the operator used for quadratures
        and for building matrices.

        .. note:: The symbolic Gaussian integral formula can not handle
                  operators at all.

        :param operator: The operator of the inner product.
                         If ``None`` a suitable identity is used.
        :param matrix: Set this to ``True`` (Default is ``False``) in case
                       we want to compute the matrix elements.
                       For nasty technical reasons we can not yet unify
                       the operator call syntax.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
                             Since we do not support operators at all, it has no effect.
        :type eval_at_once: Boolean, default is ``False``.
        """
        # Operator is None is interpreted as identity transformation
        if operator is None:
            self._operator = lambda nodes, dummy, entry=None: ones((1,nodes.shape[1])) if entry[0] == entry[1] else zeros((1,nodes.shape[1]))
        else:
            raise ValueError("The 'GaussianIntegral' can not handle operators.")


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


    def exact_result_gauss(self, Pibra, Piket, D, eps):
        r"""Compute the overlap integral :math:`\langle \phi_{\underline{0}} | \phi_{\underline{0}} \rangle`
        of the groundstate :math:`\phi_{\underline{0}}` by using the symbolic formula:

        .. math::
            \langle \phi_{\underline{0}} | \phi_{\underline{0}} \rangle
            & = \int C \exp\left(-\frac{1}{2} \underline{x}^{\mathrm{T}} \mathbf{A} \underline{x}
                           +\underline{b}^{\mathrm{T}} \underline{x}
                           + c
                     \right) \mathrm{d}\underline{x} \\
            & = C \sqrt{\frac{\left(2\pi\right)^D}{\det \mathbf{A}}}
              \exp\left(-\frac{1}{2} \underline{b}^{\mathrm{T}} \mathbf{A}^{-\mathrm{H}} \underline{b}\right)
              \exp\left(c\right)

        In a first step we combine the exponential parts of both wavepackets into
        :math:`\underline{x}^{\mathrm{H}} \mathbf{A} \underline{x} + \underline{b}^{\mathrm{T}} \underline{x} + c`.
        Then we transform :math:`\mathbf{A}`, :math:`\underline{b}` and :math:`c`
        such that this matches the integrand above. The necessary transformations read:

        .. math::
            \mathbf{A}^{\prime} &= -2 \frac{i}{\varepsilon^2} \mathbf{A} \\
            \underline{b}^{\prime} &= \frac{i}{\varepsilon^2} \underline{b} \\
            c &= \frac{i}{\varepsilon^2} c

        Note that this is an internal method and usually there is no
        reason to call it from outside.

        :param Pibra: The parameter set :math:`\Pi = \{q_1,p_1,Q_1,P_1\}` of the bra :math:`\langle \phi_0 |`.
        :param Piket: The parameter set :math:`\Pi^\prime = \{q_2,p_2,Q_2,P_2\}` of the ket :math:`| \phi_0 \rangle`.
        :param D: The space dimension :math:`D` the packets have.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon`.
        :return: The value of the integral :math:`\langle \phi_{\underline{0}} | \phi_{\underline{0}} \rangle`.
        """
        qr, pr, Qr, Pr = Pibra
        qc, pc, Qc, Pc = Piket
        hbar = eps**2

        Gr = dot(Pr, inv(Qr))
        Gc = dot(Pc, inv(Qc))

        # Merge exponential parts
        A = 0.5 * (Gc - conjugate(transpose(Gr)))
        b = (0.5 * (  dot(Gr, qr)
                    - dot(conjugate(transpose(Gc)), qc)
                    + dot(transpose(Gr), conjugate(qr))
                    - dot(conjugate(Gc), conjugate(qc))
                   )
             + (pc - conjugate(pr))
            )
        b = conjugate(b)
        c = (0.5 * (  dot(conjugate(transpose(qc)), dot(Gc, qc))
                    - dot(conjugate(transpose(qr)), dot(conjugate(transpose(Gr)), qr)))
                 + (dot(conjugate(transpose(qr)), pr) - dot(conjugate(transpose(pc)), qc))
            )

        A = 1.0j/hbar * A
        b = 1.0j/hbar * b
        c = 1.0j/hbar * c

        A = -2.0 * A

        # Gaussian formula
        I = sqrt(det(2.0*pi*inv(A))) * exp(0.5*dot(transpose(b),dot(conjugate(inv(A)),b))) * exp(c)

        # Prefactors
        pfbra = (pi*eps**2)**(-D/4.0) * 1.0/sqrt(det(Qr))
        pfket = (pi*eps**2)**(-D/4.0) * 1.0/sqrt(det(Qc))

        return conjugate(pfbra)*pfket * I


    def perform_quadrature(self, row, col):
        r"""Evaluates the integral :math:`\langle \Phi_i | \Phi^\prime_j \rangle`
        by an exact symbolic formula.

        .. warning:: This method does only take into account the ground state
                     basis components :math:`\phi_{\underline{0}}` from both,
                     the 'bra' and the 'ket'. If the wavepacket :math:`\Phi`
                     contains higher order basis functions :math:`\phi_{\underline{k}}`
                     with non-zero coefficients :math:`c_{\underline{k}}`, the inner products
                     computed are wrong! There is also no warning about that.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A single complex floating point number.
        """
        eps = self._packet.get_eps()
        D = self._packet.get_dimension()

        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        cbra = self._pacbra.get_coefficient_vector(component=row)
        cket = self._packet.get_coefficient_vector(component=col)
        Kbra = self._pacbra.get_basis_shapes(component=row)
        Kket = self._packet.get_basis_shapes(component=col)

        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))

        z = tuple(D*[0])
        cr = cbra[Kbra[z],0]
        cc = cket[Kket[z],0]
        i = self.exact_result_gauss(Pibra[:4], Piket[:4], D, eps)
        result = phase * conjugate(cr) * cc * i

        return result


    def perform_build_matrix(self, row, col):
        r"""Computes the matrix elements :math:`\langle\Phi_i |\Phi^\prime_j\rangle`
        by an exact symbolic formula.

        .. warning:: This method does only take into account the ground state
                     basis components :math:`\phi_{\underline{0}}` from both,
                     the 'bra' and the 'ket'. If the wavepacket :math:`\Phi`
                     contains higher order basis functions :math:`\phi_{\underline{k}}`
                     with non-zero coefficients :math:`c_{\underline{k}}`, the inner products
                     computed are wrong! There is also no warning about that.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}^\prime_j|`.
        """
        eps = self._packet.get_eps()
        D = self._packet.get_dimension()

        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        Kbra = self._pacbra.get_basis_shapes(component=row)
        Kket = self._packet.get_basis_shapes(component=col)

        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))

        z = tuple(D*[0])
        M = zeros((Kbra.get_basis_size(),Kket.get_basis_size()), dtype=complexfloating)
        M[Kbra[z], Kket[z]] = squeeze(phase * self.exact_result_gauss(Pibra[:4], Piket[:4], D, eps))

        return M
