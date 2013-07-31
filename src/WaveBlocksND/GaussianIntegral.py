"""The WaveBlocks Project

Use a symbolic exact formula for computing the inner product
between two Gaussian wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze, conjugate, ones, zeros, sqrt, dot, transpose, pi
from numpy.linalg import det, inv
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

        Note that the symbolic Gaussian integral can not handle operators at all.

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


    def exact_result(self, Pibra, Piket, eps, D):
        r"""Compute the overlap integral :math:`\langle \phi_0 | \phi_0 \rangle` of
        the groundstate :math:`\phi_0` by using the Gaussian integral formula:

        .. math::
            \int \exp\left(-\frac{1}{2} \underline{x}^{\textnormal{T}} \mathbf{U} \underline{x}
                           + \underline{v}^{\textnormal{T}} \underline{x}\right) \mathrm{d}x =
            \sqrt{\det\left(2 \pi \mathbf{U}^{-1}\right)}
            \exp\left(\frac{1}{2} \underline{v}^{\textnormal{T}} \mathbf{U}^{\textnormal{-T}} \underline{v}\right)

        Note that this is an internal method and usually there is no
        reason to call it from outside.

        :param Pibra: The parameter set :math:`\Pi_k = (\underline{q_k}, \underline{p_k}, \mathbf{Q_k}, \mathbf{P_k})`
                      of the bra :math:`\langle \phi_0 |`.
        :param Piket: The parameter set :math:`\Pi_l = (\underline{q_l}, \underline{p_l}, \mathbf{Q_l}, \mathbf{P_l})`
                      of the ket :math:`| \phi_0 \rangle`.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon`.
        :param D: The dimensionality of space.
        :return: The value of the integral :math:`\langle \phi_0 | \phi_0 \rangle`.
        """
        qr, pr, Qr, Pr = Pibra[:4]
        qc, pc, Qc, Pc = Piket[:4]
        Gr = dot(Pr, inv(Qr))
        Gc = dot(Pc, inv(Qc))
        # exp(conjugate(...) + ...) = exp( (i/eps^2) * (x^T A x  +  b^T x  +  c) )
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
                 + (dot(conjugate(transpose(qr)),pr) - dot(conjugate(transpose(pc)),qc))
            )
        # Include the common factor i/eps**2
        A = 1.0j/eps**2 * A
        b = 1.0j/eps**2 * b
        c = 1.0j/eps**2 * c
        # Rewrite
        U = -2.0 * A
        v = b
        # Global prefactor
        pf = (pi*eps**2)**(-D*0.25)
        return pf**2 * sqrt(det(2.0*pi*inv(U))) * exp(0.5*dot(transpose(v), dot(transpose(inv(U)), v))) * exp(c)


    def perform_quadrature(self, row, col):
        r"""Evaluates the integral :math:`\langle \Phi_i | \Phi^\prime_j \rangle`
        by an exact Gaussian integral formula.

        .. warning::
            Note that this method does not check if the wavepackets are pure Gaussians
            and in case they are not, the values returned will be wrong.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A single complex floating point number.
        """
        eps = self._packet.get_eps()
        D = self._packet.get_dimension()
        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        cbra = squeeze(self._pacbra.get_coefficient_vector(component=row))
        cket = squeeze(self._packet.get_coefficient_vector(component=col))
        result = conjugate(cbra) * cket * self.exact_result(Pibra[:4], Piket[:4], eps, D)
        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))
        return squeeze(phase * result)


    def perform_build_matrix(self, row, col):
        r"""Computes the matrix elements :math:`\langle\Phi_i |\Phi^\prime_j\rangle`
        by an exact Gaussian integral formula.

        .. warning::
            Note that this method does not check if the wavepackets are pure Gaussians
            and in case they are not, the values returned will be wrong.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`1 \times 1`.
        """
        eps = self._packet.get_eps()
        D = self._packet.get_dimension()
        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        result = self.exact_result(Pibra[:4], Piket[:4], eps, D)
        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))
        return phase * result
