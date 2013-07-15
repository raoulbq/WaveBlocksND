"""The WaveBlocks Project

Use a symbolic exact formula for computing the inner product
between two Gaussian wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze, conjugate, sqrt, ones, zeros, complexfloating, arange
from scipy import exp
from scipy.misc import factorial
from scipy.special import binom
#from scipy.special.orthogonal import eval_hermite

from Quadrature import Quadrature

__all__ = ["GaussianIntegral"]


class GaussianIntegral(Quadrature):
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


    def initialize_packet(self, pacbra, packet=None):
        r"""Provide the wavepacket parts of the inner product to evaluate.

        :param pacbra: The packet that is used for the 'bra' part.
        :param packet: The packet that is used for the 'ket' part.
        :raises: :py:class:`ValueError` if the dimension of :math:`\Psi` is not 1.
        """
        # Allow to ommit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        if not pacbra.get_dimension() == 1:
            raise ValueError("The 'GaussianIntegral' applies in the 1D case only.")

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


    def exact_result(self, Pibra, Piket, eps):
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


    def perform_quadrature(self, row, col):
        r"""Evaluates the integral :math:`\langle \Phi_i | \Phi^\prime_j \rangle`
        by an exact Gaussian integral formula.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A single complex floating point number.
        """
        eps = self._packet.get_eps()
        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        cbra = squeeze(self._pacbra.get_coefficient_vector(component=row))
        cket = squeeze(self._packet.get_coefficient_vector(component=col))
        result = conjugate(cbra) * cket * self.exact_result(Pibra[:4], Piket[:4], eps)
        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))
        return phase * result


    def perform_build_matrix(self, row, col):
        r"""Computes the matrix elements :math:`\langle\Phi_i |\Phi^\prime_j\rangle`
        by an exact Gaussian integral formula.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}^\prime_j|`.
        """
        eps = self._packet.get_eps()
        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        result = conjugate(cbra) * cket * self.exact_result(Pibra[:4], Piket[:4], eps)
        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))
        return phase * result
