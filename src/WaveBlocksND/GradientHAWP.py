"""The WaveBlocks Project

Compute the action of the gradient operator applied to a Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, conjugate, squeeze
from scipy import sqrt

from Gradient import Gradient

__all__ = ["GradientHAWP"]


class GradientHAWP(Gradient):
    r"""This class implements the computation of the action of the
    gradient operator :math:`-i \varepsilon^2 \nabla_x` applied to
    a Hagedorn wavepacket :math:`\Psi`.
    """

    def __init__(self):
        pass


    def apply_gradient(self, wavepacket, component=0):
        r"""Compute the effect of the gradient operator :math:`-i \varepsilon^2 \nabla_x` on the basis
        functions :math:`\phi(x)` of a component :math:`\Phi_i` of the Hagedorn wavepacket :math:`\Psi`.

        :param wavepacket: The wavepacket :math:`\Psi` containing :math:`\Phi_i`.
        :type wavepacket: A :py:class:`HagedornWavepacketBase` subclass instance.
        :param component: The index :math:`i` of the component :math:`\Phi_i`.
        :type component: int
        :return: Extended basis shape :math:`\mathcal{\dot{K}}` and new coefficients :math:`c^\prime`.
        """
        # TODO: Consider moving this method into the HAWP class?
        D = wavepacket.get_dimension()
        eps = wavepacket.get_eps()
        q, p, Q, P, S = wavepacket.get_parameters(component=component)
        Pbar = conjugate(P)
        coeffs = wavepacket.get_coefficients(component=component)

        # Prepare storage for new coefficients
        K = wavepacket.get_basis_shape(component=component)
        Ke = K.extend()
        size = Ke.get_basis_size()
        cnew = zeros((size,D), dtype=complexfloating)

        # We implement the more efficient scatter type stencil here
        for k in K.get_node_iterator():
            # Central phi_i coefficient
            cnew[Ke[k],:] += squeeze(coeffs[K[k]] * p)

            # Backward neighbours phi_{i - e_d}
            nbw = K.get_neighbours(k, selection="backward")

            for d, nb in nbw:
                cnew[Ke[nb],:] += sqrt(eps**2/2.0) * sqrt(k[d]) * coeffs[K[k]] * Pbar[:,d]

            # Forward neighbours phi_{i + e_d}
            nfw = K.get_neighbours(k, selection="forward")

            for d, nb in nfw:
                cnew[Ke[nb],:] += sqrt(eps**2/2.0) * sqrt(k[d]+1.0) * coeffs[K[k]] * P[:,d]

        return (Ke, cnew)
