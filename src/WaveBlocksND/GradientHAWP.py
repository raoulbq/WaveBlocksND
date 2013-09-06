"""The WaveBlocks Project

Compute the action of the gradient operator applied to a Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
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


    def apply_gradient(self, wavepacket, component=0, as_packet=False):
        r"""Compute the effect of the gradient operator :math:`-i \varepsilon^2 \nabla_x` on the basis
        functions :math:`\phi(x)` of a component :math:`\Phi_i` of the Hagedorn wavepacket :math:`\Psi`.

        :param wavepacket: The wavepacket :math:`\Psi` containing :math:`\Phi_i`.
        :type wavepacket: A :py:class:`HagedornWavepacketBase` subclass instance.
        :param component: The index :math:`i` of the component :math:`\Phi_i`.
        :type component: Integer.
        :param as_packet: Whether to return a full packet.
        :type as_packet: Boolean, default is ``False``.
        :return: Extended basis shape :math:`\mathfrak{\dot{K}}` and new coefficients :math:`c^\prime`.
                 If requested, copies of the original wavepacket are returned with these new values set.
                 There is one packet for each component of the gradient.
        """
        # TODO: Consider moving this method into the HAWP class?
        D = wavepacket.get_dimension()
        eps = wavepacket.get_eps()
        q, p, Q, P, S = wavepacket.get_parameters(component=component)
        Pbar = conjugate(P)
        coeffs = wavepacket.get_coefficients(component=component)

        # Prepare storage for new coefficients
        K = wavepacket.get_basis_shapes(component=component)
        Ke = K.extend()
        size = Ke.get_basis_size()
        cnew = zeros((size,D), dtype=complexfloating)

        # We implement the more efficient scatter type stencil here
        for k in K.get_node_iterator():
            # Central phi_i coefficient
            cnew[Ke[k],:] += squeeze(coeffs[K[k]] * p)

            # Backward neighbours phi_{i - e_d}
            nbw = Ke.get_neighbours(k, selection="backward")

            for d, nb in nbw:
                cnew[Ke[nb],:] += sqrt(eps**2/2.0) * sqrt(k[d]) * coeffs[K[k]] * Pbar[:,d]

            # Forward neighbours phi_{i + e_d}
            nfw = Ke.get_neighbours(k, selection="forward")

            for d, nb in nfw:
                cnew[Ke[nb],:] += sqrt(eps**2/2.0) * sqrt(k[d]+1.0) * coeffs[K[k]] * P[:,d]

        if as_packet is True:
            new_wps = []
            for d in xrange(D):
                hawp_new = wavepacket.clone()
                hawp_new.set_basis_shapes(Ke, component=component)
                hawp_new.set_coefficients(cnew[:,d], component=component)
                new_wps.append(hawp_new)
            return new_wps
        else:
            return (Ke, cnew)
