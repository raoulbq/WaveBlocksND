"""The WaveBlocks Project

Compute the action of the gradient operator applied to a
linear combination of Hagedorn wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze

from Gradient import Gradient
from GradientHAWP import GradientHAWP
from LinearCombinationOfHAWPs import LinearCombinationOfHAWPs

__all__ = ["GradientLinearCombinationHAWP"]


class GradientLinearCombinationHAWP(Gradient):
    r"""This class implements the computation of the action of the
    gradient operator :math:`-i \varepsilon^2 \nabla_x` applied to
    a linear combination :math:`\Upsilon` of Hagedorn wavepackets :math:`\Psi`.
    """

    def __init__(self):
        pass


    # TODO: Find a more efficient way to compute gradients

    def apply_gradient(self, lincomb):
        r"""
        """
        D = lincomb.get_dimension()
        N = lincomb.get_number_components()
        J = lincomb.get_number_packets()
        Cj = squeeze(lincomb.get_coefficients())
        G = GradientHAWP()

        new_lincombs = [ LinearCombinationOfHAWPs(D, N) for d in xrange(D) ]

        # Handle each wavepacket individually
        for j in xrange(J):
            packet = lincomb.get_wavepacket(j)
            grads = G.apply_gradient(packet, as_packet=True)

            for d, grad in enumerate(grads):
                new_lincombs[d].add_wavepacket(grad, Cj[j])

        return new_lincombs
