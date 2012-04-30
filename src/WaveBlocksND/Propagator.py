"""The WaveBlocks Project

This file contains the abstract base class for general time propagators.
It defines the interface every subclass must support to implement a
time propagation algorithm.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

class Propagator(object):
    r"""Propagators can numerically simulate the time evolution of quantum states
    :math:`\psi(x,t)` as described by the time-dependent Schroedinger equation

    .. math:: i \varepsilon^2 \frac{\partial}{\partial t} \psi(x,t) = H \psi(x,t)

    where the semi-classical scaling parameter :math:`\varepsilon > 0` is already included.
    The Hamiltonian operator :math:`H` is defined as

    .. math:: H = T + V(x) = -\frac{\varepsilon^4}{2} \Delta + V(x)
    """

    def __init__(self):
        r"""Initialize a new :py:class:`Propagator` instance.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Propagator' is an abstract base class.")


    def __str__(self):
        r"""Prepare a printable string representing the :py:class:`Propagator` instance.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Propagator' is an abstract base class.")


    def get_number_components(self):
        r""":return: The number :math:`N` components of :math:`\psi(x,t)`.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("get_number_components(...)")


    def get_potential(self):
        r"""Returns the potential :math:`V(x)` used for time propagation.

        :return: A :py:class:`MatrixPotential` subclass instance.
        """
        return self._potential


    def propagate(self):
        r"""Given the wavefunction :math:`\psi` at time :math:`t`, calculate
        the new :math:\psi` at time :math:`t + \tau`. We do exactly one timestep
        of size :math:`\tau` here.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("propagate(...)")
