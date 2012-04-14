# TODO: Adapt to WBND

"""The WaveBlocks Project

This file contains the abstract base class for general time propagators.
It defines the interface every subclass must support to implement a
time propagation algorithm.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

class Propagator(object):
    """Propagators can numerically simulate the time evolution of quantum states
    as described by the time-dependent Schroedinger equation.
    """

    def __init__(self):
        """Initialize a new I{Propagator} instance.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Propagator' is an abstract base class.")


    def __str__(self):
        """Prepare a printable string representing the I{Propagator} instance.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Propagator' is an abstract base class.")


    def get_number_components(self):
        """@return: The number of components of $\Ket{\Psi}$.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("get_number_components(...)")


    def get_potential(self):
        """@return: The embedded I{MatrixPotential} instance.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("get_potential(...)")


    def propagate(self):
        """Given the wavefunction $\Psi$ at time $t$, calculate the new $\Psi$
        at time $t + \tau$. We do exactly one timestep $\tau$ here.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("propagate(...)")
