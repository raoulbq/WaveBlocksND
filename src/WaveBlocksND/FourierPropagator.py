r"""The WaveBlocks Project

This file contains the Fourier propagator class. The wavefunction
:math:`Psi` is propagated in time with a strang splitting of the
exponential :math:`\exp(-\frac{i}{\varepsilon^2} \tau H)`.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating
from numpy.fft import fftn, ifftn

from Propagator import Propagator
from KineticOperator import KineticOperator

__all__ = ["FourierPropagator"]


class FourierPropagator(Propagator):
    r"""This class can numerically propagate given initial values :math:`\Psi(x_0, t_0)` on
    a potential hypersurface :math:`V(x)`. The propagation is done with a Strang splitting
    of the time propagation operator :math:`\exp(-\frac{i}{\varepsilon^2} \tau H)`.
    """

    def __init__(self, potential, initial_values, para):
        r"""Initialize a new :py:class:`FourierPropagator` instance. Precalculate the
        the kinetic operator :math:`T_e` and the potential operator :math:`V_e`
        used or time propagation.

        :param potential: The potential :math:`V(x)` governing the time evolution.
        :type potential: A :py:class:`MatrixPotential` instance.
        :param initial_values: The initial values :math:`\Psi(\Gamma, t_0)` given
                               in the canonical basis.
        :type initial_values: A :py:class:`WaveFunction` instance.
        :param para: The set of simulation parameters. It must contain at least
                     the semi-classical parameter :math:`\varepsilon` and the
                     time step size :math:`tau`.

        :raise ValueError: If the number of components of :math:`\Psi` does not match the
                           number of energy surfaces :math:`\lambda_i(x)` of the potential.
        """
        # The embedded 'MatrixPotential' instance representing the potential 'V'.
        self._potential = potential

        # The initial values of the components '\psi_i' sampled at the given grid.
        self._psi = initial_values

        if self._potential.get_number_components() != self._psi.get_number_components():
            raise ValueError("Potential dimension and number of components do not match.")

        # The position space grid nodes '\Gamma'.
        self._grid = initial_values.get_grid()

        # The kinetic operator 'T' defined in momentum space.
        self._KO = KineticOperator(self._grid)
        self._KO.calculate_operator(para["dt"], para["eps"])

        # Exponential '\exp(-i/eps^2*dt*T)' used in the Strang splitting.
        self._TE = self._KO.evaluate_exponential_at()

        # Exponential '\exp(-i/eps^2*dt*V)' used in the Strang splitting.
        # TODO: Think about the interface 'calculate_exponential'
        self._potential.calculate_exponential(-0.5j * para["dt"] / para["eps"]**2)
        VE = self._potential.evaluate_exponential_at(self._grid)
        self._VE = tuple([ ve.reshape(self._grid.get_number_nodes()) for ve in VE ])


    # TODO: Consider removig this, duplicate
    def get_number_components(self):
        r"""Get the number :math:`N` of components of :math:`\Psi`.

        :return: The number :math:`N`.
        """
        return self._potential.get_number_components()


    def get_wavefunction(self):
        r"""Get the wavefunction that stores the current data :math:`\Psi(\Gamma)`.

        :return: The :py:class:`WaveFunction` instance.
        """
        return self._psi


    def get_operators(self):
        r"""Get the kinetic and potential operators :math:`T(\Omega)` and :math:`V(\Gamma)`.

        :return: A tuple :math:`(T, V)` containing two ``ndarrays``.
        """
        # TODO: What kind of object exactly do we want to return?
        T = self._KO.evaluate_at()
        V = self._potential.evaluate_at(self._grid)
        V = tuple([ v.reshape(self._grid.get_number_nodes()) for v in V ])
        return (T, V)


    def propagate(self):
        r"""Given the wavefunction values :math:`\Psi(\Gamma)` at time :math:`t`, calculate
        new values :math:`\Psi^\prime(\Gamma)` at time :math:`t + \tau`. We perform exactly
        one single timestep of size :math:`\tau` within this function.
        """
        # How many components does Psi have
        N = self._psi.get_number_components()

        # Unpack the values from the current WaveFunction
        values = self._psi.get_values()

        tmp = [ zeros(value.shape, dtype=complexfloating) for value in values ]

        # The first step with the potential
        for row in xrange(0, N):
            for col in xrange(0, N):
                tmp[row] = tmp[row] + self._VE[row*N+col] * values[col]

        # Go to Fourier space
        tmp = [ fftn(component) for component in tmp ]

        # Apply the kinetic operator
        tmp = [ self._TE * component for component in tmp ]

        # Go back to real space
        tmp = [ ifftn(component) for component in tmp ]

        # The second step with the potential
        values = [ zeros(component.shape, dtype=complexfloating) for component in tmp ]
        for row in xrange(0, N):
            for col in xrange(0, N):
                values[row] = values[row] + self._VE[row*N+col] * tmp[col]

        # Pack values back to WaveFunction object
        # TODO: Consider squeeze(.) of data before repacking
        self._psi.set_values(values)
