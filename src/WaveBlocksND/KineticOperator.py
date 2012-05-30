"""The WaveBlocks Project

This file contains code for the representation of the kinetic
operator :math:`-\frac{1}{2} \Delta` in the Fourier space.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import roll, exp, pi
from numpy.fft import fftfreq

__all__ = ["KineticOperator"]


class KineticOperator(object):
    """This class represents the kinetic operator :math:`T` in Fourier space.
    """

    def __init__(self, grid, eps=None):
        r"""Compute the Fourier transformation of the position space representation
        of the kinetic operator :math:`T`.

        :param grid: The position space grid :math:`\Gamma` of which we compute
                     its Fourier transform :math:`\Omega`.
        :param eps: The semi-classical scaling parameter :math:`\varepsilon`. (optional)
        """
        # Cache the value of eps if given
        self._eps = eps

        # TODO: Consider class "FourierSpaceGrid"
        D = grid.get_dimension()
        N = grid.get_number_nodes()
        T = grid.get_extensions()

        # Fourier domain nodes
        prefactors = [ t / (2.0*pi) for t in T ]
        omega = [ fftfreq(n, d=1.0/n) for n in N ]
        omega = [ o / p for o, p in zip(omega, prefactors) ]

        # Reshape properly
        shape = [-1] + (D-1)*[1]
        omega = [ o.reshape(roll(shape, i)) for i, o in enumerate(omega) ]

        # Compute the dot product of omega with itself
        omega_sqr = map(lambda x: x*x, omega)
        omega_sqr = reduce(lambda x,y: x+y, omega_sqr)

        # Fourier space grid axes
        self._omega = omega
        # Omega dot Omega
        self._omega_sqr = omega_sqr


    def get_fourier_grid_axes(self):
        r"""Return the grid axes of the Fourier space grid :math:`\Omega`.
        """
        return tuple([ o.copy() for o in self._omega ])


    def calculate_operator(self, eps=None):
        r"""Calculate the kinetic operator :math:`\hat{T} = \frac{\varepsilon^4}{2} \underline{\omega} \cdot \underline{\omega}`
        in Fourier space.

        :param eps: The semi-classical scaling parameter :math:`\varepsilon`. It has to
                    be given here or during the initialization of the current instance.
        """
        if eps is None:
            eps = self._eps
        if eps is None:
            # The value was not cached at initialization
            raise AttributeError("Parameter eps not given")

        self._OpT = 0.5 * eps**4 * self._omega_sqr


    def calculate_exponential(self, factor=1.0):
        r"""Calculate the exponential :math:`\exp(\alpha \underline{\omega} \cdot \underline{\omega})`
        used in the Strang splitting.

        :param factor: The prefactor :math:`\alpha`. It defaults to 1 but is
                       ususally set to :math:`-\frac{i}{2} \varepsilon^2 \tau`
                       by the caller.
        """
        self._expOpT = exp(factor * self._omega_sqr)


    def evaluate_at(self, grid=None):
        r"""Evaluate the kinetic operator :math:`\hat{T} = \frac{\varepsilon^4}{2} \underline{\omega} \cdot \underline{\omega}`
        in Fourier space. This returns an numpy ``ndarray`` by using a specific :math:`\underline{\omega}`.

        :param grid: Unused dummy parameter.
        """
        return self._OpT.copy()


    def evaluate_exponential_at(self, grid=None):
        r"""Evaluate the exponential :math:`\exp(\alpha \hat{T})` in Fourier space.
        This returns a numpy ``ndarray`` by using a specific :math:`\underline{\omega}`.
        The factor :math:`\alpha` can be set by the corresponding method.

        :param grid: Unused dummy parameter.
        """
        return self._expOpT.copy()
