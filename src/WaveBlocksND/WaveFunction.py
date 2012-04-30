r"""The WaveBlocks Project

This file contains code to numerically represent multiple
components of vector valued wave functions together with
the grid nodes the values belong to. In addition there
are some methods for calculating obervables as for example
:math:`L^2` norms and kinetic and potential energies.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, atleast_1d, product, array, conjugate, sum
from numpy.fft import fftn
from scipy import sqrt
from scipy import linalg as la

__all__ = ["WaveFunction"]


class WaveFunction(object):
    r"""This class represents a vector valued wavefuncton :math:`\Psi`
    as used in the vector valued time-dependent Schroedinger equation. The
    state :math:`\Psi` consists of the components :math:`\psi_0` to
    :math:`\psi_{N-1}`.
    """

    def __init__(self, parameters):
        r"""Initialize the :py:class:`WaveFunction` instance that represents
        the vector :math:`\Psi` of components :math:`\psi_i`.

        :param parameters: A :py:class:`ParameterProvider` instance having at
                           least the number :math:`N` of components.
        """
        self._number_components = parameters["ncomponents"]

        self._grid = None
        self._values = []


    def get_number_components(self):
        r"""The number of components :math:`\psi_i` the vector :math:`\Psi` consists of.
        """
        return self._number_components


    def get_grid(self):
        r"""Return the :py:class:`Grid` instance representing the grid :math:`\Gamma`.
        The wavefunction :math:`\Psi` is evaluated on the grid nodes to get :math:`\Psi(\Gamma)`.
        """
        return self._grid


    def set_grid(self, grid):
        r"""Assign a new grid :math:`\Gamma` to this :py:class:`WaveFunction` instance.

        Note: The user of this class has to make sure that the grid :math:`\Gamma` and
        the wavefunction values :math:`\Psi(\Gamma)` are consistent with each other!

        :param grid: A new :py:class:`Grid` instance.
        """
        self._grid = grid


    def get_values(self, components=None):
        r"""Get the wavefunction values :math:`\psi(\Gamma)` for each
        component :math:`\psi_i` of :math:`\Psi`.

        :param components: The components :math:`i` for which we want to get
                           the wavefunction values :math:`\psi_i`.
        :type components: A single integer or a list of integers. If set to
                          `None` (default) we return the data for all components.
        :return: A list of the values :math:`\psi_i` for all components :math:`i`.
        """
        if components is None:
            components = xrange(self._number_components)

        return [ self._values[c] for c in atleast_1d(components) ]


    def set_values(self, values, components=None):
        r"""Assign new wavefunction values :math:`\psi_i(\Gamma)` for each component
        :math:`i` of :math:`\Psi` to the current :py:class:`WaveFunction` instance.

        :param values: A list with the new values of all components we want to change.
        :type values: Each entry of the list has to be an ndarray.
        :param components: The components :math:`i` for which we want to set
                           the new wavefunction values :math:`\psi_i`.
        :type components: A single integer or a list of integers. If set to
                          `None` (default) we set the data for all components.

        Note: This method does NOT copy the input data arrays.

        :raise ValueError: If the list of `values` has the wrong length.
        """
        if components is None:
            assert len(values) == self._number_components
            # Set all components
            self._values = values[:]
        else:
            # Set only specified components
            for component, value in zip(atleast_1d(components), values):
                self._values[component] = value

    # TODO: Decide whether to move the following methods into an own class

    def norm(self, values=None, summed=False, components=None):
        r"""Calculate the :math:`L^2` norm of the whole vector values wavefunction
        :math:`\Psi` or some individual components :math:`\psi_i`. The calculation
        is done in momentum space.

        :param values: Allows to use this function for external data, similar to a
                       static function.
        :param summed: Whether to sum up the norms of the individual components.
        :type summed: Boolean, default is `False`.
        :param components: The components :math:`\psi_i` of which the norms are calculated.
        :type components: A single integer or a list of integers. If set to
                          `None` (default) we compute the norm for all components.
        :return: The :math:`L^2` norm of :math:`\Psi` or a list of :math:`L^2`
                 norms of the specified components :math:`\psi_i`.
        """
        # TODO: Assumption: values[i].shape is (N_1, ..., N_D) and not (D, product(N_i))
        if values is None:
            values = self._values

        if components is None:
            components = range(self._number_components)

        # Compute the prefactor
        T = self._grid.get_extensions()
        N = self._grid.get_number_nodes()
        prefactor = product( sqrt(array(T)) / (1.0*array(N)) )

        # Compute the norm for all components specified
        # TODO: Consider splitting into cases `fft` versus `fftn`
        norms = prefactor *array([ la.norm(fftn(self._values[component])) for component in atleast_1d(components) ])

        # Sum the individual norms if requested
        if summed is True:
            norms = map(lambda x: x**2, norms)
            return sqrt(sum(norms))
        else:
            return norms


    def kinetic_energy(self, kinetic, summed=False):
        """Calculate the kinetic energy :math:`E_{\text{kin}} := \Braket{\Psi|T|\Psi}`
        of the different components :math:`\psi_i`.

        :param kinetic: The kinetic energy operator :math:`T(\omega)`.
        :type kinetic: A :py:class:`KineticOperator` instance.
        :param summed: Whether to sum up the kinetic energies :math:`E_i` of the individual
                       components :math:`\psi_i`. Default is `False`.
        :return: A list with the kinetic energies of the individual components
        or the overall kinetic energy of the wavefunction. (Depending on the optional arguments.)
        """
        # TODO: Consider using less declarative coding style.
        #       Issue: Compute fft of each component only once
        #              AND avoid storing fft of all components.

        # Fourier transform the components
        ftc = [ fftn(component) for component in self._values ]

        # Compute the prefactor
        T = self._grid.get_extensions()
        N = self._grid.get_number_nodes()
        prefactor = product( array(T) / (1.0*array(N)**2) )

        # TODO: Consider taking the result of this call as input for efficiency?
        KO = kinetic.evaluate_at()

        # Compute the braket in Fourier space
        ekin = [ prefactor * sum(conjugate(item) * KO * item) for item in ftc ]

        if summed is True:
            ekin = sum(ekin)

        return ekin


    def potential_energy(self, potential, summed=False):
        """Calculate the potential energy :math:`E_{\text{pot}} := \Braket{\Psi|V|\Psi}`
        of the different components :math:`\psi_i`.

        :param potential: The potential energy operator :math:`V(x)`.
        :param summed: Whether to sum up the potential energies :math:`E_i` of the individual
                       components :math:`\psi_i`. Default is `False`.
        :return: A list with the potential energies of the individual components
        or the overall potential energy of the wavefunction. (Depending on the optional arguments.)
        """
        # Compute the prefactor
        T = self._grid.get_extensions()
        N = self._grid.get_number_nodes()
        prefactor = product( array(T) / (1.0*array(N)**2) )

        # Reshape from (1, prod_d^D N_d) to (N_1, ..., N_D) shape
        potential = [ pot.reshape(N) for pot in potential ]

        # Apply the matrix potential to the ket
        tmp = [ zeros(component.shape, dtype=complexfloating) for component in self._values ]
        for row in xrange(0, self._number_components):
            for col in xrange(0, self._number_components):
                tmp[row] = tmp[row] + potential[row*self._number_components+col] * self._values[col]

        # Fourier transform the components
        ftcbra = [ fftn(component) for component in self._values ]
        ftcket = [ fftn(component) for component in tmp ]

        # Compute the braket in Fourier space
        epot = [ prefactor * sum(conjugate(cbra) * cket) for cbra, cket in zip(ftcbra, ftcket) ]

        if summed is True:
            epot = sum(epot)

        return epot
