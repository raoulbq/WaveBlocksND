"""The WaveBlocks Project

Compute some observables like kinetic and potential energy
of Hagedorn wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
from numpy import conjugate, squeeze, sum

from Observables import Observables
from GradientHAWP import GradientHAWP

__all__ = ["ObservablesHAWP"]


class ObservablesHAWP(Observables):
    r"""This class implements observable computation for Hagedorn wavepackets :math:`\Psi`.
    """

    def __init__(self, innerproduct=None):
        r"""Initialize a new :py:class:`ObservablesHAWP` instance for observable computation
        of Hagedorn wavepackets.

        :param innerproduct: An inner product for computing the integrals. The inner product is only used for
                             the computation of the potential energy :math:`\langle\Psi|V(x)|\Psi\rangle`
                             but not for the kinetic energy.
        :type innerproduct: A :py:class:`InnerProduct` subclass instance.
        """
        # A innerproduct to compute the integrals
        if innerproduct is not None:
            self._innerproduct = innerproduct

        self._gradient = GradientHAWP()


    def set_innerproduct(self, innerproduct):
        r"""Set the innerproduct.

        :param innerproduct: An innerproduct for computing the integrals. The inner product is only used for
                             the computation of the potential energy :math:`\langle\Psi|V(x)|\Psi\rangle`
                             but not for the kinetic energy.
        :type innerproduct: A :py:class:`InnerProduct` subclass instance.
        """
        self._innerproduct = innerproduct


    def norm(self, wavepacket, component=None, summed=False):
        r"""Calculate the :math:`L^2` norm :math:`\langle\Psi|\Psi\rangle` of the wavepacket :math:`\Psi`.

        Note: this method is just a shortcut and calls the ``norm`` method
        of the given wavepacket.

        :param wavepacket: The wavepacket :math:`\Psi` of which we compute the norm.
        :type wavepacket: A :py:class:`HagedornWavepacketBase` subclass instance.
        :param component: The index :math:`i` of the component :math:`\Phi_i` whose norm is calculated.
                          The default value is ``None`` which means to compute the norms of all :math:`N` components.
        :type component: int or ``None``.
        :param summed: Whether to sum up the norms :math:`\langle\Phi_i|\Phi_i\rangle` of the
                       individual components :math:`\Phi_i`.
        :type summed: Boolean, default is ``False``.
        :type summed: Boolean, default is ``False``.
        :return: The norm of :math:`\Psi` or the norm of :math:`\Phi_i` or a list with the :math:`N`
                 norms of all components. Depending on the values of ``component`` and ``summed``.
        """
        return wavepacket.norm(component=component, summed=summed)


    def kinetic_energy(self, wavepacket, component=None, summed=False):
        r"""Compute the kinetic energy :math:`E_{\text{kin}} := \langle\Psi|T|\Psi\rangle`
        of the different components :math:`\Phi_i` of the wavepacket :math:`\Psi`.

        :param wavepacket: The wavepacket :math:`\Psi` of which we compute the kinetic energy.
        :type wavepacket: A :py:class:`HagedornWavepacketBase` subclass instance.
        :param component: The index :math:`i` of the component :math:`\Phi_i` whose
                          kinetic energy we want to compute. If set to ``None`` the
                          computation is performed for all :math:`N` components.
        :type component: Integer or ``None``.
        :param summed: Whether to sum up the kinetic energies :math:`E_i` of the individual
                       components :math:`\Phi_i`. Default is ``False``.
        :type summed: Boolean
        :return: A list with the kinetic energies of the individual components or the
                 overall kinetic energy of the wavepacket. (Depending on the optional arguments.)
        """
        if component is None:
            N = wavepacket.get_number_components()
            components = xrange(N)
        else:
            components = [component]

        ekin = []

        for n in components:
            Kprime, cnew = self._gradient.apply_gradient(wavepacket, n)
            ekin.append(0.5 * sum(sum(conjugate(cnew)*cnew, axis=1), axis=0))

        if summed is True:
            ekin = sum(ekin)
        elif component is not None:
            # Do not return a list for specific single components
            ekin = ekin[0]

        return ekin


    def potential_energy(self, wavepacket, potential, component=None, summed=False):
        r"""Compute the potential energy :math:`E_{\text{pot}} := \langle\Psi|V|\Psi\rangle`
        of the different components :math:`\Phi_i` of the wavepacket :math:`\Psi`.

        :param wavepacket: The wavepacket :math:`\Psi` of which we compute the potential energy.
        :type wavepacket: A :py:class:`HagedornWavepacketBase` subclass instance.
        :param potential: The potential :math:`V(x)`.
        :param component: The index :math:`i` of the component :math:`\Phi_i` whose
                          potential energy we want to compute. If set to ``None`` the
                          computation is performed for all :math:`N` components.
        :type component: Integer or ``None``.
        :param summed: Whether to sum up the potential energies :math:`E_i` of the individual
                       components :math:`\Phi_i`. Default is ``False``.
        :type summed: Boolean
        :return: A list with the potential energies of the individual components or the
                 overall potential energy of the wavepacket. (Depending on the optional arguments.)
        """
        N = wavepacket.get_number_components()

        # TODO: Better take 'V' instead of 'V.evaluate_at' as argument?
        #f = partial(potential.evaluate_at, as_matrix=True)
        f = partial(potential, as_matrix=True)

        # Compute the brakets for each component
        if component is not None:
            Q = self._innerproduct.quadrature(wavepacket, operator=f, diag_component=component, eval_at_once=True)
        else:
            Q = self._innerproduct.quadrature(wavepacket, operator=f, eval_at_once=True)

        # And don't forget the summation in the matrix multiplication of 'operator' and 'ket'
        # TODO: Should this go inside the innerproduct?
        tmp = map(squeeze, Q)
        epot = [ sum(tmp[i*N:(i+1)*N]) for i in xrange(N) ]

        if summed is True:
            epot = sum(epot)
        elif component is not None:
            # Do not return a list for specific single components
            epot = epot[0]

        return epot
