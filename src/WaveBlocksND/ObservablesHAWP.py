"""The WaveBlocks Project

Compute some observables like kinetic and potential energy
of Hagedorn wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
from numpy import zeros, complexfloating, conjugate, dot, squeeze, sum
from scipy import sqrt

from Observables import Observables

__all__ = ["ObservablesHAWP"]


class ObservablesHAWP(Observables):
    r"""This class implements observable computation for Hagedorn wavepackets :math:`\Psi`.
    """

    def __init__(self, quadrature=None):
        r"""Initialize a new :py:class:`ObservablesHAWP` instance for observable computation
        of Hagedorn wavepackets.

        :param quadrature: A quadrature for computing the integrals. Quadrature is only used for
                           the computation of the potential energy :math:`\langle\Psi|V(x)|\Psi\rangle`
                           but not for the kinetic energy.
        :type quadrature: A :py:class:`Quadrature` subclass instance.
        """
        # A quadrature to compute the integrals
        if quadrature is not None:
            self._quadrature = quadrature


    def set_quadrature(self, quadrature):
        r"""Set the quadrature.

        :param quadrature: A quadrature for computing the integrals. Quadrature is only used for
                           the computation of the potential energy :math:`\langle\Psi|V(x)|\Psi\rangle`
                           but not for the kinetic energy.
        :type quadrature: A :py:class:`Quadrature` subclass instance.
        """
        self._quadrature = quadrature


    def apply_gradient(self, wavepacket, component):
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

        # We implement the less efficient gather type stencil here
        for i in Ke.get_node_iterator():
            # Central phi_i coefficient
            if i in K:
                ccur = coeffs[K[i]]
            else:
                ccur = 0.0j

            # Backward neighbours phi_{i - e_d}
            cbw = zeros((D,1), dtype=complexfloating)
            nbw = K.get_neighbours(i, selection="backward")

            for d, nb in nbw:
                cbw[d] = coeffs[K[nb]] * sqrt(i[d])

            # Forward neighbours phi_{i + e_d}
            cfw = zeros((D,1), dtype=complexfloating)
            nfw = K.get_neighbours(i, selection="forward")

            for d, nb in nfw:
                cfw[d] = coeffs[K[nb]] * sqrt(i[d] + 1.0)

            # Compute parts and assemble
            cnew[Ke[i],:] = squeeze(sqrt(eps**2/2.0) * (dot(Pbar, cfw) + dot(P, cbw)) + p * ccur)

        return (K, cnew)


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
                       components :math:`\Phi_i`. Default is `False`.
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
            Kprime, cnew = self.apply_gradient(wavepacket, n)
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
                       components :math:`\Phi_i`. Default is `False`.
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
            Q = self._quadrature.quadrature(wavepacket, operator=f, diag_component=component)
        else:
            Q = self._quadrature.quadrature(wavepacket, operator=f)

        # And don't forget the summation in the matrix multiplication of 'operator' and 'ket'
        # TODO: Should this go inside the quadrature?
        tmp = map(squeeze, Q)
        epot = [ sum(tmp[i*N:(i+1)*N]) for i in xrange(N) ]

        if summed is True:
            epot = sum(epot)
        elif component is not None:
            # Do not return a list for specific single components
            epot = epot[0]

        return epot
