"""The WaveBlocks Project

This file contains the interface for observable computators.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

__all__ = ["Observables"]


class Observables(object):
    r"""This class is the interface definition for general observable
    computation procedures.
    """

    def __init__(self):
        r"""
        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Observables' is an abstract interface.")


    def kinetic_energy(self, ket, T):
        r"""Compute the kinetic energy :math:`E_{\text{kin}} := \langle\psi|T|\psi\rangle`.

        :param ket: The object denoted by :math:`\psi`.
        :param T: The kinetic energy operator :math:`T`.
        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Observables' is an abstract interface.")


    def potential_energy(self, ket, potential):
        r"""Compute the potential energy :math:`E_{\text{pot}} := \langle\psi|V|\psi\rangle`.

        :param ket: The object denoted by :math:`\psi`.
        :param potential: The potential :math:`V(x)`.
        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Observables' is an abstract interface.")
