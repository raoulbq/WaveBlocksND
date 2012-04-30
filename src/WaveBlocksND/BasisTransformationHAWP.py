r"""The WaveBlocks Project

This file contains the class for basis transformations of Hagedorn wavepackets
between the canonical basis and the basis spanned by the eigenvectors
of the potential.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import dot, transpose, conjugate, vsplit

from BasisTransformation import BasisTransformation


__all__ = ["BasisTransformationHAWP"]


class BasisTransformationHAWP(BasisTransformation):
    r"""This class implements basis transformations of Hagedorn wavepackets
    :math:`\Psi(x)` between the canonical basis of and the basis :math:`\Lambda(x)`
    spanned by the eigenvectors :math:`\nu_i(x)` of the potential :math:`V(x)`.
    """

    def __init__(self, potential, builder=None):
        r"""Create a new :py:class:`BasisTransformationHAWP` instance for a
        given potential matrix :math:`V(x)`.

        :param potential: The potential underlying the basis transformation.
        :type potential: A :py:class:`MatrixPotential` instance.
        :param builder: An object that can compute this matrix.
        :type builder: A :py:class:`Quadrature` subclass instance.
        """
        # Keep a reference to the potential
        self._potential = potential

        # Precompute eigenvectors is case it is necessary
        self._potential.calculate_eigenvectors()

        if builder is not None:
            self.set_matrix_builder(builder)


    def set_matrix_builder(self, builder):
        r"""Set the matrix builder. It is responsible for computing the matrix
        elements :math:`\langle\phi_i|V_{i,j}|\phi_j\rangle`. This matrix
        is used during the basis transformation.

        :param builder: An object that can compute this matrix.
        :type builder: A :py:class:`Quadrature` subclass instance.
        """
        # Keep a reference to the matrix builder
        self._builder = builder


    def transform_to_canonical(self, wavepacket):
        r"""Transform the wavepacket :math:`\Psi` given
        in the eigenbasis to the canonical basis.

        Note that this method acts destructively on the given
        :py:class:`Wavepacket` instance. If this is not desired,
        clone the packet before handing it over to this method.

        :param wavepacket: The Hagedorn wavepacket to transform.
        :type wavepacket: A :py:class:`Wavepacket` subclass instance.
        :return: Another :py:class:`Wavepacket` instance containing the
                 transformed wavepacket :math:`\Psi^\prime`.
        """
        # No transformation for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if self._potential.get_number_components() == 1:
            return

        # Basically an ugly hack to overcome some shortcomings of the
        # matrix function and of the data layout.
        # TODO: Fix and remove
        def f(x, dummy):
            # x is given as (D,|QR|) array
            z = self._potential.evaluate_eigenvectors_at(x)
            # returned is a N list of (N,|QR|) arrays
            # we need a N**2 list of (|QR|,) arrays
            N = wavepacket.get_number_components()
            result = []
            for nu in z:
                result.extend(vsplit(nu, N))

            return result

        # And now compute the transformation
        F = transpose(conjugate(self._builder.build_matrix(wavepacket, f)))
        c = wavepacket.get_coefficient_vector()
        d = dot(F, c)

        wavepacket.set_coefficient_vector(d)
        return wavepacket


    def transform_to_eigen(self, wavepacket):
        r"""Transform the wavepacket :math:`\Psi^\prime` given
        in the canonical basis to the eigenbasis.

        Note that this method acts destructively on the given
        :py:class:`Wavepacket` instance. If this is not desired,
        clone the packet before handing it over to this method.

        :param wavepacket: The Hagedorn wavepacket to transform.
        :type wavepacket: A :py:class:`Wavepacket` subclass instance.
        :return: Another :py:class:`Wavepacket` instance containing the
                 transformed wavepacket :math:`\Psi`.
        """
        # No transformation for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if self._potential.get_number_components() == 1:
            return

        # Basically an ugly hack to overcome some shortcomings of the
        # matrix function and of the data layout.
        # TODO: Fix and remove
        def f(x, dummy):
            # x is given as (D,|QR|) array
            z = self._potential.evaluate_eigenvectors_at(x)
            # returned is a N list of (N,|QR|) arrays
            # we need a N**2 list of (|QR|,) arrays
            N = wavepacket.get_number_components()
            result = []
            for nu in z:
                result.extend(vsplit(nu, N))

            return result

        # And now compute the transformation
        F = self._builder.build_matrix(wavepacket, f)
        c = wavepacket.get_coefficient_vector()
        d = dot(F, c)

        wavepacket.set_coefficient_vector(d)
        return wavepacket
