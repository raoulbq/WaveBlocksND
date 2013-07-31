"""The WaveBlocks Project

This file contains code for constructing phase space lattices.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, sqrt, pi, mgrid, complexfloating, einsum, product, sum

__all__ = ["PhaseSpaceLattice"]


class PhaseSpaceLattice(object):
    r"""A phase space lattice centered around an energy :math:`E_0`
    and bounded by an energy delta :math:`\Delta E`.
    """

    def __init__(self, potential, energy, energydelta, eps, D):
        r"""
        Configure a new phase space lattice centered around an energy :math:`E_0`
        and bounded by an energy delta :math:`\Delta E`. The actual lattice points
        are computed by the member function :py:func:`compute_lattice`.

        :param potential: The potential :math:`V`.
        :type potential: Not a :py:class:`MatrixPotential` instance but possible a (partially evaluated)
                         member method like :py:func:`evaluate_at` or any scalar valued function.
        :param energy: The energy :math:`E_0` around which to center the lattice points.
        :param energydelta: The energy delta :math:`\Delta E`.
        :param eps: The semiclassical scaling parameter :math:`\varepsilon`.
        :param D: The dimensionality :math:`D` of position space.
        """
        self._potential = potential
        self._eps = eps
        self._energy = energy
        self._energydelta = energydelta
        self._dimension = D
        self._lattice_computed = False


    def compute_lattice(self, qlimits, plimits):
        r"""Compute the lattice points. Search for valid points an a hypercubic
        region of phase space given by limits for position :math:`q` and
        momentum :math:`p`.

        :param qlimits: The position limits :math:`q^d_{\mathrm{min}}` and :math:`q^d_{\mathrm{max}}`
                        for all dimensions :math:`d \in [0,\ldots,D-1]`.
        :type qlimits: A list of pairs of floats.
        :param plimits: The momentum limit :math:`p^d_{\mathrm{min}}` and :math:`p^d_{\mathrm{max}}`
                        for all dimensions :math:`d \in [0,\ldots,D-1]`.
        :type plimits: A list of pairs of floats.
        """
        dimension = self._dimension
        latdist = 0.75 * self._eps * sqrt(pi)

        qslicers = [ slice(lims[0], lims[1]+latdist, latdist) for lims in qlimits ]
        pslicers = [ slice(lims[0], lims[1]+latdist, latdist) for lims in plimits ]

        qgrid = array(mgrid[qslicers], dtype=complexfloating).reshape((dimension, -1))
        pgrid = array(mgrid[pslicers], dtype=complexfloating).reshape((dimension, -1))

        qvals = self._potential(qgrid)
        pvals = 0.5 * einsum("ij,ij->j", pgrid, pgrid).reshape(-1,1)

        Z = qvals + pvals
        indices = (abs(Z - self._energy) < self._energydelta)

        keepq = []
        keepp = []
        rows, cols = indices.shape
        for r in xrange(rows):
            for c in xrange(cols):
                if bool(indices[r,c]) is True:
                    keepq.append(c)
                    keepp.append(r)

        qgridf = qgrid[:,keepq]
        pgridf = pgrid[:,keepp]

        ps_size = sum(indices)
        ps_size_full = product(Z.shape)

        print("Phase space lattice size: "+str(ps_size))
        print(" number candidates tested: "+str(ps_size_full))
        print(" pruning factor: "+str((1.0 - ps_size / (1.0*ps_size_full)) * 100)+"%")

        self._qgrid = qgridf
        self._pgrid = pgridf
        self._lattice_size = ps_size
        self._lattice_computed = True


    def get_lattice(self):
        r"""
        :return: The phase space lattice in the form ``(qgrid, pgrid)``
                 where each grid is an :py:class:`ndarray` of shape ``(D, lattice_size)``
        """
        if self._lattice_computed:
            return (self._qgrid, self._pgrid)
        else:
            raise ValueError("Phase space lattice not yet computed")


    def get_lattice_size(self):
        r"""
        :return: The number of points :math:`(q_i,p_i)` in the lattice.
        """
        if self._lattice_computed:
            return self._lattice_size
        else:
            raise ValueError("Phase space lattice not yet computed")
