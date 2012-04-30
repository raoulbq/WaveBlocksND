"""The WaveBlocks Project

This file contains several different algorithms to compute the
matrix exponential. Currently we have an exponential based on
Pade approximations and an Arnoldi iteration method.

@author: R. Bourquin
@copyright: Copyright (C) 2007 V. Gradinaru
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, hstack, mat, dot, complexfloating, asarray
from scipy.linalg import norm, expm


def matrix_exp_pade(A, v, factor):
    r"""Compute the solution of :math:`v' = A v` with a full
    matrix exponential via Pade approximation.

    :param A: The matrix :math:`A` of shape :math:`N \times N`.
    :param v: The vector :math:`v` of length :math:`N`.
    :param factor: An additional scalar factor :math:`\alpha`.
    :return: The (approximate) value of :math:`\exp\left(-i \alpha A\right) v`
    """
    return dot(expm(-1.0j*A*factor), v)


def arnoldi(A, v0, k):
    r"""Arnoldi algorithm to compute the Krylov approximation :math:`H` of a matrix :math:`A`.

    :param A: The matrix :math:`A` of shape :math:`N \times N` to approximate.
    :param v0: The initial vector :math:`v_0` of length :math:`N`. (Should be
               in matrix shape :math:`(N,1)` for practical reasons.)
    :param k: The number :math:`k` of Krylov steps performed.
    :return: A tuple :math:`(V, H)` where :math:`V` is the large matrix of shape
             :math:`N \times k` containing the orthogonal vectors and :math:`H` is the
             small matrix of shape :math:`k \times k` containing the Krylov approximation
             of :math:`A`.
    """
    V = mat(v0.copy() / norm(v0))
    H = mat(zeros((k+1,k)), dtype=complexfloating)
    for m in xrange(k):
        vt = A * V[:,m]
        for j in xrange(m+1):
            H[j,m] = (V[:,j].H*vt)[0,0]
            vt -= H[j,m] * V[:,j]
        H[m+1,m] = norm(vt)
        V = hstack((V, vt.copy()/H[m+1,m]))
    return (V, H)


def matrix_exp_arnoldi(A, v, factor, k):
    r"""Compute the solution of :math:`v' = A v` via :math:`k`
    steps of a the Arnoldi krylov method.

    :param A: The matrix :math:`A` of shape :math:`N \times N`.
    :param v: The vector :math:`v` of length :math:`N`.
    :param factor: An additional scalar factor :math:`\alpha`.
    :param k: The number :math:`k` of Krylov steps performed.
    :return: The (approximate) value of :math:`\exp\left(-i \alpha A\right) v`.
    """
    V, H = arnoldi(A, v, min(min(A.shape), k))
    eH = mat(expm(-1.0j*factor*H[:-1,:]))
    r = V[:,:-1] * eH[:,0]
    return asarray(r * norm(v))
