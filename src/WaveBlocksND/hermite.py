"""The WaveBlocks Project

This file implements computation of Guass-Hermite
quadrature nodes and weights. For high order a novel
asymptotic linear time algorithm is used.

@author: R. Bourquin
@copyright: Copyright (C) 2014, 2015 R. Bourquin
@license: Modified BSD License
"""

from numpy import (sqrt, exp, sin, cos, arccos, pi, floor, ceil, around, arange,
                   array, ones_like, zeros_like, row_stack, column_stack, hstack,
                   flipud, floating, dot)
from scipy.special import airy


def pbcf_asy_s(n, theta):
    """Asymptotic series expansion of parabolic cylinder function.
    """
    mu = sqrt(2.0*n + 1.0)
    st = sin(theta)
    ct = cos(theta)
    # http://dlmf.nist.gov/12.10#vii
    mu = 2.0*n + 1.0
    # http://dlmf.nist.gov/12.10#E23
    eta = 0.5*theta - 0.5*st*ct
    # http://dlmf.nist.gov/12.10#E39
    zeta = -(3.0*eta/2.0) ** (2.0/3.0)
    # http://dlmf.nist.gov/12.10#E40
    phi = (-zeta / st**2) ** (0.25)
    # Coefficients
    # http://dlmf.nist.gov/12.10#E43
    a0 =  1.0
    a1 =  0.10416666666666666667
    a2 =  0.08355034722222222222
    a3 =  0.12822657455632716049
    a4 =  0.29184902646414046425
    a5 =  0.88162726744375765242
    b0 =  1.0
    b1 = -0.14583333333333333333
    b2 = -0.09874131944444444444
    b3 = -0.14331205391589506173
    b4 = -0.31722720267841354810
    b5 = -0.94242914795712024914
    a = (a0, a1, a2, a3, a4, a5)
    b = (b0, b1, b2, b3, b4, b5)
    # Polynomials
    # http://dlmf.nist.gov/12.10#E9
    # http://dlmf.nist.gov/12.10#E10
    u0 = 1.0
    u1 = (       1.0*ct**3 -          6.0*ct) / 24.0
    u2 = (      -9.0*ct**4 +        249.0*ct**2 +         145.0) / 1152.0
    u3 = (   -4042.0*ct**9 +      18189.0*ct**7 -       28287.0*ct**5 -      151995.0*ct**3 -     259290.0*ct) / 414720.0
    u4 = (   72756.0*ct**10 -    321339.0*ct**8 -      154982.0*ct**6 +    50938215.0*ct**4 +  122602962.0*ct**2 + 12773113.0) / 39813120.0
    u5 = (82393456.0*ct**15 - 617950920.0*ct**13 + 1994971575.0*ct**11 - 3630137104.0*ct**9 + 4433574213.0*ct**7
          - 37370295816.0*ct**5 - 119582875013.0*ct**3 -34009066266.0*ct) /6688604160.0
    #v0 = 1.0
    #v1 = (      1.0*ct**3 +           6.0*ct) / 24.0
    #v2 = (     15.0*ct**4 -         327.0*ct**2 -         143.0) / 1152.0
    #v3 = (  -4042.0*ct**9 +       18189.0*ct**7 -       36387.0*ct**5 +      238425.0*ct**3 +     259290.0*ct) / 414720.0
    #v4 = (-121260.0*ct**10 +     551733.0*ct**8 -      151958.0*ct**6 -    57484425.0*ct**4 -  132752238.0*ct**2 - 12118727) / 39813120.0
    #v5 = (82393456.0*ct**15 - 617950920.0*ct**13 + 2025529095.0*ct**11 - 3750839308.0*ct**9 + 3832454253.0*ct**7
    #      + 35213253348.0*ct**5 + 130919230435.0*ct**3 + 34009066266*ct) / 6688604160.0
    u = (u0, u1, u2, u3, u4, u5)
    #v = (v0, v1, v2, v3, v4, v5)
    # Airy Evaluation (Bi and Bip unused)
    Ai, Aip, Bi, Bip = airy(mu**(4.0/6.0) * zeta)
    # Prefactor for U
    P = 2.0*sqrt(pi) * mu**(1.0/6.0) * phi
    # Terms for U
    # http://dlmf.nist.gov/12.10#E42
    A0 =   b[0]*u[0]
    A1 =  (b[2]*u[0] + phi**6*b[1]*u[1] + phi**12*b[0]*u[2]) / zeta**3
    A2 =  (b[4]*u[0] + phi**6*b[3]*u[1] + phi**12*b[2]*u[2] + phi**18*b[1]*u[3] + phi**24*b[0]*u[4]) / zeta**6
    B0 = -(a[1]*u[0] + phi**6*a[0]*u[1]) / zeta**2
    B1 = -(a[3]*u[0] + phi**6*a[2]*u[1] + phi**12*a[1]*u[2] + phi**18*a[0]*u[3]) / zeta**5
    B2 = -(a[5]*u[0] + phi**6*a[4]*u[1] + phi**12*a[3]*u[2] + phi**18*a[2]*u[3] + phi**24*a[1]*u[4] + phi**30*a[0]*u[5]) / zeta**8
    # U
    # http://dlmf.nist.gov/12.10#E35
    U = P * (Ai  * (A0 + A1/mu**2.0 + A2/mu**4.0) +
             Aip * (B0 + B1/mu**2.0 + B2/mu**4.0) / mu**(8.0/6.0))
    # Prefactor for derivative of U
    #Pd = sqrt(2.0*pi) * mu**(2.0/6.0) / phi
    # Terms for derivative of U
    # http://dlmf.nist.gov/12.10#E46
    #C0 = -(b[1]*v[0] + phi**6*b[0]*v[1]) / zeta
    #C1 = -(b[3]*v[0] + phi**6*b[2]*v[1] + phi**12*b[1]*v[2] + phi**18*b[0]*v[3]) / zeta**4
    #C2 = -(b[5]*v[0] + phi**6*b[4]*v[1] + phi**12*b[3]*v[2] + phi**18*b[2]*v[3] + phi**24*b[1]*v[4] + phi**30*b[0]*v[5]) / zeta**7
    #D0 =   a[0]*v[0]
    #D1 =  (a[2]*v[0] + phi**6*a[1]*v[1] + phi**12*a[0]*v[2]) / zeta**3
    #D2 =  (a[4]*v[0] + phi**6*a[3]*v[1] + phi**12*a[2]*v[2] + phi**18*a[1]*v[3] + phi**24*a[0]*v[4]) / zeta**6
    # Derivative of U
    # http://dlmf.nist.gov/12.10#E36
    #Ud = Pd * (Ai  * (C0 + C1/mu**2.0 + C2/mu**4.0) / mu**(4.0/6.0) +
    #           Aip * (D0 + D1/mu**2.0 + D2/mu**4.0) )
    return U#, Ud


def hermite_asy_s(n, x):
    mu = sqrt(2.0*n + 1.0)
    t = x / mu
    theta = arccos(t)
    u = pbcf_asy_s(n, theta)
    return u







# OLD CODE

def pbcf(n, theta):
    """Asymptotic series expansion of parabolic cylinder function

    The implementation is based on sections 3.2 and 3.3 from the
    original paper. Compared to the published version this code
    adds one more term to the asymptotic series. The detailed
    formulas can be found at [parabolic-asymptotics]_. The evaluation
    is done in a transformed variable :math:`\theta := \arccos(t)`
    where :math:`t := x / \mu` and :math:`\mu := \sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    theta : ndarray
        Transformed position variable

    Returns
    -------
    U : ndarray
        Value of the parabolic cylinder function :math:`U(a, \theta)`.
    Ud : ndarray
        Value of the derivative :math:`U^{\prime}(a, \theta)` of
        the parabolic cylinder function.

    See Also
    --------
    h_roots_asy

    References
    ----------
    .. [parabolic-asymptotics]
       http://dlmf.nist.gov/12.10#vii
    """
    mu = sqrt(2.0*n + 1.0)
    st = sin(theta)
    ct = cos(theta)
    # http://dlmf.nist.gov/12.10#vii
    mu = 2.0*n + 1.0
    # http://dlmf.nist.gov/12.10#E23
    eta = 0.5*theta - 0.5*st*ct
    # http://dlmf.nist.gov/12.10#E39
    zeta = -(3.0*eta/2.0) ** (2.0/3.0)
    # http://dlmf.nist.gov/12.10#E40
    phi = (-zeta / st**2) ** (0.25)
    # Coefficients
    # http://dlmf.nist.gov/12.10#E43
    a0 =  1.0
    a1 =  0.10416666666666666667
    a2 =  0.08355034722222222222
    a3 =  0.12822657455632716049
    a4 =  0.29184902646414046425
    a5 =  0.88162726744375765242
    b0 =  1.0
    b1 = -0.14583333333333333333
    b2 = -0.09874131944444444444
    b3 = -0.14331205391589506173
    b4 = -0.31722720267841354810
    b5 = -0.94242914795712024914
    a = (a0, a1, a2, a3, a4, a5)
    b = (b0, b1, b2, b3, b4, b5)
    # Polynomials
    # http://dlmf.nist.gov/12.10#E9
    # http://dlmf.nist.gov/12.10#E10
    u0 = 1.0
    ctp = ct ** arange(16).reshape((-1,1))
    u1 = (       1.0*ctp[3,:] -          6.0*ct) / 24.0
    u2 = (      -9.0*ctp[4,:] +        249.0*ctp[2,:] +         145.0) / 1152.0
    u3 = (   -4042.0*ctp[9,:] +      18189.0*ctp[7,:] -       28287.0*ctp[5,:] -      151995.0*ctp[3,:] -     259290.0*ct) / 414720.0
    u4 = (   72756.0*ctp[10,:] -    321339.0*ctp[8,:] -      154982.0*ctp[6,:] +    50938215.0*ctp[4,:] +  122602962.0*ctp[2,:] + 12773113.0) / 39813120.0
    u5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 1994971575.0*ctp[11,:] - 3630137104.0*ctp[9,:] + 4433574213.0*ctp[7,:]
          - 37370295816.0*ctp[5,:] - 119582875013.0*ctp[3,:] - 34009066266.0*ct) / 6688604160.0
    v0 = 1.0
    v1 = (       1.0*ctp[3,:] +           6.0*ct) / 24.0
    v2 = (      15.0*ctp[4,:] -         327.0*ctp[2,:] -         143.0) / 1152.0
    v3 = (   -4042.0*ctp[9,:] +       18189.0*ctp[7,:] -       36387.0*ctp[5,:] +      238425.0*ctp[3,:] +     259290.0*ct) / 414720.0
    v4 = ( -121260.0*ctp[10,:] +     551733.0*ctp[8,:] -      151958.0*ctp[6,:] -    57484425.0*ctp[4,:] -  132752238.0*ctp[2,:] - 12118727) / 39813120.0
    v5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 2025529095.0*ctp[11,:] - 3750839308.0*ctp[9,:] + 3832454253.0*ctp[7,:]
          + 35213253348.0*ctp[5,:] + 130919230435.0*ctp[3,:] + 34009066266*ct) / 6688604160.0
    u = (u0, u1, u2, u3, u4, u5)
    v = (v0, v1, v2, v3, v4, v5)
    # Airy Evaluation (Bi and Bip unused)
    Ai, Aip, Bi, Bip = airy(mu**(4.0/6.0) * zeta)
    # Prefactor for U
    P = 2.0*sqrt(pi) * mu**(1.0/6.0) * phi
    # Terms for U
    # http://dlmf.nist.gov/12.10#E42
    A0 =   b[0]*u[0]
    A1 =  (b[2]*u[0] + phi**6*b[1]*u[1] + phi**12*b[0]*u[2]) / zeta**3
    A2 =  (b[4]*u[0] + phi**6*b[3]*u[1] + phi**12*b[2]*u[2] + phi**18*b[1]*u[3] + phi**24*b[0]*u[4]) / zeta**6
    B0 = -(a[1]*u[0] + phi**6*a[0]*u[1]) / zeta**2
    B1 = -(a[3]*u[0] + phi**6*a[2]*u[1] + phi**12*a[1]*u[2] + phi**18*a[0]*u[3]) / zeta**5
    B2 = -(a[5]*u[0] + phi**6*a[4]*u[1] + phi**12*a[3]*u[2] + phi**18*a[2]*u[3] + phi**24*a[1]*u[4] + phi**30*a[0]*u[5]) / zeta**8
    # U
    # http://dlmf.nist.gov/12.10#E35
    U = P * (Ai  * (A0 + A1/mu**2.0 + A2/mu**4.0) +
             Aip * (B0 + B1/mu**2.0 + B2/mu**4.0) / mu**(8.0/6.0))
    # Prefactor for derivative of U
    Pd = sqrt(2.0*pi) * mu**(2.0/6.0) / phi
    # Terms for derivative of U
    # http://dlmf.nist.gov/12.10#E46
    C0 = -(b[1]*v[0] + phi**6*b[0]*v[1]) / zeta
    C1 = -(b[3]*v[0] + phi**6*b[2]*v[1] + phi**12*b[1]*v[2] + phi**18*b[0]*v[3]) / zeta**4
    C2 = -(b[5]*v[0] + phi**6*b[4]*v[1] + phi**12*b[3]*v[2] + phi**18*b[2]*v[3] + phi**24*b[1]*v[4] + phi**30*b[0]*v[5]) / zeta**7
    D0 =   a[0]*v[0]
    D1 =  (a[2]*v[0] + phi**6*a[1]*v[1] + phi**12*a[0]*v[2]) / zeta**3
    D2 =  (a[4]*v[0] + phi**6*a[3]*v[1] + phi**12*a[2]*v[2] + phi**18*a[1]*v[3] + phi**24*a[0]*v[4]) / zeta**6
    # Derivative of U
    # http://dlmf.nist.gov/12.10#E36
    Ud = Pd * (Ai  * (C0 + C1/mu**2.0 + C2/mu**4.0) / mu**(4.0/6.0) +
               Aip * (D0 + D1/mu**2.0 + D2/mu**4.0) )
    return U, Ud
