"""The WaveBlocks Project

Some selected functions for complex math.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, hstack, cumsum, diff, around, abs, angle, exp, sqrt, pi

__all__ = ["continuate", "cont_angle", "cont_sqrt"]


def continuate(data, jump=2.0*pi, reference=0.0):
    r"""Make the given data continuous by removing all jumps of size :math:`k\cdot\text{jump}`
    but not touching jumps of any other size. This can be used to overcome issues with
    the branch cut along the negative axis. There may be arise problems with jumps that
    are of size nearly :math:`k\cdot\text{jump}`.

    :param data: An array with the input data.
    :param jump: The basic size of jumps which will be removed. Default is :math:`2 \pi`.
    :param reference: This value allows the specify the starting point for continuation
                       explicitly. It can be used together with ``data``.
    :type reference: A single float number.
    """
    data = hstack([array(reference), array(data)])
    return (data - jump*hstack([ 0.0, cumsum(around( diff(data)/jump )) ]))[1:]


def cont_angle(data, reference=None):
    r"""Compute the angle of a complex number *not* constrained to
    the principal value and avoiding discontinuities at the branch cut.
    This function just applies 'continuate(.)' to the complex phase.

    :param data: An array with the input data.
    :param reference: This value allows the specify the starting point for
                      continuation explicitly. It can be used together with ``data``.
    :type reference: A single float number.
    """
    if reference is None:
        # Return just cont_f(x)
        return continuate(angle(data))
    else:
        # Return a 2-tuple ( cont_f(x), new_reference )
        return 2*( continuate(angle(data), reference=reference) ,)


def cont_sqrt(data, reference=None):
    r"""Compute the complex square root (following the Riemann surface)
    yields a result *not* constrained to the principal value and avoiding
    discontinuities at the branch cut. This function applies 'continuate(.)'
    to the complex phase and computes the complex square root according to
    the formula :math:`\sqrt{z} = \sqrt{r} \exp \left( i \cdot \frac{\phi}{2} \right)`.

    :param data: An array with the input data.
    :param reference: This value allows the specify the starting point for
                      continuation explicitly. It can be used together with ``data``.
    :type reference: A single float number.
    """
    if reference is None:
        # Return just cont_f(x)
        return sqrt(abs(data))*exp(1.0j*continuate(angle(data))/2)
    else:
        # Return a 2-tuple ( cont_f(x), new_reference )
        phi = continuate(angle(data), reference=reference)
        return (sqrt(abs(data))*exp(1.0j*phi/2), phi)


class ContinuousSqrt(object):
    r"""Class for computing continuous square roots. All stuff about
    referencing is hidden from the user.
    """

    def __init__(self, reference=0.0j):
        self._reference = reference

    def clone(self):
        return ContinuousSqrt(reference=self._reference)

    def __call__(self, radicand):
        root, ref = cont_sqrt(radicand, reference=self._reference)
        self._reference = ref[0]
        return root

    def set(self, newvalue):
        self._reference = newvalue

    def get(self):
        return self._reference
