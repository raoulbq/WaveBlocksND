"""The WaveBlocks Project

Various small utility functions.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze, asarray


def meshgrid_nd(arrays):
    """Like 'meshgrid()' but for arbitrary number of dimensions.
    """
    arrays = tuple(map(squeeze, arrays))

    if not len([ None for a in arrays if a.ndim != 1 ]) == 0:
        raise ValueError("Arrays must be 1-dimensional")

    # The dimension
    D = len(arrays)

    # The number of elements in each array
    nelements = map(len, arrays)

    result = []

    for d, a in enumerate(arrays):
        # The new shape
        shape = D * [1]
        shape[d] = nelements[d]

        # Reshape
        A = asarray(a).reshape(shape)

        # And repeat as many times as necessary
        for ax, n in enumerate(nelements):
            if not ax == d:
                A = A.repeat(n, axis=ax)

        result.append(A)

    return tuple(result)
