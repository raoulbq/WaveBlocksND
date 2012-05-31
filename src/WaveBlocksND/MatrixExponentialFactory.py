"""The WaveBlocks Project

This file contains a simple function that selects the desired
matrix exponential routine.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial


def create_matrixexponential(description):
    """Returns the requested matrix exponential routine.

    :param description: A :py:class:`ParameterProvider` instance containing at least the
                       key ``matrix_exponential`` and depending on its values more keys.
    """
    method = description["matrix_exponential"]

    if method == "pade":
        from MatrixExponential import matrix_exp_pade
        return matrix_exp_pade
    elif method == "arnoldi":
        from MatrixExponential import matrix_exp_arnoldi
        try:
            arnoldi_steps = min(description["basis_size"], description["arnoldi_steps"])
        except:
            arnoldi_steps = description["arnoldi_steps"]
        return partial(matrix_exp_arnoldi, k=arnoldi_steps)
    else:
        raise ValueError("Unknown matrix exponential algorithm")
