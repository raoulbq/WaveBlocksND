"""The WaveBlocks Project

This file contains a simple function that selects the desired
matrix exponential routine.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial


class MatrixExponentialFactory:
    """A factory for matrix exponential routines.
    """

    def __init__(self):
        pass


    def get_matrixexponential(self, parameters):
        """Returns the requested matrix exponential routine.

        :param parameters: A :py:class:`ParameterProvider` instance containing at least the
                           key ``matrix_exponential`` and depending on its values more keys.
        """
        method = parameters["matrix_exponential"]

        if method == "pade":
            from MatrixExponential import matrix_exp_pade
            return matrix_exp_pade
        elif method == "arnoldi":
            from MatrixExponential import matrix_exp_arnoldi
            try:
                arnoldi_steps = min(parameters["basis_size"], parameters["arnoldi_steps"])
            except:
                arnoldi_steps = parameters["arnoldi_steps"]
            return partial(matrix_exp_arnoldi, k=arnoldi_steps)
        else:
            raise ValueError("Unknown matrix exponential algorithm")
