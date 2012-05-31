"""The WaveBlocks Project

IOM plugin providing functions for handling the
propagation operators that appear in the Fourier
algorithm.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_fourieroperators(self, parameters, blockid=0):
    """Add storage for the Fourier propagation operators.

    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the keys `ncomponents` and `number_grid_nodes`.
    :param blockid: The ID of the data block to operate on.
    """
    grp_pr = self._srf[self._prefixb+str(blockid)].create_group("propagation")
    grp_op = grp_pr.create_group("operators")
    grp_op.create_dataset("opkinetic", list(parameters["number_nodes"]), np.complexfloating)
    grp_op.create_dataset("oppotential", [parameters["ncomponents"]**2]+list(parameters["number_nodes"]), np.complexfloating)


def delete_fourieroperators(self, blockid=0):
    """Remove the stored Fourier operators.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/propagation/operators"]
        # Check if there are other children, if not remove the whole node.
        if len(self._srf[self._prefixb+str(blockid)+"/propagation"].keys()) == 0:
            del self._srf[self._prefixb+str(blockid)+"/propagation"]
    except KeyError:
        pass


def has_fourieroperators(self, blockid=0):
    """Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return ("propagation" in self._srf[self._prefixb+str(blockid)].keys() and
            "operators" in self._srf[self._prefixb+str(blockid)]["propagation"].keys())


def save_fourieroperators(self, operators, blockid=0):
    """Save the kinetic and potential operator to a file.

    :param operators: The operators to save, given as tuple :math:`(T, V)`.
    :param blockid: The ID of the data block to operate on.
    """
    # Save the kinetic propagation operator
    path = "/"+self._prefixb+str(blockid)+"/propagation/operators/opkinetic"
    self._srf[path][...] = np.squeeze(operators[0].astype(np.complexfloating))
    # Save the potential propagation operator
    path = "/"+self._prefixb+str(blockid)+"/propagation/operators/oppotential"
    for index, item in enumerate(operators[1]):
        self._srf[path][index,...] = item.astype(np.complexfloating)


def load_fourieroperators(self, blockid=0):
    """
    :param blockid: The ID of the data block to operate on.
    """
    path = "/"+self._prefixb+str(blockid)+"/propagation/operators/"
    opT = self._srf[path+"opkinetic"]
    opV = self._srf[path+"oppotential"]
    opV = [ opV[index,...] for index in xrange(opV.shape[0]) ]

    return (opT, opV)
