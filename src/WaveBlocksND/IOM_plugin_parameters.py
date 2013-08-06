"""The WaveBlocks Project

IOM plugin providing functions for handling simulation parameter data.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import pickle

import ParameterProvider as ParameterProvider


def add_parameters(self, blockid="global"):
    r"""Add storage for the simulation parameters.

    :param blockid: The ID of the data block to operate on.
    """
    # Store the simulation parameters
    # We are only interested in the attributes of this data set
    # as they are used to store the simulation parameters.
    self._srf[self._prefixb+str(blockid)].create_dataset("simulation_parameters", (1,1))


def delete_parameters(self, blockid="global"):
    r"""Remove the stored simulation parameters.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/simulation_parameters"]
    except KeyError:
        pass


def has_parameters(self, blockid="global"):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return "simulation_parameters" in self._srf[self._prefixb+str(blockid)].keys()


def save_parameters(self, parameters, blockid="global"):
    r"""Save the simulation parameters.

    :param parameters: The simulation parameters to store.
    :param blockid: The ID of the data block to operate on.
    """
    paset = self._srf["/"+self._prefixb+str(blockid)+"/simulation_parameters"]

    for param, value in parameters:
        # Store all the values as pickled strings because hdf can
        # only store strings or ndarrays as attributes.
        paset.attrs[param] = pickle.dumps(value)


def load_parameters(self, blockid="global"):
    r"""Load the simulation parameters.

    :param blockid: The ID of the data block to operate on.
    """
    p = self._srf["/"+self._prefixb+str(blockid)+"/simulation_parameters"].attrs
    PP = ParameterProvider.ParameterProvider()

    for key, value in p.iteritems():
        PP[key] = pickle.loads(value)

    # Compute some values on top of the given input parameters
    PP.compute_parameters()

    return PP


def update_parameters(self, parameters, blockid="global"):
    r"""Update the parameters by some new values.

    :param parameters: The parameters containing updated values.
    :param blockid: The ID of the data block to operate on.
    """
    params = self.load_parameters(blockid=blockid)
    self.delete_parameters(blockid=blockid)
    params.update_parameters(parameters)
    self.add_parameters(blockid=blockid)
    self.save_parameters(params, blockid=blockid)
