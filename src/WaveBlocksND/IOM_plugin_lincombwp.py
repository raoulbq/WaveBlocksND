"""The WaveBlocks Project

IOM plugin providing functions for handling
linear combinations of general wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

import pickle
import numpy as np


def add_lincombwp(self, parameters, timeslots=None, blockid=0):
    r"""Add storage for the linear combination of general wavepackets.

    :param parameters: An :py:class:`ParameterProvider` instance with at
                       least the key ``ncomponents``.
    :param timeslots: The number of time slots we need. Can be ``None``
                      to get automatically growing datasets.
    :param blockid: The ID of the data block to operate on.
    """
    N = parameters["ncomponents"]
    # TODO: Handle multi-component packets
    assert N == 1

    # TODO: Handle an 'assume_constant' option
    #       None -> False -> lcsize dynamic, int -> lcsize fix
    # Could maybe avoid many resize operations

    # The overall group containing all lincombwp data
    grp_lc = self._srf[self._prefixb+str(blockid)].require_group("lincombwp")

    # TODO: Consider merging both cases

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_tg = grp_lc.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_lcsize = grp_lc.create_dataset("lincomb_size", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        # Coefficients
        daset_ci = grp_lc.create_dataset("coefficients", (0, 0), dtype=np.complexfloating, chunks=True, maxshape=(None,None))
        # Packet IDs
        daset_pids = grp_lc.create_dataset("packet_ids", (0, 0), dtype=np.integer, chunks=True, maxshape=(None,None))
    else:
        # User specified how much space is necessary.
        daset_tg = grp_lc.create_dataset("timegrid", (timeslots,), dtype=np.integer)
        daset_lcsize = grp_lc.create_dataset("lincomb_size", (timeslots,), dtype=np.integer)
        # Coefficients
        daset_ci = grp_lc.create_dataset("coefficients", (timeslots, 0), dtype=np.complexfloating, chunks=True, maxshape=(timeslots,None))
        # Packet IDs
        daset_pids = grp_lc.create_dataset("packet_ids", (timeslots, 0), dtype=np.integer, chunks=True, maxshape=(timeslots,None))

        # Mark all steps as invalid
        daset_tg[...] = -1.0

    # Attach pointer to timegrid
    daset_tg.attrs["pointer"] = 0


def delete_lincombwp(self, blockid=0):
    r"""Remove the stored linear combination.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/lincombwp"]
    except KeyError:
        pass


def has_lincombwp(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return "lincombwp" in self._srf[self._prefixb+str(blockid)].keys()


def save_lincombwp_description(self, descr, blockid=0):
    r"""Save the description of this linear combination.

    :param descr: The description.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp"
    # Save the description
    for key, value in descr.iteritems():
        # Store all the values as pickled strings because hdf can
        # only store strings or ndarrays as attributes.
        self._srf[pathd].attrs[key] = pickle.dumps(value)


def save_lincombwp_coefficients(self, coefficients, timestep=None, blockid=0):
    r"""Save the coefficients of the linear combination to a file.

    :param coefficients: The coefficients of the linear combination of wavepackets.
    :type coefficients: A single, suitable ``ndarray``.
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombwp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp/coefficients"

    timeslot = self._srf[pathtg].attrs["pointer"]

    # Write the data
    self.must_resize(pathlcs, timeslot)
    self.must_resize(pathd, timeslot)
    J = np.size(coefficients)
    self.must_resize(pathd, J-1, axis=1)
    self._srf[pathlcs][timeslot] = J
    self._srf[pathd][timeslot,:J] = np.squeeze(coefficients)

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


def save_lincombwp_wavepackets(self, packetlist, blockid=0):
    raise NotImplementedError()


def load_lincombwp_description(self, blockid=0):
    r"""Load the description of this linear combination.

    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp"

    # Load and return all descriptions available
    descr = {}
    for key, value in self._srf[pathd].attrs.iteritems():
        descr[key] = pickle.loads(value)
    return descr


def load_lincombwp_timegrid(self, blockid=0):
    r"""Load the timegrid of this linear combination.

    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid"
    return self._srf[pathtg][:]


def load_lincombwp_coefficients(self, timestep=None, blockid=0):
    r"""Load the coefficients of this linear combination.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombwp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp/coefficients"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        J = self._srf[pathlcs][index]
        return self._srf[pathd][index,:J]
    else:
        index = slice(None)
        return self._srf[pathd][index,:]


def load_lincombwp_wavepackets(self, timestep=None, blockid=0):
    raise NotImplementedError()


#
# The following two methods are only for convenience and are NOT particularly efficient.
#


def load_lincombwp(self, timestep, blockid=0):
    raise NotImplementedError()


def save_lincombwp(self, lincomb, timestep, blockid=None):
    raise NotImplementedError()
