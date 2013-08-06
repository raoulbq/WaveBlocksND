"""The WaveBlocks Project

IOM plugin providing functions for handling autocorrelation data.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_autocorrelation(self, parameters, timeslots=None, blockid=0):
    r"""Add storage for the auto correlation computations.

    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the key `ncomponents`.
    :param timeslots: The number of time slots we need. Can be set to ``None``
                      to get automatically growing datasets.
    :param blockid: The ID of the data block to operate on.
    """
    N = parameters["ncomponents"]

    if timeslots is None:
        T = 0
        Ts = None
    else:
        T = timeslots
        Ts = timeslots

    # Check that the "observables" group is present
    grp_ob = self._srf[self._prefixb+str(blockid)].require_group("observables")
    # Create the dataset with appropriate parameters
    grp_ac = grp_ob.create_group("autocorrelation")
    daset_tg = grp_ac.create_dataset("timegrid", (T,), dtype=np.integer, chunks=True, maxshape=(Ts,))
    grp_ac.create_dataset("autocorrelation", (T,N), dtype=np.complexfloating, chunks=(64,N), maxshape=(Ts,N))
    daset_tg.attrs["pointer"] = 0


def delete_autocorrelation(self, blockid=0):
    r"""Remove the stored autocorrelations.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/observables/autocorrelation"]
        # Check if there are other children, if not remove the whole node.
        if len(self._srf[self._prefixb+str(blockid)+"/observables"].keys()) == 0:
            del self._srf[self._prefixb+str(blockid)+"/observables"]
    except KeyError:
        pass


def has_autocorrelation(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return ("observables" in self._srf[self._prefixb+str(blockid)].keys() and
            "autocorrelation" in self._srf[self._prefixb+str(blockid)]["observables"].keys())


def save_autocorrelation(self, autocorrelations, timestep=None, blockid=0):
    r"""Save the autocorrelation of wavefunctions or wavepackets.

    :param autocorrelations: The autocorrelation values to save.
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/autocorrelation/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/observables/autocorrelation/autocorrelation"
    timeslot = self._srf[pathtg].attrs["pointer"]

    # TODO: refactor, remove np.array
    autocorrelations = np.squeeze(np.array(autocorrelations))

    # Write the data
    self.must_resize(pathd, timeslot)
    self._srf[pathd][timeslot,:] = autocorrelations

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


def load_autocorrelation_timegrid(self, blockid=0):
    r"""Load the timegrid corresponding to the autocorrelation data.

    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/autocorrelation/timegrid"
    return self._srf[pathtg][:]


def load_autocorrelation(self, timestep=None, split=False, blockid=0):
    r"""Load the autocorrelation data.

    :param timestep: Load only the data of this timestep.
    :param split: Split the data array into one array for each component.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/autocorrelation/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/observables/autocorrelation/autocorrelation"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        axis = 0
    else:
        index = slice(None)
        axis = 1

    if split is True:
        return self.split_data( self._srf[pathd][index,...], axis)
    else:
        return self._srf[pathd][index,...]
