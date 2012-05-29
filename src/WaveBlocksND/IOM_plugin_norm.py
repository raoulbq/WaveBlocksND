"""The WaveBlocks Project

IOM plugin providing functions for handling norm data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_norm(self, parameters, timeslots=None, blockid=0):
    """Add storage for the norms.

    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the key `ncomponents`.
    :param blockid: The ID of the data block to operate on.
    """
    grp_ob = self._srf[self._prefixb+str(blockid)].require_group("observables")

    # Create the dataset with appropriate parameters
    grp_no = grp_ob.create_group("norm")

    if timeslots is None:
        # This case is event based storing
        daset_n = grp_no.create_dataset("norm", (0, parameters["ncomponents"]), dtype=np.floating, chunks=True, maxshape=(None,parameters["ncomponents"]))
        daset_tg = grp_no.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
    else:
        # User specified how much space is necessary.
        daset_n = grp_no.create_dataset("norm", (timeslots, parameters["ncomponents"]), dtype=np.floating)
        daset_tg = grp_no.create_dataset("timegrid", (timeslots,), dtype=np.integer)

    # Mark all steps as invalid
    daset_tg[...] = -1.0
    daset_tg.attrs["pointer"] = 0


def delete_norm(self, blockid=0):
    """Remove the stored norms.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/observables/norm"]
        # Check if there are other children, if not remove the whole node.
        if len(self._srf[self._prefixb+str(blockid)+"/observables"].keys()) == 0:
            del self._srf[self._prefixb+str(blockid)+"/observables"]
    except KeyError:
        pass


def has_norm(self, blockid=0):
    """Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return ("observables" in self._srf[self._prefixb+str(blockid)].keys() and
            "norm" in self._srf[self._prefixb+str(blockid)]["observables"].keys())


def save_norm(self, norm, timestep=None, blockid=0):
    """Save the norm of wavefunctions or wavepackets.

    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/norm/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/observables/norm/norm"
    timeslot = self._srf[pathtg].attrs["pointer"]

    # TODO: refactor, remove np.array
    norms = np.real(np.array(norm))

    # Write the data
    self.must_resize(pathd, timeslot)
    self._srf[pathd][timeslot,:] = norms

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


def load_norm_timegrid(self, blockid=0):
    """Load the timegrid corresponding to the norm data.

    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/norm/timegrid"
    return self._srf[pathtg][:]


def load_norm(self, timestep=None, split=False, blockid=0):
    """Load the norm data.

    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/norm/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/observables/norm/norm"

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
