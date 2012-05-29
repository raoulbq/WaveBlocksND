"""The WaveBlocks Project

IOM plugin providing functions for handling wavefunction data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_wavefunction(self, parameters, timeslots=None, blockid=0):
    """Add storage for the sampled wavefunction.

    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the keys `grid_number_nodes` and `ncomponents`.
    :param blockid: The ID of the data block to operate on.
    """
    grp_wf = self._srf[self._prefixb+str(blockid)].require_group("wavefunction")

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_psi = grp_wf.create_dataset("Psi", [0, parameters["ncomponents"]] + list(parameters["grid_number_nodes"]),
                                          dtype=np.complexfloating, chunks=True,
                                          maxshape=[None, parameters["ncomponents"]] + list(parameters["grid_number_nodes"]))
        daset_psi_tg = grp_wf.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
    else:
        # User specified how much space is necessary.
        daset_psi = grp_wf.create_dataset("Psi", [timeslots, parameters["ncomponents"]] + list(parameters["grid_number_nodes"]), dtype=np.complexfloating)
        daset_psi_tg = grp_wf.create_dataset("timegrid", (timeslots,), dtype=np.integer)

    # Mark all steps as invalid
    daset_tg[...] = -1.0
    daset_psi_tg.attrs["pointer"] = 0


def delete_wavefunction(self, blockid=0):
    """Remove the stored wavefunction.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavefunction"]
    except KeyError:
        pass


def has_wavefunction(self, blockid=0):
    """Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return "wavefunction" in self._srf[self._prefixb+str(blockid)].keys()


def save_wavefunction(self, wavefunctionvalues, blockid=0, timestep=None):
    """Save the values :math:`\psi_i` of a :py:class:`WaveFunction` instance.

    :param wavefunctionvalues: A list of the values to save.
    :type wavefunctionvalues: A list of ndarrays.
    :param blockid: The ID of the data block to operate on.
    """
    #@refactor: take wavefunction or wavefunction.get_values() as input?
    pathtg = "/"+self._prefixb+str(blockid)+"/wavefunction/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavefunction/Psi"
    timeslot = self._srf[pathtg].attrs["pointer"]

    # Store the values given
    self.must_resize(pathd, timeslot)

    for index, item in enumerate(wavefunctionvalues):
        self._srf[pathd][timeslot,index,...] = item

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


def load_wavefunction_timegrid(self, blockid=0):
    """
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavefunction/timegrid"
    return self._srf[pathtg][...]


def load_wavefunction(self, timestep=None, blockid=0):
    """
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavefunction/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavefunction/Psi"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self._srf[pathd][index,...]
    else:
        return self._srf[pathd][...]
