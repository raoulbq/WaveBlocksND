"""The WaveBlocks Project

IOM plugin providing functions for handling energy data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_energy(self, parameters, timeslots=None, blockid=0, total=False):
    """
    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the key `ncomponents`.
    :param blockid: The ID of the data block to operate on.
    """
    # TODO: refactor, keyword 'total'
    # Store the potential and kinetic energies
    grp_ob = self._srf[self._prefixb+str(blockid)].require_group("observables")

    # Create the dataset with appropriate parameters
    grp_en = grp_ob.create_group("energies")

    if timeslots is None:
        # This case is event based storing
        daset_ek = grp_en.create_dataset("kinetic", (0, parameters["ncomponents"]), dtype=np.floating, chunks=True, maxshape=(None,parameters["ncomponents"]))
        daset_ep = grp_en.create_dataset("potential", (0, parameters["ncomponents"]), dtype=np.floating, chunks=True, maxshape=(None,parameters["ncomponents"]))
        daset_tg = grp_en.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))

        if total is True:
            daset_to = grp_en.create_dataset("total", (0, 1), dtype=np.floating, chunks=True, maxshape=(None,1))
            daset_to.attrs["pointer"] = 0
    else:
        # User specified how much space is necessary.
        daset_ek = grp_en.create_dataset("kinetic", (timeslots, parameters["ncomponents"]), dtype=np.floating)
        daset_ep = grp_en.create_dataset("potential", (timeslots, parameters["ncomponents"]), dtype=np.floating)
        daset_tg = grp_en.create_dataset("timegrid", (timeslots,), dtype=np.integer)

        if total is True:
            daset_to = grp_en.create_dataset("total", (timeslots, 1), dtype=np.floating)
            daset_to.attrs["pointer"] = 0

    # Mark all steps as invalid
    daset_tg[...] = -1.0
    daset_tg.attrs["pointer"] = 0


def delete_energy(self, blockid=0):
    """Remove the stored energies

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/observables/energies"]
        # Check if there are other children, if not remove the whole node.
        if len(self._srf[self._prefixb+str(blockid)+"/observables"].keys()) == 0:
            del self._srf[self._prefixb+str(blockid)+"/observables"]
    except KeyError:
        pass


def has_energy(self, blockid=0):
    """Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return ("observables" in self._srf[self._prefixb+str(blockid)].keys() and
            "energies" in self._srf[self._prefixb+str(blockid)]["observables"].keys())


def save_energy(self, energies, timestep=None, blockid=0):
    """Save the kinetic and potential energies to a file.

    :param energies: A tuple :math:`(E_\text{kin}, E_\text{pot})` containing the energies.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid"
    pathd1 = "/"+self._prefixb+str(blockid)+"/observables/energies/kinetic"
    pathd2 = "/"+self._prefixb+str(blockid)+"/observables/energies/potential"
    timeslot = self._srf[pathtg].attrs["pointer"]

    # TODO: remove np.array
    ekin = np.real(np.array(energies[0]))
    epot = np.real(np.array(energies[1]))

    # Write the data
    self.must_resize(pathd1, timeslot)
    self.must_resize(pathd2, timeslot)
    self._srf[pathd1][timeslot,:] = ekin
    self._srf[pathd2][timeslot,:] = epot

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


def save_energy_total(self, total_energy, timestep=None, blockid=0):
    """Save the total energy to a file.

    :param total_energy: An array containing a time series of the total energy.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/total"

    timeslot = self._srf[pathd].attrs["pointer"]

    #todo: remove np,array
    etot = np.real(np.array(total_energy))

    # Write the data
    self.must_resize(pathd, timeslot)
    self._srf[pathd][timeslot,0] = etot

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def load_energy_timegrid(self, blockid=0):
    """
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid"
    return self._srf[pathtg][:]


def load_energy(self, timestep=None, split=False, blockid=0):
    """
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid"
    pathd1 = "/"+self._prefixb+str(blockid)+"/observables/energies/kinetic"
    pathd2 = "/"+self._prefixb+str(blockid)+"/observables/energies/potential"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        axis = 0
    else:
        index = slice(None)
        axis = 1

    if split is True:
        ekin = self.split_data( self._srf[pathd1][index,...], axis)
        epot = self.split_data( self._srf[pathd2][index,...], axis)
    else:
        ekin = self._srf[pathd1][index,...]
        epot = self._srf[pathd2][index,...]

    return (ekin, epot)


def load_energy_total(self, timestep=None, blockid=0):
    """
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/total"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
    else:
        index = slice(None)

    return self._srf[pathd][index,...]
