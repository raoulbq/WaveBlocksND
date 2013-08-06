"""The WaveBlocks Project

IOM plugin providing functions for handling energy data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_energy(self, parameters, timeslots=None, blockid=0, key=("kin", "pot")):
    r"""
    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the key `ncomponents`.
    :param timeslots: The number of time slots we need. Can be set to ``None``
                      to get automatically growing datasets.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which energies to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``kin``, ``pot`` and ``tot``.
               Default is ``("kin", "pot")``.
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
    # Add a new group for energies
    grp_en = grp_ob.create_group("energies")

    # Add all requested data sets
    if "kin" in key and not "kinetic" in grp_en.keys():
        daset_tgek = grp_en.create_dataset("timegrid_kin", (T,), dtype=np.integer, chunks=True, maxshape=(Ts,), fillvalue=-1)
        grp_en.create_dataset("kinetic", (T,N), dtype=np.floating, chunks=(64,N), maxshape=(Ts,N))
        daset_tgek.attrs["pointer"] = 0

    if "pot" in key and not "potential" in grp_en.keys():
        daset_tgep = grp_en.create_dataset("timegrid_pot", (T,), dtype=np.integer, chunks=True, maxshape=(Ts,), fillvalue=-1)
        grp_en.create_dataset("potential", (T,N), dtype=np.floating, chunks=(64,N), maxshape=(Ts,N))
        daset_tgep.attrs["pointer"] = 0

    if "tot" in key and not "total" in grp_en.keys():
        daset_tget = grp_en.create_dataset("timegrid_tot", (T,), dtype=np.integer, chunks=True, maxshape=(Ts,), fillvalue=-1)
        grp_en.create_dataset("total", (T,1), dtype=np.floating, chunks=(64,1), maxshape=(Ts,1))
        daset_tget.attrs["pointer"] = 0


def delete_energy(self, blockid=0):
    r"""Remove the stored energies

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/observables/energies"]
        # Check if there are other children, if not remove the whole node.
        if len(self._srf[self._prefixb+str(blockid)+"/observables"].keys()) == 0:
            del self._srf[self._prefixb+str(blockid)+"/observables"]
    except KeyError:
        pass


def has_energy(self, blockid=0, key=("kin", "pot")):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    :param key: Specify which energies to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``kin``, ``pot`` and ``tot``.
               Default is ``("kin", "pot")``.
    """
    r = True
    r &= ("observables" in self._srf[self._prefixb+str(blockid)].keys())

    if r is True:
        r &= ("energies" in self._srf[self._prefixb+str(blockid)]["observables"].keys())

    if r and "kin" in key:
        r &= ("kinetic" in self._srf[self._prefixb+str(blockid)]["observables/energies"].keys())
    if r and "pot" in key:
        r &= ("potential" in self._srf[self._prefixb+str(blockid)]["observables/energies"].keys())
    if r and "tot" in key:
        r &= ("total" in self._srf[self._prefixb+str(blockid)]["observables/energies"].keys())

    return r


def save_energy(self, energies, timestep=None, blockid=0, key=("kin", "pot")):
    r"""Save the kinetic and potential energies to a file.

    :param energies: A tuple containing the energies. The order is important,
                     it has to match the order in the ``key`` argument. Per default
                     the order has to be :math:`(E_\text{kin}, E_\text{pot})`.
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which energies to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``kin``, ``pot`` and ``tot``.
               Default is ``("kin", "pot")``.
    """
    for item, datum in zip(key, energies):
        if item == "kin":
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_kin"
            pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/kinetic"
        elif item == "pot":
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_pot"
            pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/potential"
        elif item == "tot":
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_tot"
            pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/total"

        timeslot = self._srf[pathtg].attrs["pointer"]

        # TODO: remove np.array
        energy = np.real(np.array(datum))

        # Write the data
        self.must_resize(pathd, timeslot)
        self._srf[pathd][timeslot,:] = energy

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self._srf[pathtg][timeslot] = timestep

        # Update the pointer
        self._srf[pathtg].attrs["pointer"] += 1


def load_energy_timegrid(self, blockid=0, key=("kin", "pot")):
    r"""Load the time grid for specified energies.

    :param blockid: The ID of the data block to operate on.
    :param key: Specify which energy timegrids to load. All are independent.
    :type key: Tuple of valid identifier strings that are ``kin``, ``pot`` and ``tot``.
               Default is ``("kin", "pot")``.
    """
    tg = []
    for item in key:
        if item == "kin":
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_kin"
            tg.append(self._srf[pathtg][:])
        elif item == "pot":
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_pot"
            tg.append(self._srf[pathtg][:])
        elif item == "tot":
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_tot"
            tg.append(self._srf[pathtg][:])

    if len(tg) == 1:
        return tg[0]
    else:
        return tuple(tg)


def load_energy(self, timestep=None, split=False, blockid=0, key=("kin", "pot")):
    r"""Load the energy data.

    :param timestep: Load only the data of this timestep.
    :param split: Split the array into arrays for each component.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which energies to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``kin``, ``pot`` and ``tot``.
               Default is ``("kin", "pot")``.
    """
    result = []

    for item in key:
        if item == "kin":
            pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/kinetic"
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_ek"
        elif item == "pot":
            pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/potential"
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_ep"
        elif item == "tot":
            pathd = "/"+self._prefixb+str(blockid)+"/observables/energies/total"
            pathtg = "/"+self._prefixb+str(blockid)+"/observables/energies/timegrid_et"

        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            axis = 0
        else:
            index = slice(None)
            axis = 1

        if split is True:
            energy = self.split_data( self._srf[pathd][index,...], axis)
        else:
            energy = self._srf[pathd][index,...]

        result.append(energy)

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)
