"""The WaveBlocks Project

IOM plugin providing functions for handling wavefunction data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_wavefunction(self, parameters, flat=False, timeslots=None, blockid=0):
    r"""Add storage for the sampled wavefunction. The wavefunction is save
    in full hypercubic array shape with :math:`D` dimensions and :math:`N_d`
    data points in each direction.

    :param parameters: A :py:class:`ParameterProvider` instance containing
                       at least the keys `number_nodes` and `ncomponents`.
    :param flat: A flag indicating if we store wavefunctions :math:`\Psi(\Gamma)`
                 in a hypercubic shape :math:`(N_1, ..., N_D)` or
                 in a flat shape :math:`(D, |\Gamma|)` with :math:`|\Gamma| = \prod_i^D N_i`.
    :type flat: Boolean, default is ``False``.
    :param timeslots: The number of time slots we need. Can be set to ``None``
                      to get automatically growing datasets.
    :param blockid: The ID of the data block to operate on.
    """
    grp_wf = self._srf[self._prefixb+str(blockid)].require_group("wavefunction")

    # TODO: Remove quick hack:
    if flat:
        datashape = [parameters["ncomponents"], np.prod(parameters["number_nodes"])]
    else:
        datashape = [parameters["ncomponents"]] + list(parameters["number_nodes"])

    if timeslots is None:
        T = 0
        Ts = None
    else:
        T = timeslots
        Ts = timeslots

    # TODO: Improve chunking
    daset_psi_tg = grp_wf.create_dataset("timegrid", [T], dtype=np.integer, chunks=True, maxshape=[Ts], fillvalue=-1)
    daset_psi = grp_wf.create_dataset("Psi", [T]+datashape, dtype=np.complexfloating, chunks=True, maxshape=[Ts]+datashape)

    daset_psi_tg.attrs["pointer"] = 0


def delete_wavefunction(self, blockid=0):
    r"""Remove the stored wavefunction.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavefunction"]
    except KeyError:
        pass


def has_wavefunction(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return "wavefunction" in self._srf[self._prefixb+str(blockid)].keys()


def save_wavefunction(self, wavefunctionvalues, timestep=None, blockid=0):
    r"""Save the values :math:`\psi_i` of a :py:class:`WaveFunction` instance.

    :param wavefunctionvalues: A list of the values to save.
    :type wavefunctionvalues: A list of ndarrays.
    :param timestep: The timestep at which we save the data.
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
    r"""Load the wavefunction timegrid.

    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavefunction/timegrid"
    return self._srf[pathtg][...]


def load_wavefunction(self, timestep=None, blockid=0):
    r"""Load the wavefunction values.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavefunction/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavefunction/Psi"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self._srf[pathd][index,...]
    else:
        return self._srf[pathd][...]
