"""The WaveBlocks Project

IOM plugin providing functions for handling
linear combinations of general wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

import pickle
import hashlib
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

    # TODO: Consider an 'assume_constant' option
    #       None -> False -> lcsize dynamic, int -> lcsize fix
    # Could maybe avoid many resize operations

    # The overall group containing all lincombwp data
    grp_lc = self._srf[self._prefixb+str(blockid)].require_group("lincombwp")

    # TODO: Consider merging both cases

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_tg_c = grp_lc.create_dataset("timegrid_coefficients", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_tg_p = grp_lc.create_dataset("timegrid_packets", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_lcsize = grp_lc.create_dataset("lincomb_size", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        # Coefficients
        daset_ci = grp_lc.create_dataset("coefficients", (0, 0), dtype=np.complexfloating, chunks=True, maxshape=(None,None))
        # Packet IDs
        daset_refs = grp_lc.create_dataset("packet_refs", (0, 0), dtype=np.integer, chunks=True, maxshape=(None,None))
    else:
        # User specified how much space is necessary.
        daset_tg_c = grp_lc.create_dataset("timegrid_coefficients", (timeslots,), dtype=np.integer)
        daset_tg_p = grp_lc.create_dataset("timegrid_packets", (timeslots,), dtype=np.integer)
        daset_lcsize = grp_lc.create_dataset("lincomb_size", (timeslots,), dtype=np.integer)
        # Coefficients
        daset_ci = grp_lc.create_dataset("coefficients", (timeslots, 0), dtype=np.complexfloating, chunks=True, maxshape=(timeslots,None))
        # Packet IDs
        daset_refs = grp_lc.create_dataset("packet_refs", (timeslots, 0), dtype=np.integer, chunks=True, maxshape=(timeslots,None))

        # Mark all steps as invalid
        daset_tg_c[...] = -1.0
        daset_tg_p[...] = -1.0

    gid = self.create_group(groupid="wavepacketsLCblock"+str(blockid))
    daset_refs.attrs["packet_gid"] = gid

    # Attach pointer to timegrid
    daset_tg_c.attrs["pointer"] = 0
    daset_tg_p.attrs["pointer"] = 0


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
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid_coefficients"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombwp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp/coefficients"

    timeslot = self._srf[pathtg].attrs["pointer"]

    # Write the data
    self.must_resize(pathlcs, timeslot)
    J = np.size(coefficients)
    self._srf[pathlcs][timeslot] = J
    self.must_resize(pathd, timeslot)
    if not J == 0:
        self.must_resize(pathd, J-1, axis=1)
        self._srf[pathd][timeslot,:J] = np.squeeze(coefficients)

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


def save_lincombwp_wavepackets(self, packetlist, timestep=None, blockid=0):
    r"""Save the wavepackets being part of this linear combination.

    Note that this is quite an expensive operation.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid_packets"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp/packet_refs"
    gid = self._srf[pathd].attrs["packet_gid"]

    timeslot = self._srf[pathtg].attrs["pointer"]

    # Book keeping
    self.must_resize(pathd, timeslot)
    K = len(packetlist)
    if not K == 0:
        self.must_resize(pathd, K-1, axis=1)

    # Save that packets
    known_packets = self.get_block_ids(groupid=gid)
    for k, packet in enumerate(packetlist):
        bid = "LC"+str(blockid)+"WP"+str(packet.get_id())
        if not bid in known_packets:
            bid = self.create_block(blockid=bid, groupid=gid)
            descr = packet.get_description()
            self.add_wavepacket(descr, blockid=bid)

        # TODO: Generalise into generic packet saver
        self.save_wavepacket(packet, timestep=timestep, blockid=bid)

        # Book keeping
        self._srf[pathd][timeslot,k] = packet.get_id()

    # Write the timestep to which the stored packets belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


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


def load_lincombwp_timegrid(self, blockid=0, key=("coeffs", "packets")):
    r"""Load the timegrid of this linear combination.

    :param blockid: The ID of the data block to operate on.
    :param key: Specify which linear combination timegrids to load. All are independent.
    :type key: Tuple of valid identifier strings that are ``ceoffs`` and ``packets``.
               Default is ``("coeffs", "packets")``.
    """
    tg = []
    for item in key:
        if item == "coeffs":
            pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid_coefficients"
            tg.append(self._srf[pathtg][:])
        elif item == "packets":
            pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid_packets"
            tg.append(self._srf[pathtg][:])

    if len(tg) == 1:
        return tg[0]
    else:
        return tuple(tg)


def load_lincombwp_size(self, timestep=None, blockid=0):
    r"""Load the size (number of packets) of this linear combination.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid_coefficients"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombwp/lincomb_size"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self._srf[pathlcs][index]
    else:
        index = slice(None)
        return self._srf[pathlcs][index]


def load_lincombwp_coefficients(self, timestep=None, blockid=0):
    r"""Load the coefficients of this linear combination.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid_coefficients"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombwp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp/coefficients"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        J = self._srf[pathlcs][index]
        return self._srf[pathd][index,:J]
    else:
        index = slice(None)
        return self._srf[pathd][index,:]


def load_lincombwp_wavepackets(self, timestep, packetindex=None, blockid=0):
    r"""Load the wavepackets being part of this linear combination.

    Note that this is quite an expensive operation.

    :param timestep: Load only the data of this timestep.
    :param packetindex: Load only the packet with this index. If ``None``
                        then load all packets for the given timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombwp/timegrid_packets"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombwp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombwp/packet_refs"

    index = self.find_timestep_index(pathtg, timestep)
    J = self._srf[pathlcs][index]
    refs = self._srf[pathd][index,:J]

    if packetindex is None:
        packets = []
        for ref in refs:
            bid = "LC"+str(blockid)+"WP"+str(ref)
            # TODO: Generalise into generic packet saver
            packets.append(self.load_wavepacket(timestep=timestep, blockid=bid))

        return tuple(packets)
    else:
        if packetindex >= J:
            raise ValueError("Packet index is invalid.")

        bid = "LC"+str(blockid)+"WP"+str(refs[packetindex])
        # TODO: Generalise into generic packet saver
        return self.load_wavepacket(timestep=timestep, blockid=bid)


#
# The following two methods are only for convenience and are NOT particularly efficient.
#


def load_lincombwp(self, timestep, blockid=0):
    r"""Load a linear combination at a given timestep and return a fully configured
    :py:class:`LinearCombinationOfWPs` instance. This method just calls some other
    :py:class:`IOManager` methods in the correct order. It is included only for
    convenience and is not particularly efficient.

    :param timestep: The timestep :math:`n` we load the wavepacket.
    :param blockid: The ID of the data block to operate on.
    :return: A :py:class:`LinearCombinationOfWPs` instance.
    """
    from LinearCombinationOfWPs import LinearCombinationOfWPs

    descr = self.load_lincombwp_description(blockid=blockid)

    J = self.load_lincombwp_size(timestep=timestep, blockid=blockid)
    if J == 0:
        return None

    # Load the data
    c = self.load_lincombwp_coefficients(timestep=timestep, blockid=blockid)
    psi = self.load_lincombwp_wavepackets(timestep=timestep, blockid=blockid)

    # Assemble the linear combination
    LC = LinearCombinationOfWPs(descr["dimension"], descr["ncomponents"])
    LC.add_wavepackets(psi, c)

    return LC


def save_lincombwp(self, lincomb, timestep, blockid=None):
    r"""Save a linear combination of general wavepackets at a given timestep and read
    all data to save from the :py:class:`LinearCombinationOfWPs` instance provided. This
    method just calls some other :py:class:`IOManager` methods in the correct order.
    It is included only for convenience and is not particularly efficient. We assume
    the linear combination is already set up with the correct :py:meth:`add_lincombwp`
    method call.

    :param lincomb: The :py:class:`LinearCombinationOfWPs` instance we want to save.
    :param timestep: The timestep :math:`n` at which we save the linear combination.
    :param blockid: The ID of the data block to operate on.
    """
    # Description
    self.save_lincombwp_description(lincomb.get_description(), blockid=blockid)
    # Wavepackets
    self.save_lincombwp_wavepackets(lincomb.get_wavepackets(), timestep=timestep, blockid=blockid)
    # Coefficients
    self.save_lincombwp_coefficients(lincomb.get_coefficients(), timestep=timestep, blockid=blockid)
