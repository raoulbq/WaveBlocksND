"""The WaveBlocks Project

IOM plugin providing functions for handling of generic wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

def add_genericwp(self, description, timeslots=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Add storage for the homogeneous wavepackets.

    :param description: An :py:class:`ParameterProvider` instance with at
                       least the keys ``dimension`` and ``ncomponents``.
    :param timeslots: The number of time slots we need. Can be ``None``
                      to get automatically growing datasets.
    :param blockid: The ID of the data block to operate on.
    """
    packet_type = description["type"]

    if packet_type == "HagedornWavepacket":
        self.add_wavepacket(description, timeslots=timeslots, blockid=blockid)
    elif packet_type == "HagedornWavepacketInhomogeneous":
        self.add_inhomogwavepacket(description, timeslots=timeslots, blockid=blockid)
    else:
        raise ValueError("Unknown waavepacket type: "+str(packet_type))


def delete_genericwp(self, blockid=0):
    r"""Remove the stored wavepackets.

    :param blockid: The ID of the data block to operate on.
    """
    self.delete_wavepacket(blockid=blockid)
    self.delete_inhomogwavepacket(blockid=blockid)


def has_wavepacket(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    has_packet = (
        self.has_wavepacket(blockid=blockid) or
        self.has_inhomogwavepacket(blockid=blockid)
    )
    return has_packet


#
# The following two methods are NOT particularly efficient.
#


def load_genericwp(self, timestep, blockid=0):
    r"""Load a wavepacket at a given timestep and return a fully configured instance.
    This method just calls some other :py:class:`IOManager` methods in the correct
    order. It is included only for convenience and is not particularly efficient.

    :param timestep: The timestep :math:`n` we load the wavepacket.
    :param blockid: The ID of the data block to operate on.
    :return: A :py:class:`Wavepacket` subclass instance.
    """
    if self.has_wavepacket(blockid=blockid):
        return self.load_wavepacket(timestep=timestep, blockid=blockid)
    elif self.has_inhomogwavepacket(blockid=blockid):
        return self.load_inhomogwavepacket(timestep=timestep, blockid=blockid)
    else:
        raise ValueError("No wavepacket of know type found in block "+str(blockid))


def save_genericwp(self, packet, timestep, blockid=None):
    r"""Save a wavepacket at a given timestep and read all data to save from the
    :py:class:`Wavepacket` subclass instance provided. This method just calls some
    other :py:class:`IOManager` methods in the correct order. It is included only
    for convenience and is not particularly efficient. We assume the wavepacket
    is already set up with the correct :py:meth:`add` method call.

    :param packet: A generic :py:class:`Wavepacket` subclass instance we want to save.
    :param timestep: The timestep :math:`n` at which we save the wavepacket.
    :param blockid: The ID of the data block to operate on.
    """
    packet_type = packet.get_description()["type"]

    if packet_type == "HagedornWavepacket":
        self.save_wavepacket(packet, timestep=timestep, blockid=blockid)
    elif packet_type == "HagedornWavepacketInhomogeneous":
        self.save_inhomogwavepacket(packet, timestep=timestep, blockid=blockid)
    else:
        raise ValueError("Unknown waavepacket type: "+str(packet_type))
