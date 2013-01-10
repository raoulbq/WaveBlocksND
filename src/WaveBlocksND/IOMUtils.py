"""The WaveBlocks Project

Various small utility functions.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from BlockFactory import BlockFactory


def load_wavepacket(iom, timestep, blockid=0):
    r"""Utility function to load a homogeneous
    wavepacket from an :py:class:`IOManager` instance.

    :param iom: The :py:class:`IOManager` instance from which to load data.
    :param timestep: Load the data corresponding to the given `timestep`.
    :param blockid: The `datablock` from which to read the data.
                    Default is the block with `blockid=0`.

    Note: This function is a pure utility function and is not efficient. It is
          built for interactive use only and should not be use in scripts.
    """
    if not iom.has_wavepacket(blockid=blockid):
        print("There is no wavepacket to load in block "+str(blockid))
        return

    BF = BlockFactory()

    wpd = iom.load_wavepacket_description(blockid=blockid)
    HAWP = BF.create_wavepacket(wpd)

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes(blockid=blockid)
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BF.create_basis_shape(descr)

    KEY = ("q","p","Q","P","S","adQ")
    # Retrieve simulation data
    params = iom.load_wavepacket_parameters(timestep=timestep, blockid=blockid, key=KEY)
    hashes, coeffs = iom.load_wavepacket_coefficients(timestep=timestep, get_hashes=True, blockid=blockid)
    # Configure the wavepacket
    HAWP.set_parameters(params, key=KEY)
    HAWP.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
    HAWP.set_coefficients(coeffs)

    return HAWP


def load_wavepacket_inhomogeneous(iom, timestep, blockid=0):
    r"""Utility function to load an inhomogeneous
    wavepacket from an :py:class:`IOManager` instance.

    :param iom: The :py:class:`IOManager` instance from which to load data.
    :param timestep: Load the data corresponding to the given `timestep`.
    :param blockid: The `datablock` from which to read the data.
                    Default is the block with `blockid=0`.

    Note: This function is a pure utility function and is not efficient. It is
          built for interactive use only and should not be use in scripts.
    """
    if not iom.has_inhomogwavepacket(blockid=blockid):
        print("There is no (inhomogeneous) wavepacket to load in block "+str(blockid))
        return

    BF = BlockFactory()

    wpd = iom.load_inhomogwavepacket_description(blockid=blockid)
    HAWP = BF.create_wavepacket(wpd)

    # Basis shapes
    BS_descr = iom.load_inhomogwavepacket_basisshapes(blockid=blockid)
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BF.create_basis_shape(descr)

    KEY = ("q","p","Q","P","S","adQ")
    # Retrieve simulation data
    params = iom.load_inhomogwavepacket_parameters(timestep=timestep, blockid=blockid, key=KEY)
    hashes, coeffs = iom.load_inhomogwavepacket_coefficients(timestep=timestep, get_hashes=True, blockid=blockid)
    # Configure the wavepacket
    HAWP.set_parameters(params, key=KEY)
    HAWP.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
    HAWP.set_coefficients(coeffs)

    return HAWP
