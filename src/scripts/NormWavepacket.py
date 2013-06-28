"""The WaveBlocks Project

Compute the norms of the homogeneous wavepackets as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocksND import BlockFactory
from WaveBlocksND import BasisTransformationHAWP


def compute_norm_hawp(iom, blockid=0, eigentrafo=True):
    """Compute the norm of a wavepacket timeseries.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    :type blockid: Integer, Default is ``0``
    :param eigentrafo: Whether to make a transformation into the eigenbasis.
    :type eigentrafo: Boolean, default is ``True``.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavepacket_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Basis transformator
    if eigentrafo is True:
        # The potential used
        Potential = BlockFactory().create_potential(parameters)
        BT = BasisTransformationHAWP(Potential)

    # We want to save norms, thus add a data slot to the data file
    iom.add_norm(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_wavepacket_description(blockid=blockid)
    HAWP = BlockFactory().create_wavepacket(descr)

    if eigentrafo is True:
        BT.set_matrix_builder(HAWP.get_innerproduct())

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes(blockid=blockid)
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    KEY = ("q","p","Q","P","S","adQ")

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing norms of timestep "+str(step))

        # Retrieve simulation data
        params = iom.load_wavepacket_parameters(timestep=step, blockid=blockid, key=KEY)
        hashes, coeffs = iom.load_wavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWP.set_parameters(params, key=KEY)
        HAWP.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
        HAWP.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        if eigentrafo is True:
            BT.transform_to_eigen(HAWP)

        # Measure norms in the eigenbasis
        norm = HAWP.norm()

        # Save the norms
        iom.save_norm(norm, timestep=step, blockid=blockid)


def compute_norm_inhawp(iom, blockid=0, eigentrafo=True):
    """Compute the norm of a wavepacket timeseries.
    This function is for inhomogeneous wavepackets.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    :type blockid: Integer, Default is ``0``
    :param eigentrafo: Whether to make a transformation into the eigenbasis.
    :type eigentrafo: Boolean, default is ``True``.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Basis transformator
    if eigentrafo is True:
        # The potential used
        Potential = BlockFactory().create_potential(parameters)
        BT = BasisTransformationHAWP(Potential)

    # We want to save norms, thus add a data slot to the data file
    iom.add_norm(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_inhomogwavepacket_description(blockid=blockid)
    HAWP = BlockFactory().create_wavepacket(descr)

    if eigentrafo is True:
        BT.set_matrix_builder(HAWP.get_innerproduct())

    # Basis shapes
    BS_descr = iom.load_inhomogwavepacket_basisshapes(blockid=blockid)
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    KEY = ("q","p","Q","P","S","adQ")

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing norms of timestep "+str(step))

        # Retrieve simulation data
        params = iom.load_inhomogwavepacket_parameters(timestep=step, blockid=blockid, key=KEY)
        hashes, coeffs = iom.load_inhomogwavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWP.set_parameters(params, key=KEY)
        HAWP.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
        HAWP.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        if eigentrafo is True:
            BT.transform_to_eigen(HAWP)

        # Measure norms in the eigenbasis
        norm = HAWP.norm()

        # Save the norms
        iom.save_norm(norm, timestep=step, blockid=blockid)
