"""The WaveBlocks Project

Compute the autocorrelations of Hagedorn wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocksND import BlockFactory
from WaveBlocksND import BasisTransformationHAWP


def compute_autocorrelation_hawp(iom, obsconfig, blockid=0, eigentrafo=True):
    """Compute the autocorrelation of a wavepacket timeseries.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param obsconfig: Configuration parameters describing f.e. the inner product to use.
    :type obsconfig: A :py:class:`ParameterProvider` instance.
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
    iom.add_autocorrelation(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_wavepacket_description(blockid=blockid)
    HAWPo = BlockFactory().create_wavepacket(descr)
    HAWPt = BlockFactory().create_wavepacket(descr)

    if eigentrafo is True:
        BT.set_matrix_builder(HAWPo.get_innerproduct())

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes()
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    # Comfigure the original wavepacket
    KEY = ("q","p","Q","P","S","adQ")
    # Retrieve simulation data
    params = iom.load_wavepacket_parameters(timestep=0, blockid=blockid, key=KEY)
    hashes, coeffs = iom.load_wavepacket_coefficients(timestep=0, get_hashes=True, blockid=blockid)
    # Configure the wavepacket
    HAWPo.set_parameters(params, key=KEY)
    HAWPo.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
    HAWPo.set_coefficients(coeffs)

    # Set up the innerproduct for solving the integrals <phi_0 | phi_t>
    IP = BlockFactory().create_inner_product(obsconfig["innerproduct"])

    # Transform to the eigenbasis.
    if eigentrafo is True:
        BT.transform_to_eigen(HAWPo)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing autocorrelation of timestep "+str(step))

        # Retrieve simulation data
        paramst = iom.load_wavepacket_parameters(timestep=step, blockid=blockid, key=KEY)
        hashes, coeffs = iom.load_wavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWPt.set_parameters(paramst, key=KEY)
        HAWPt.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
        HAWPt.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        if eigentrafo is True:
            BT.transform_to_eigen(HAWPt)

        # Measure autocorrelations in the eigenbasis
        acs = IP.quadrature(HAWPo, HAWPt, diagonal=True)

        # Save the autocorrelations
        iom.save_autocorrelation(acs, timestep=step, blockid=blockid)


def compute_autocorrelation_inhawp(iom, obsconfig, blockid=0, eigentrafo=True):
    """Compute the autocorrelation of a wavepacket timeseries.
    This function is for inhomogeneous wavepackets.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param obsconfig: Configuration parameters describing f.e. the inner product to use.
    :type obsconfig: A :py:class:`ParameterProvider` instance.
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

    # We want to save autocorrelations, thus add a data slot to the data file
    iom.add_autocorrelation(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_inhomogwavepacket_description(blockid=blockid)
    HAWPo = BlockFactory().create_wavepacket(descr)
    HAWPt = BlockFactory().create_wavepacket(descr)

    if eigentrafo is True:
        BT.set_matrix_builder(HAWPo.get_innerproduct())

    # Basis shapes
    BS_descr = iom.load_inhomogwavepacket_basisshapes()
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    # Comfigure the original wavepacket
    # Retrieve simulation data
    params = iom.load_wavepacket_parameters(timestep=0, blockid=blockid)
    hashes, coeffs = iom.load_wavepacket_coefficients(timestep=0, get_hashes=True, blockid=blockid)
    # Configure the wavepacket
    HAWPo.set_parameters(params)
    HAWPo.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
    HAWPo.set_coefficients(coeffs)

    # Set up the innerproduct for solving the integrals <phi_0 | phi_t>
    IP = BlockFactory().create_inner_product(obsconfig["innerproduct"])

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing autocorrelations of timestep "+str(step))

        # Retrieve simulation data
        params = iom.load_inhomogwavepacket_parameters(timestep=step, blockid=blockid)
        hashes, coeffs = iom.load_inhomogwavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWPt.set_parameters(params)
        HAWPt.set_basis_shapes([ BS[int(ha)] for ha in hashes ])
        HAWPt.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        if eigentrafo is True:
            BT.transform_to_eigen(HAWPt)

        # Measure autocorrelations in the eigenbasis
        acs = IP.quadrature(HAWPo, HAWPt, diagonal=True)

        # Save the autocorrelations
        iom.save_autocorrelation(acs, timestep=step, blockid=blockid)
