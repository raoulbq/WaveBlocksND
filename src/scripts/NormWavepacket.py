"""The WaveBlocks Project

Compute the norms of the homogeneous wavepackets as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import unique

from WaveBlocksND import PotentialFactory
from WaveBlocksND import BlockFactory
from WaveBlocksND import BasisTransformationHAWP


def compute_norm(iom, blockid=0):
    """Compute the norm of a wavepacket timeseries.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    :type blockid: Integer, Default is ``0``
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavepacket_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # The potential used
    Potential = PotentialFactory().create_potential(parameters)

    # Basis transformator
    BT = BasisTransformationHAWP(Potential)

    # We want to save norms, thus add a data slot to the data file
    iom.add_norm(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_wavepacket_description(blockid=blockid)
    HAWP = BlockFactory().create_wavepacket(descr)

    BT.set_matrix_builder(HAWP.get_quadrature())

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes()
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing norms of timestep "+str(step))

        # Retrieve simulation data
        params = iom.load_wavepacket_parameters(timestep=step, blockid=blockid)
        hashes, coeffs = iom.load_wavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWP.set_parameters(params)
        HAWP.set_basis_shape([ BS[int(ha)] for ha in hashes ])
        HAWP.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        BT.transform_to_eigen(HAWP)

        # Measure norms in the eigenbasis
        norm = HAWP.norm()

        # Save the norms
        iom.save_norm(norm, timestep=step, blockid=blockid)
