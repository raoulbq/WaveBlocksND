"""The WaveBlocks Project

Sample wavepackets at the nodes of a given grid and save the results back
to the given simulation data file.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocksND import BlockFactory
from WaveBlocksND import HagedornWavepacket
from WaveBlocksND import WaveFunction


def compute_evaluate_wavepackets(pp, iom, blockid=0, eigentrafo=True):
    """Evaluate a homogeneous Hagedorn wavepacket on a given grid for each timestep.

    :param pp: An :py:class:`ParameterProvider` instance providing the grid data.
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    :param eigentrafo: Whether or not do an eigentransformation before evaluation is done.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavepacket_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Prepare the potential for basis transformations
    Potential = BlockFactory().create_potential(parameters)
    grid = BlockFactory().create_grid(pp)

    # We want to save wavefunctions, thus add a data slot to the data file
    d = {"ncomponents":parameters["ncomponents"],
         "number_nodes":pp["number_nodes"],
         "dimension":parameters["dimension"]}
    iom.add_grid(d, blockid=blockid)
    iom.add_wavefunction(d, timeslots=nrtimesteps, flat=True, blockid=blockid)

    iom.save_grid(grid.get_nodes(), blockid=blockid)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_wavepacket_description(blockid=blockid)
    HAWP = BlockFactory().create_wavepacket(descr)

    # Basis transformator
    if eigentrafo is True:
        BT = BasisTransformationHAWP(Potential)
        BT.set_matrix_builder(HAWP.get_quadrature())

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes()
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    WF = WaveFunction(parameters)
    WF.set_grid(grid)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Evaluating homogeneous wavepacket at timestep "+str(step))

        # Retrieve simulation data
        params = iom.load_wavepacket_parameters(timestep=step, blockid=blockid)
        hashes, coeffs = iom.load_wavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWP.set_parameters(params)
        HAWP.set_basis_shape([ BS[int(ha)] for ha in hashes ])
        HAWP.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        if eigentrafo is True:
            BT.transform_to_eigen(HAWP)

        # Evaluate the wavepacket
        values = HAWP.evaluate_at(grid, prefactor=True)
        WF.set_values(values)

        # Save the wave function
        iom.save_wavefunction(WF.get_values(), timestep=step, blockid=blockid)