"""The WaveBlocks Project

Calculate the norms of the different wavefunctions as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocksND import BlockFactory
from WaveBlocksND import WaveFunction
from WaveBlocksND import BasisTransformationWF


def compute_norm(iom, blockid=0, eigentrafo=True):
    """Compute the norm of a wavefunction timeseries.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read. Default is `0`.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavefunction_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Construct the grid from the parameters
    grid = BlockFactory().create_grid(parameters)

    # The potential used
    Potential = BlockFactory().create_potential(parameters)

    # Basis transformator
    if eigentrafo is True:
        BT = BasisTransformationWF(Potential)
        BT.set_grid(grid)

    # And two empty wavefunctions
    WF = WaveFunction(parameters)
    WF.set_grid(grid)

    # We want to save norms, thus add a data slot to the data file
    iom.add_norm(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing norms of timestep "+str(step))

        # Retrieve simulation data
        values = iom.load_wavefunction(timestep=step, blockid=blockid)
        values = [ values[j,...] for j in xrange(parameters["ncomponents"]) ]
        WF.set_values(values)

        # Project wavefunction values to eigenbasis
        if eigentrafo is True:
            BT.transform_to_eigen(WF)

        # Calculate the norm of the wave functions projected into the eigenbasis
        norms = WF.norm()

        iom.save_norm(norms, timestep=step, blockid=blockid)
