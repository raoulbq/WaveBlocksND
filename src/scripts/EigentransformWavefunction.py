"""The WaveBlocks Project

Compute the transformation to the eigen basis for wavefunction.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros

from WaveBlocksND import BlockFactory
from WaveBlocksND import WaveFunction
from WaveBlocksND import BasisTransformationWF


def transform_wavefunction_to_eigen(iomin, iomout, blockidin=0, blockidout=0):
    """Compute the transformation to the eigenbasis for a wavefunction.
    Save the result back to a file.

    :param iomin: An :py:class:`IOManager: instance providing the simulation data.
    :param iomout: An :py:class:`IOManager: instance for saving the transformed data.
    :param blockidin: The data block from which the values are read. Default is `0`.
    :param blockidout: The data block to which the values are written. Default is `0`.
    """
    parameters = iomin.load_parameters()

    # Number of time steps we saved
    timesteps = iomin.load_wavefunction_timegrid(blockid=blockidin)
    nrtimesteps = timesteps.shape[0]

    iomout.add_wavefunction(parameters, timeslots=nrtimesteps, blockid=blockidout)

    # The grid on the domain
    grid = BlockFactory().create_grid(parameters)

    # The potential used
    Potential = BlockFactory().create_potential(parameters)

    # Basis transformator
    BT = BasisTransformationWF(Potential)
    BT.set_grid(grid)

    # And two empty wavefunctions
    WF = WaveFunction(parameters)
    WF.set_grid(grid)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Compute eigentransform at timestep # " + str(step))

        # Retrieve simulation data
        values = iomin.load_wavefunction(timestep=step, blockid=blockidin)
        values = [ values[j,...] for j in xrange(parameters["ncomponents"]) ]
        WF.set_values(values)

        # Project wavefunction values to eigenbasis
        BT.transform_to_eigen(WF)

        # Save the transformed values
        iomout.save_wavefunction(WF.get_values(), timestep=step, blockid=blockidout)
