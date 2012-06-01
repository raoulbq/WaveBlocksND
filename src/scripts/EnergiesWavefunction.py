"""The WaveBlocks Project

Compute the kinetic and potential energies of a wavefunction.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros

from WaveBlocksND import BlockFactory
from WaveBlocksND import KineticOperator
from WaveBlocksND import WaveFunction
from WaveBlocksND import BasisTransformationWF


def compute_energy(iom, blockid=0, eigentrafo=True):
    """
    :param iom: An :py:class:`IOManager: instance providing the simulation data.
    :param blockid: The data block from which the values are read. Default is `0`.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavefunction_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Construct grid from the parameters
    grid = BlockFactory().create_grid(parameters)

    # The potential used
    Potential = BlockFactory().create_potential(parameters)

    # The operators
    KO = KineticOperator(grid)
    KO.calculate_operator(parameters["eps"])
    opT = KO
    if eigentrafo is True:
        opV = Potential.evaluate_at(grid)
    else:
        opV = Potential.evaluate_eigenvalues_at(grid, as_matrix=True)

    # Basis transformator
    if eigentrafo is True:
        BT = BasisTransformationWF(Potential)
        BT.set_grid(grid)

    # And two empty wavefunctions
    WF = WaveFunction(parameters)
    WF.set_grid(grid)
    WF2 = WaveFunction(parameters)
    WF2.set_grid(grid)

    # We want to save norms, thus add a data slot to the data file
    iom.add_energy(parameters, timeslots=nrtimesteps, blockid=blockid)

    nst = Potential.get_number_components()

    if eigentrafo is True:

        # Iterate over all timesteps
        for i, step in enumerate(timesteps):
            print(" Computing energies of timestep # " + str(step))

            # Retrieve simulation data
            values = iom.load_wavefunction(timestep=step, blockid=blockid)
            values = [ values[j,...] for j in xrange(parameters["ncomponents"]) ]
            WF.set_values(values)

            # Project wavefunction values to eigenbasis
            BT.transform_to_eigen(WF)

            ekinlist = []
            epotlist = []

            # For each component of |Psi>
            values = WF.get_values()

            for index, item in enumerate(values):
                # tmp is the Vector (0, 0, 0, \psi_i, 0, 0, ...)
                tmp = [ zeros(item.shape) for z in xrange(nst) ]
                tmp[index] = item
                WF2.set_values(tmp)

                # Project this vector to the canonical basis
                BT.transform_to_canonical(WF2)

                # And calculate the energies of these components
                ekinlist.append(WF2.kinetic_energy(opT, summed=True))
                epotlist.append(WF2.potential_energy(opV, summed=True))

            iom.save_energy((ekinlist, epotlist), timestep=step, blockid=blockid)

    else:

        # Iterate over all timesteps
        for i, step in enumerate(timesteps):
            print(" Computing energies of timestep # " + str(step))

            # Retrieve simulation data
            values = iom.load_wavefunction(timestep=step, blockid=blockid)
            values = [ values[j,...] for j in xrange(parameters["ncomponents"]) ]
            WF.set_values(values)

            # And calculate the energies of these components
            ekinlist = WF.kinetic_energy(opT, summed=False)
            epotlist = WF.potential_energy(opV, summed=False)

            iom.save_energy((ekinlist, epotlist), timestep=step, blockid=blockid)
