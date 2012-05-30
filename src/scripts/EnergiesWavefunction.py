"""The WaveBlocks Project

Compute the kinetic and potential energies of a wavefunction.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros

from WaveBlocksND import PotentialFactory
from WaveBlocksND import GridFactory
from WaveBlocksND import KineticOperator
from WaveBlocksND import WaveFunction
from WaveBlocksND import BasisTransformationWF


def compute_energy(iom, blockid=0):
    """
    :param iom: An :py:class:`IOManager: instance providing the simulation data.
    :param blockid: The data block from which the values are read. Default is `0`.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavefunction_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Try to load the grid, otherwise construct one from the parameters
    # TODO: We can only do this iff we are able to construct a grid from the data
    # if iom.has_grid(blockid=blockid):
    #     grid = iom.load_grid(blockid=blockid)
    # elif  iom.has_grid(blockid="global"):
    #     grid = iom.load_grid(blockid="global")
    # else:
    grid = GridFactory().create_grid(parameters)

    # The potential used
    Potential = PotentialFactory().create_potential(parameters)

    # Try to load the operators, otherwise reconstruct them
    # TODO: We can only do this iff we are able to construct operators from the data
    # if iom.has_fourieroperators(blockid=blockid):
    #     opT, opV = iom.load_fourieroperators(blockid=blockid)
    # else:
    KO = KineticOperator(grid)
    KO.calculate_operator(parameters["eps"])
    opT = KO
    opV = Potential.evaluate_at(grid)

    # Basis transformator
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
