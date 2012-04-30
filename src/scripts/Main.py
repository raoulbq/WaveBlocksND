"""The WaveBlocks Project

This file is main script for running simulations with WaveBlocks.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocksND import ParameterLoader


# Read the path for the configuration file we use for this simulation.
try:
    parametersfile = sys.argv[1]
except IndexError:
    raise ValueError("No configuration file given!")

print("Using configuration from file: " + parametersfile)

# Set up the parameter provider singleton
PA = ParameterLoader().load_from_file(parametersfile)

# Print the parameters that apply for this simulation
print(PA)

# Decide which simulation loop to use
if PA["algorithm"] == "fourier":
    from WaveBlocksND import SimulationLoopFourier
    SL = SimulationLoopFourier(PA)

elif PA["algorithm"] == "hagedorn":
    from WaveBlocksND import SimulationLoopHagedorn
    SL = SimulationLoopHagedorn(PA)

# TODO: Add new algorithms here

else:
    raise ValueError("Invalid propagator algorithm.")

# Initialize and run the simulation
SL.prepare_simulation()
SL.run_simulation()

# End the simulation, close output files etc.
SL.end_simulation()
