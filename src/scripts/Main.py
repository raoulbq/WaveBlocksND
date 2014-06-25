"""The WaveBlocks Project

This file is main script for running simulations with WaveBlocks.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import argparse

from WaveBlocksND import ParameterLoader

parser = argparse.ArgumentParser()

parser.add_argument("parametersfile",
                    type = str,
                    help = "The simulation configuration parameters file")

args = parser.parse_args()

print("Using configuration from file: " + args.parametersfile)

# Set up the parameter provider singleton
PA = ParameterLoader().load_from_file(args.parametersfile)

# Print the parameters that apply for this simulation
print(PA)

# Decide which simulation loop to use
if PA["algorithm"] == "fourier":
    from WaveBlocksND import SimulationLoopFourier
    SL = SimulationLoopFourier(PA)

elif PA["algorithm"] == "hagedorn":
    from WaveBlocksND import SimulationLoopHagedorn
    SL = SimulationLoopHagedorn(PA)

elif PA["algorithm"] == "hagedorn_inhomog":
    from WaveBlocksND import SimulationLoopHagedornInhomogeneous
    SL = SimulationLoopHagedornInhomogeneous(PA)

elif PA["algorithm"] == "fourier_nlse":
    from WaveBlocksND import SimulationLoopFourierNLSE
    SL = SimulationLoopFourierNLSE(PA)

# NOTE: Add new algorithms here

else:
    raise ValueError("Invalid propagator algorithm.")

# Initialize and run the simulation
SL.prepare_simulation()
SL.run_simulation()

# End the simulation, close output files etc.
SL.end_simulation()
