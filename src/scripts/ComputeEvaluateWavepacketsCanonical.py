"""The WaveBlocks Project

Sample wavepackets at the nodes of a given grid and save
the results back to the given simulation data file.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import argparse

from WaveBlocksND import IOManager
from WaveBlocksND import ParameterLoader
from WaveBlocksND import GlobalDefaults as GD

parser = argparse.ArgumentParser()

parser.add_argument("simfile",
                    type = str,
                    help = "The simulation data file",
                    nargs = "?",
                    default = GD.file_resultdatafile)

parser.add_argument("-b", "--blockid",
                    help = "The data block to handle",
                    nargs = "*",
                    default = [0])

parser.add_argument("-p", "--params",
                    help = "An additional configuration parameters file")

args = parser.parse_args()

# Read file with simulation data
iom = IOManager()
iom.open_file(filename=args.simfile)

# Read the additional grid parameters
if args.params:
    parametersfile = args.params
    PA = ParameterLoader().load_from_file(parametersfile)
else:
    PA = None

# Which blocks to handle
if "all" in args.blockid:
    blocks_to_handle = iom.get_block_ids()
else:
    blocks_to_handle = map(int, args.blockid)

# Iterate over all blocks
for blockid in blocks_to_handle:
    print("Evaluating wavepackets in data block '"+str(blockid)+"'")

    if iom.has_wavefunction(blockid=blockid):
        print("Datablock '"+str(blockid)+"' already contains wavefunction data, silent skip.")
        continue

    # NOTE: Add new algorithms here

    if iom.has_wavepacket(blockid=blockid):
        import EvaluateWavepackets
        EvaluateWavepackets.compute_evaluate_wavepackets(PA, iom, blockid=blockid, eigentrafo=False)
    elif iom.has_inhomogwavepacket(blockid=blockid):
        import EvaluateWavepacketsInhomog
        EvaluateWavepacketsInhomog.compute_evaluate_wavepackets(PA, iom, blockid=blockid, eigentrafo=False)
    else:
        print("Warning: Not evaluating any wavepackets in block '"+str(blockid)+"'!")

iom.finalize()
