"""The WaveBlocks Project

Sample wavepackets at the nodes of a given grid and save
the results back to the given simulation data file.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocksND import IOManager
from WaveBlocksND import ParameterLoader


if __name__ == "__main__":

    iom = IOManager()
    PL = ParameterLoader()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Read file with parameter data for grid
    try:
        PP = PL.load_from_file(sys.argv[2])
    except IndexError:
        raise ValueError("No parameter file given")

    # Iterate over all blocks
    for blockid in iom.get_block_ids():
        print("Evaluating wavepackets in data block '"+str(blockid)+"'")

        if iom.has_wavefunction(blockid=blockid):
            print("Datablock '"+str(blockid)+"' already contains wavefunction data, silent skip.")
            continue

        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(blockid=blockid):
            import EvaluateWavepacketsInhomog
            EvaluateWavepacketsInhomog.compute_evaluate_wavepackets(PP, iom, blockid=blockid, eigentrafo=False)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(blockid=blockid):
            import EvaluateWavepackets
            EvaluateWavepackets.compute_evaluate_wavepackets(PP, iom, blockid=blockid, eigentrafo=True)
        # If there is also no wavefunction, then there is nothing to compute the norm
        else:
            print("Warning: Not evaluating any wavepackets in block '"+str(blockid)+"'!")

    iom.finalize()
