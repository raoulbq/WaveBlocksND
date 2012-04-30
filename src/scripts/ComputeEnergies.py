"""The WaveBlocks Project

Compute the energies of the different wavepackets or wavefunctions.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocksND import IOManager


if __name__ == "__main__":

    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Iterate over all blocks
    for blockid in iom.get_block_ids():
        print("Computing the energies in data block '"+str(blockid)+"'")

        if iom.has_energy(blockid=blockid):
            print("Datablock '"+str(blockid)+"' already contains energy data, silent skip.")
            continue

        # TODO: Add new algorithms here

        # If not, we test for a homogeneous wavepacket next
        if iom.has_wavepacket(blockid=blockid):
            import EnergiesWavepacket
            EnergiesWavepacket.compute_energy(iom, blockid=blockid)
        # If we have no wavepacket, then we try for a wavefunction
        elif iom.has_wavefunction(blockid=blockid):
            import EnergiesWavefunction
            EnergiesWavefunction.compute_energy(iom, blockid=blockid)
            # If there is also no wavefunction, then there is nothing to compute the energies
        else:
            print("Warning: Not computing any energies in block '"+str(blockid)+"'!")

    iom.finalize()
