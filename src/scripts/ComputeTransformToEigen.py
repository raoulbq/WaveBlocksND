"""The WaveBlocks Project

Compute the eigen transformation of some simulation results.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocksND import IOManager
from WaveBlocksND import GlobalDefaults as GD


if __name__ == "__main__":

    iomc = IOManager()
    iome = IOManager()

    # Read file with simulation data
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = GD.file_resultdatafile

    iomc.open_file(filename=filename)

    # New file for eigen transformed data
    P = iomc.load_parameters()
    iome.create_file(P, filename=filename[:-5]+"_eigen.hdf5")

    # Iterate over all groups
    for groupid in iomc.get_group_ids():

        # Create the group if necessary
        if not groupid in iome.get_group_ids():
            iome.create_group(groupid=groupid)

        for blockid in iomc.get_block_ids(groupid=groupid):
            print("Computing eigentransformation of data in block '"+str(blockid)+"'")

            # Create the block if necessary
            if not blockid in iome.get_block_ids(groupid=groupid):
                iome.create_block(blockid=blockid, groupid=groupid)

            # See if we have a wavefunction
            if iomc.has_wavefunction(blockid=blockid):
                from EigentransformWavefunction import transform_wavefunction_to_eigen
                transform_wavefunction_to_eigen(iomc, iome, blockidin=blockid, blockidout=blockid)

            # See if we have a homogeneous wavepacket next
            if iomc.has_wavepacket(blockid=blockid):
                from EigentransformHagedornWavepacket import transform_hawp_to_eigen
                transform_hawp_to_eigen(iomc, iome, blockidin=blockid, blockidout=blockid)

            # See if we have an inhomogeneous wavepacket next
            if iomc.has_inhomogwavepacket(blockid=blockid):
                from EigentransformHagedornWavepacket import transform_hawpih_to_eigen
                transform_inhawp_to_eigen(iomc, iome, blockidin=blockid, blockidout=blockid)

    iomc.finalize()
    iome.finalize()
