"""The WaveBlocks Project

Compute the autocorrelations of the different wavepackets or wavefunctions.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocksND import IOManager
from WaveBlocksND import ParameterLoader


if __name__ == "__main__":

    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Do we have a specifc configuration file holding
    # the definitions for inner products to use?
    try:
        parametersfile = sys.argv[2]
        PA = ParameterLoader().load_from_file(parametersfile)

    except IndexError:
        # None given, try to load from simulation file
        try:
            PA = iom.load_parameters()
        except:
            PA = None

    # See if we have a description for observables and
    # especially autocorrelation computation
    if PA is not None:
        if PA.has_key("observables"):
            PA = PA["observables"]
            if PA.has_key("autocorrelation"):
                PA = PA["autocorrelation"]
            else:
                PA = None
        else:
            PA = None

    if PA is None:
        print("Warning: Using (possibly improper) default values for inner product")
        PA = {}
        PA["innerproduct"] = {
            "type" : "InhomogeneousInnerProduct",
            "delegate" : {
                "type" : "NSDInhomogeneous",
                'qr': {
                    'type': 'GaussLaguerreQR',
                    'order': 5,
                    'a': -0.5
                    }
                }
            }

    # Iterate over all blocks
    for blockid in iom.get_block_ids():
        print("Computing the autocorrelation in data block '"+str(blockid)+"'")

        if iom.has_autocorrelation(blockid=blockid):
            print("Datablock '"+str(blockid)+"' already contains autocorrelation data, silent skip.")
            continue

        # TODO: Add new algorithms here

        # We test for an inhomogeneous wavepacket next
        if iom.has_inhomogwavepacket(blockid=blockid):
            import AutocorrelationWavepacket
            AutocorrelationWavepacket.compute_autocorrelation_inhawp(iom, PA, blockid=blockid)
        # We test for a homogeneous wavepacket next
        elif iom.has_wavepacket(blockid=blockid):
            import AutocorrelationWavepacket
            AutocorrelationWavepacket.compute_autocorrelation_hawp(iom, PA, blockid=blockid)
        # We have no wavepacket, then we try for a wavefunction
        elif iom.has_wavefunction(blockid=blockid):
            import AutocorrelationWavefunction
            AutocorrelationWavefunction.compute_autocorrelation(iom, PA, blockid=blockid)
        # If there is also no wavefunction, then there is nothing to compute the autocorrelation
        else:
            print("Warning: Not computing any autocorrelations in block '"+str(blockid)+"'!")

    iom.finalize()
