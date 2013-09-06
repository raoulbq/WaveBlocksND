"""The WaveBlocks Project

Compute the autocorrelations of the different wavepackets or wavefunctions.
This script does not do an eigentransformation of the data.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
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

# TODO: Filter type of objects
# parser.add_argument("-t", "--type",
#                     help = "The type of objects to consider",
#                     type = str,
#                     default = "all")

args = parser.parse_args()

# Read file with simulation data
iom = IOManager()
iom.open_file(filename=args.simfile)

# Which blocks to handle
if "all" in args.blockid:
    blocks_to_handle = iom.get_block_ids()
else:
    blocks_to_handle = map(int, args.blockid)

# Do we have a specifc configuration file holding
# the definitions for inner products to use?
if args.params:
    parametersfile = args.params
    PA = ParameterLoader().load_from_file(parametersfile)
else:
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

# No configuration parameters so far, use a more or less sane default
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
for blockid in blocks_to_handle:
    print("Computing the autocorrelation in data block '"+str(blockid)+"'")

    if iom.has_autocorrelation(blockid=blockid):
        print("Datablock '"+str(blockid)+"' already contains autocorrelation data, silent skip.")
        continue

    # NOTE: Add new algorithms here

    if iom.has_wavepacket(blockid=blockid):
        import AutocorrelationWavepacket
        AutocorrelationWavepacket.compute_autocorrelation_hawp(iom, PA, blockid=blockid, eigentrafo=False)
    elif iom.has_wavefunction(blockid=blockid):
        import AutocorrelationWavefunction
        AutocorrelationWavefunction.compute_autocorrelation(iom, PA, blockid=blockid, eigentrafo=False)
    elif iom.has_inhomogwavepacket(blockid=blockid):
        import AutocorrelationWavepacket
        AutocorrelationWavepacket.compute_autocorrelation_inhawp(iom, PA, blockid=blockid, eigentrafo=False)
    else:
        print("Warning: Not computing any autocorrelations in block '"+str(blockid)+"'!")

iom.finalize()
