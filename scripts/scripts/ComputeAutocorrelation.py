"""The WaveBlocks Project

Compute the autocorrelations of the different wavepackets or wavefunctions.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013, 2014 R. Bourquin
@license: Modified BSD License
"""

import argparse

from WaveBlocksND import IOManager
from WaveBlocksND import ParameterLoader
from WaveBlocksND import GlobalDefaults as GD

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datafile",
                    type = str,
                    help = "The simulation data file",
                    nargs = "?",
                    default = GD.file_resultdatafile)

parser.add_argument("-b", "--blockid",
                    type = str,
                    help = "The data block to handle",
                    nargs = "*",
                    default = ["all"])

parser.add_argument("-p", "--params",
                    help = "An additional configuration parameters file")

parser.add_argument("-et", "--eigentransform",
                    help = "Transform the data into the eigenbasis before computing norms",
                    action = "store_false")

# TODO: Filter type of objects
# parser.add_argument("-t", "--type",
#                     help = "The type of objects to consider",
#                     type = str,
#                     default = "all")

args = parser.parse_args()

# Read file with simulation data
iom = IOManager()
iom.open_file(filename=args.datafile)

# Which blocks to handle
blockids = iom.get_block_ids()
if not "all" in args.blockid:
    blockids = [ bid for bid in args.blockid if bid in blockids ]

# Do we have a specifc configuration file holding
# the definitions for inner products to use?
if args.params:
    parametersfile = args.params
    PA = ParameterLoader().load_from_file(parametersfile)
else:
    # None given, try to load from simulation file
    if iom.has_parameters():
        PA = iom.load_parameters()
    else:
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
    descr = iom.load_wavepacket_description(blockid=blockids[0])
    D = descr["dimension"]
    PA = {}
    PA["innerproduct"] = {
        "type" : "InhomogeneousInnerProduct",
        "delegate" : {
            "type" : "NSDInhomogeneous",
            'qr': {
                'type': 'TensorProductQR',
                'dimension': D,
                'qr_rules': D * [{'type': 'GaussHermiteOriginalQR', 'order': 5}]
                }
            }
        }

# Iterate over all blocks
for blockid in blockids:
    print("Computing the autocorrelation in data block '%s'" % blockid)

    if iom.has_autocorrelation(blockid=blockid):
        print("Datablock '%s' already contains autocorrelation data, silent skip." % blockid)
        continue

    # NOTE: Add new algorithms here

    if iom.has_wavepacket(blockid=blockid):
        import AutocorrelationWavepacket
        AutocorrelationWavepacket.compute_autocorrelation_hawp(iom, PA, blockid=blockid, eigentrafo=args.eigentransform)
    elif iom.has_wavefunction(blockid=blockid):
        import AutocorrelationWavefunction
        AutocorrelationWavefunction.compute_autocorrelation(iom, PA, blockid=blockid, eigentrafo=args.eigentransform)
    elif iom.has_inhomogwavepacket(blockid=blockid):
        import AutocorrelationWavepacket
        AutocorrelationWavepacket.compute_autocorrelation_inhawp(iom, PA, blockid=blockid, eigentrafo=args.eigentransform)
    else:
        print("Warning: Not computing any autocorrelations in block '%s'!" % blockid)

iom.finalize()