#!/usr/bin/env python
"""The WaveBlocks Project

Compute the autocorrelations of the different wavepackets or wavefunctions.
This script does not do an eigentransformation of the data.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013, 2014, 2016 R. Bourquin
@license: Modified BSD License
"""

import argparse
import os

from WaveBlocksND import IOManager
from WaveBlocksND import ParameterLoader
from WaveBlocksND import GlobalDefaults as GD

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datafile",
                    type = str,
                    help = "The simulation data file.",
                    nargs = "?",
                    default = GD.file_resultdatafile)

parser.add_argument("-b", "--blockid",
                    type = str,
                    help = "The data block to handle.",
                    nargs = "*",
                    default = ["all"])

parser.add_argument("-p", "--parametersfile",
                    type = str,
                    help = "An additional configuration parameters file.",
                    nargs = "?",
                    default = None)

parser.add_argument("-r", "--resultspath",
                    type = str,
                    help = "Path where to put the results.",
                    nargs = "?",
                    default = '.')

parser.add_argument("-et", "--eigentransform",
                    help = "Transform the data into the eigenbasis before computing norms.",
                    action = "store_true")

# TODO: Filter type of objects
# parser.add_argument("-t", "--type",
#                     help = "The type of objects to consider.",
#                     type = str,
#                     default = "all")

args = parser.parse_args()


# File with the simulation data
resultspath = os.path.abspath(args.resultspath)

if not os.path.exists(resultspath):
    raise IOError("The results path does not exist: " + args.resultspath)

datafile = os.path.abspath(os.path.join(args.resultspath, args.datafile))

# Read file with simulation data
iom = IOManager()
iom.open_file(filename=datafile)

# Which blocks to handle
blockids = iom.get_block_ids()
if "all" not in args.blockid:
    blockids = [bid for bid in args.blockid if bid in blockids]

# Do we have a specifc configuration file holding
# the definitions for inner products to use?
if args.parametersfile:
    parametersfile = os.path.abspath(os.path.join(args.resultspath, args.parametersfile))
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
    if "observables" in PA:
        PA = PA["observables"]
        if "autocorrelation" in PA:
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
        "type": "InhomogeneousInnerProduct",
        "delegate": {
            "type": "NSDInhomogeneous",
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
        from WaveBlocksND.Interface import AutocorrelationWavepacket
        AutocorrelationWavepacket.compute_autocorrelation_hawp(iom, PA, blockid=blockid, eigentrafo=args.eigentransform)
    elif iom.has_wavefunction(blockid=blockid):
        from WaveBlocksND.Interface import AutocorrelationWavefunction
        AutocorrelationWavefunction.compute_autocorrelation(iom, PA, blockid=blockid, eigentrafo=args.eigentransform)
    elif iom.has_inhomogwavepacket(blockid=blockid):
        from WaveBlocksND.Interface import AutocorrelationWavepacket
        AutocorrelationWavepacket.compute_autocorrelation_inhawp(iom, PA, blockid=blockid, eigentrafo=args.eigentransform)
    else:
        print("Warning: Not computing any autocorrelations in block '%s'!" % blockid)

iom.finalize()