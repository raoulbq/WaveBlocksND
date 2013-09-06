"""The WaveBlocks Project

This file contains code some simple code to call a given
python script for a bunch of simulation result files.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import argparse
from glob import glob
import subprocess as sp
import time

from WaveBlocksND.FileTools import get_result_dirs, get_results_file
from WaveBlocksND import GlobalDefaults

try:
    from GraphicsDefaults import output_format
except:
    output_format = ".png"


def execute_for_all(resultspath, scriptcode, scriptargs):
    """Call a given python script with the simulation results data file as first
    command line argument. The script and the data file are specified by (relative)
    file system paths.

    :param resultspath: The path where to look for simulation data.
    :param scriptcode: The python script that gets called for all simulations.
    :param scriptargs: Optional (constant) arguments to the script.
    """
    # UNBUFFERED timelog file
    with open("logtime_forall", "w", 0) as timelog:

        for simulationpath in get_result_dirs(resultspath):
            starttime = time.time()
            timelog.writelines(["New script starting at: " + time.ctime(starttime) + "\n",
                                "Executing code for datafile in " + simulationpath + "\n"])
            print(" Executing code for datafile in " + simulationpath)

            # The file with the simulation data
            afile = get_results_file(simulationpath)

            # Call the given script
            sp.call(["python", scriptcode, afile] + scriptargs)

            endtime = time.time()
            timelog.writelines(["Script now finished at timestamp: " + time.ctime(endtime) + "\n",
                                "Script took " + str(endtime-starttime) + " second to run\n\n"])

            # Move plots away if any
            for afile in glob("*"+output_format):
                sp.call(["mv", afile, simulationpath])




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("scriptcode",
                        type = str,
                        help = "The script file to execute.")

    parser.add_argument("-r", "--results",
                        type = str,
                        help = "The script file to execute.",
                        default = GlobalDefaults.path_to_results)

    parser.add_argument("scriptargs",
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print("Will execute the code from '" + str(args.scriptcode) + "' for all files in '" + str(args.results) + "'")
    print("with optional arguments " + str(args.scriptargs))

    execute_for_all(args.results, args.scriptcode, args.scriptargs)

    print("Done")
