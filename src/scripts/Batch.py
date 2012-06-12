"""The WaveBlocks Project

Simple script to run several simulations with a given set of parameter files.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
import os
from glob import glob
import subprocess as sp

from WaveBlocksND import GlobalDefaults


def batch_run(call_simulation, call_for_each, call_once):
    # Gather cmd line arguments or use defaults if none given
    if len(sys.argv) >= 3:
        configpath = sys.argv[2]
    else:
        configpath = GlobalDefaults.path_to_configs

    if len(sys.argv) >= 4:
        resultpath = sys.argv[3]
    else:
        resultpath = GlobalDefaults.path_to_results

    # The directories involved
    currentdir = os.getcwd()
    configurationsdir = os.path.join(currentdir, configpath)
    resultsdir = os.path.join(currentdir, resultpath)

    print("The working directory for the simulations is: " + currentdir)
    print("The parameter sets are read from files in:    " + configurationsdir)
    print("The results are put into subdirectories of:   " + resultsdir)

    # Check if the cofigurations are available
    if os.path.lexists(configurationsdir) and os.path.isdir(configurationsdir):
        configurations = os.listdir(configurationsdir)
        if len(configurations) == 0:
            raise ValueError("There are no configurations!")
    else:
        raise ValueError("There is no configurations directory!")

    # Check if the destination for the results is prepared
    if not os.path.lexists(resultsdir):
        os.mkdir(resultsdir)
    else:
        print("Warning: The results directory already exists.")
        print("May overwrite older results.")

    # Start the batch loop
    for configuration in configurations:
        print("Current configuration is: " + configuration)

        filepath = os.path.join(configurationsdir, configuration)

        # Call the scripts running the simulation with argument 'filepath'
        for command in  call_simulation:
            sp.call(["python", command, filepath])

        # Call all evaluation scripts need to be run for each simulation
        for command in call_for_each:
            if type(command) == list or type(command) == tuple:
                sp.call(["python"] + list(command))
            else:
                sp.call(["python", command])

        # Clean up and move results
        simulationid = configuration[:-3]
        resultspath = os.path.join(resultsdir, simulationid)

        if not os.path.lexists(resultspath):
            os.mkdir(resultspath)

        sp.call(["cp", filepath, resultspath])
        for afile in glob("*.hdf5"):
            sp.call(["mv", afile, resultspath])
        for afile in glob("*.png"):
            sp.call(["mv", afile, resultspath])
        for afile in glob("*.pdf"):
            sp.call(["mv", afile, resultspath])

    print("Finished batch loop")

    # Postprocessing, call scripts that should get called once after all simulations finished
    for command in call_once:
        if type(command) == list or type(command) == tuple:
            sp.call(["python"] + list(command))
        else:
            sp.call(["python", command])

    print("All simulations finished")


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        batchconfigfile = sys.argv[1]
    else:
        batchconfigfile = GlobalDefaults.file_batchconfiguration

    # Read the batch configuration file
    f = open(batchconfigfile)
    content = f.read()
    f.close()

    # Execute the batchconfiguration file
    # Assuming that it defines the three lists 'call_simulation',
    # 'call_for_each', 'call_once' in the toplevel namespace.
    exec(content)

    # Really start the scripts
    batch_run(call_simulation, call_for_each, call_once)
