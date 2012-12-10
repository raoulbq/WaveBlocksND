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
import time

from WaveBlocksND import GlobalDefaults


def batch_run(call_prerun, call_simulation, call_for_each, call_postrun, result_files):
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
        allfiles = os.listdir(configurationsdir)
        # Only python files can be valid configurations
        configurations = filter(lambda x: x.endswith(".py"), allfiles)


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

    # UNBUFFERED timelog file
    timelog = open("logtime", "w", 0)


    # Preprocessing
    # Call scripts that should get called once before any simulations start
    for command in call_prerun:
        if type(command) == list or type(command) == tuple:
            sp.call(["python"] + list(command))
        else:
            sp.call(["python", command])


    # Start the main batch loop
    for configuration in configurations:
        starttime = time.time()
        timelog.writelines(["New simulation at: " + time.ctime(starttime) + "\n",
                            "Current configuration is: " + configuration + "\n"])
        print("Starting new simulation at timestamp: " + time.ctime(starttime))
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

        endtime = time.time()
        timelog.writelines(["Simulation now finished at timestamp: " + time.ctime(endtime) + "\n",
                            "Simulation took " + str(endtime-starttime) + " second to run\n\n"])
        print("Simulation now finished at timestamp: " + time.ctime(endtime))
        print("Simulation took " + str(endtime-starttime) + " second to run")

        # Clean up and move results
        simulationid = configuration[:-3]
        resultspath = os.path.join(resultsdir, simulationid)

        if not os.path.lexists(resultspath):
            os.mkdir(resultspath)

        sp.call(["cp", filepath, resultspath])

        # Copy output files away
        for spec in result_files:
            for afile in glob(spec):
                sp.call(["mv", afile, resultspath])

    timelog.close()
    print("Finished batch loop")


    # Postprocessing
    # Call scripts that should get called once after all simulations finished
    for command in call_postrun:
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
    # Assuming that it defines the four lists with script commands
    #  'call_prerun'
    #  'call_simulation'
    #  'call_for_each'
    #  'call_postrun'
    # and the list with output file regexp patterns
    #  'result_files'
    # in the toplevel namespace.
    exec(content)

    # Really start the scripts
    batch_run(call_prerun, call_simulation, call_for_each, call_postrun, result_files)
