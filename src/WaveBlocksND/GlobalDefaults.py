"""The WaveBlocks Project

This file contains some global defaults, for example file names for output files.
If a :py:class:`ParameterProvider` instance is asked about a key which it does not
know about it tries to look it up here to see if a default value is available.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

# Some global parameters related to file naming
path_to_autogen_configs = "autogen_configurations"
path_to_configs = "configurations"
path_to_results = "results"

file_metaconfiguration = "metaconfiguration.py"
file_resultdatafile = "simulation_results.hdf5"
file_batchconfiguration = "batchconfiguration.py"

# Left, middle and right delimiter for key->value pairs
# encoded into filenames (as used by the FileTools)
kvp_ldel = "["
kvp_mdel = "="
kvp_rdel = "]"


# Defaults for some internal initializations

# For creating Hagedorn wavepackets
default_Pi = [1.0j, 1.0, 0.0, 0.0, 0.0]
default_basis_size = 8


# Defaults for some simulation configuration parameters
try_simplification = True


# Matrix exponential algorithm
matrix_exponential = "arnoldi"
arnoldi_steps = 20

# Default values about when to save the results
write_nth = 0
save_at = []
