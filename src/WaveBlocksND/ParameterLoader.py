"""The WaveBlocks Project

Reads configuration files containing the simulation parameters and
puts the values into a parameter provider instance.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import types
from copy import deepcopy

from ParameterProvider import ParameterProvider


class ParameterLoader(object):

    def __init__(self):
        pass


    def _get_configuration_variables(self, _scriptcode):
        """Clean environment for reading in local parameters.
        :param _scriptcode: String with the configuration code to execute.
        """
        # Execute the configuration file, they are plain python files
        exec(_scriptcode)

        # Filter out private variables (the ones prefixed by "_")
        # Instances like "self" and imported modules.
        parameters = locals().items()

        parameters = [ item for item in parameters if not type(item[1]) == types.ModuleType ]
        parameters = [ item for item in parameters if not type(item[1]) == types.InstanceType ]
        parameters = [ item for item in parameters if not item[0].startswith("_") ]

        return dict(parameters)


    def load_from_file(self, filepath):
        """Read the parameters from a configuration file.

        :param filepath: Path to the configuration file.
        :return: A :py:class:`ParameterProvider` instance.
        """
        # Read the configuration file
        cf = open(filepath)
        content = cf.read()
        cf.close()

        # All the parameters as dict
        params = self._get_configuration_variables(content)

        return self.load_from_dict(params)


    def load_from_dict(self, adict):
        """Construct a :py:class:`ParameterProvider` instance from a
        common python key-value dict.

        :param adict: A plain python `dict` with key-value pairs.
        :return: A :py:class:`ParameterProvider` instance.
        """
        PP = ParameterProvider()

        # Put the values into a ParameterProvider instance
        for key, value in adict.iteritems():
            PP[key] = deepcopy(value)

        # Compute some values on top of the given input parameters
        PP.compute_parameters()

        return PP
