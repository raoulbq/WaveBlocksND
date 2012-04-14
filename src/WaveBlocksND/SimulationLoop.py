"""The WaveBlocks Project

This file contains the main simulation loop interface.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

class SimulationLoop(object):
    """This class acts as the main simulation loop. It owns a propagator that
    propagates a set of initial values during a time evolution. It is responsible
    for preparing the simulation, setting up initial values and store the simulation
    data with the help of an :py:class:`IOManager` instance.
    """

    def __init__(self, parameters):
        """Create a new simulation loop instance.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")


    def prepare_simulation(self):
        """Set up a propagator for the simulation loop. Set the
        potential and initial values according to the configuration.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")


    def run_simulation(self):
        """Run the simulation. This method will implement the central loop running
        over all timesteps. Inside of this loop it will call the `propagate` method
        of its propagator and save the simulation results.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")


    def end_simulation(self):
        """Do the necessary cleanup after a simulation. For example request the
        IOManager to write the data and close the output files. Shut down the
        simulation process.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")
