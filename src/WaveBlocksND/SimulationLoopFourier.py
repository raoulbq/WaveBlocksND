"""The WaveBlocks Project

This file contains the main simulation loop
for the Fourier propagator.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from GridFactory import GridFactory
from PotentialFactory import PotentialFactory
from Initializer import Initializer
from BasisTransformationWF import BasisTransformationWF
from FourierPropagator import FourierPropagator
from SimulationLoop import SimulationLoop
from IOManager import IOManager

__all__ = ["SimulationLoopFourier"]


class SimulationLoopFourier(SimulationLoop):
    """This class acts as the main simulation loop. It owns a propagator that
    propagates a set of initial values during a time evolution.
    """

    def __init__(self, parameters):
        """Create a new simulation loop instance for a simulation
        using the Fourier propagation method.

        :param parameters: The simulation parameters.
        :type parameters: A :py:class:`ParameterProvider` instance.
        """
        # Keep a reference to the simulation parameters
        self.parameters = parameters

        # The time propagator instance driving the simulation.
        self.propagator = None

        # An `IOManager` instance for saving simulation results.
        self.IOManager = None

        # Which data do we want to save
        self._tm = self.parameters.get_timemanager()

        # Set up serialization of simulation data
        self.IOManager = IOManager()
        self.IOManager.create_file(self.parameters)
        self.IOManager.create_block()


    def prepare_simulation(self):
        r"""Set up a Fourier propagator for the simulation loop. Set the
        potential and initial values according to the configuration.

        :raise ValueError: For invalid or missing input data.
        """
        # The potential instance
        potential = PotentialFactory().create_potential(self.parameters)

        # Compute the position space grid points
        grid = GridFactory().create_grid(self.parameters)

        # Construct initial values
        I = Initializer(self.parameters)
        initialvalues = I.initialize_for_fourier(grid)

        # Transform the initial values to the canonical basis
        BT = BasisTransformationWF(potential)
        BT.set_grid(grid)
        BT.transform_to_canonical(initialvalues)

        # Finally create and initialize the propagator instace
        self.propagator = FourierPropagator(potential, initialvalues, self.parameters)

        # Write some initial values to disk
        slots = self._tm.compute_number_saves()

        self.IOManager.add_grid(self.parameters, blockid="global")
        self.IOManager.add_fourieroperators(self.parameters)
        self.IOManager.add_wavefunction(self.parameters, timeslots=slots)

        self.IOManager.save_grid(grid.get_nodes(flat=False), blockid="global")
        self.IOManager.save_fourieroperators(self.propagator.get_operators())
        self.IOManager.save_wavefunction(initialvalues.get_values(), timestep=0)


    def run_simulation(self):
        r"""Run the simulation loop for a number of time steps.
        """
        # The number of time steps we will perform.
        nsteps = self._tm.compute_number_timesteps()

        # Run the simulation for a given number of timesteps
        for i in xrange(1, nsteps+1):
            print(" doing timestep "+str(i))

            self.propagator.propagate()

            # Save some simulation data
            if self._tm.must_save(i):
                self.IOManager.save_wavefunction(self.propagator.get_wavefunction().get_values(), timestep=i)


    def end_simulation(self):
        """Do the necessary cleanup after a simulation. For example request the
        :py:class:`IOManager` to write the data and close the output files.
        """
        self.IOManager.finalize()
