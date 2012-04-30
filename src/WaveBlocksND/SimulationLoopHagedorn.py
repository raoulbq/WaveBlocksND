"""The WaveBlocks Project

This file contains the main simulation loop
for the homogeneous Hagedorn propagator.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from SimulationLoop import SimulationLoop
from IOManager import IOManager
from TimeManager import TimeManager
from PotentialFactory import PotentialFactory
from BlockFactory import BlockFactory
from BasisTransformationHAWP import BasisTransformationHAWP
from HagedornPropagator import HagedornPropagator
from BlockFactory import BlockFactory


class SimulationLoopHagedorn(SimulationLoop):
    r"""This class acts as the main simulation loop. It owns a propagator that
    propagates a set of initial values during a time evolution.
    """

    def __init__(self, parameters):
        r"""Create a new simulation loop instance for a simulation
        using the semiclassical Hagedorn wavepacket based propagation
        method.

        :param parameters: The simulation parameters.
        :type parameters: A :py:class:`ParameterProvider` instance.
        """
        # Keep a reference to the simulation parameters
        self.parameters = parameters

        # The time propagator instance driving the simulation.
        self.propagator = None

        # A `IOManager` instance for saving simulation results.
        self.IOManager = None

        # Which data do we want to save
        self._tm = TimeManager(self.parameters)

        # Set up serializing of simulation data
        self.IOManager = IOManager()
        self.IOManager.create_file(self.parameters)


    def prepare_simulation(self):
        r"""Set up a Hagedorn propagator for the simulation loop. Set the
        potential and initial values according to the configuration.

        :raise ValueError: For invalid or missing input data.
        """
        # The potential instance
        potential = PotentialFactory().create_potential(self.parameters)

        # Project the initial values to the canonical basis
        BT = BasisTransformationHAWP(potential)

        # Finally create and initialize the propagator instace
        # TODO: Attach the "leading_component to the hawp as codata
        self.propagator = HagedornPropagator(self.parameters, potential)

        # Create  suitable wavepackets
        chi = self.parameters["leading_component"]

        for packet_descr in self.parameters["initvals"]:
            packet = BlockFactory().create_wavepacket(packet_descr)
            # Transform to canonical basis
            BT.set_matrix_builder(packet.get_quadrature())
            BT.transform_to_canonical(packet)
            # And hand over
            self.propagator.add_wavepacket((packet, chi))

        # Add storage for each packet
        npackets = len(self.parameters["initvals"])
        slots = self._tm.compute_number_saves()

        for i in xrange(npackets):
            bid = self.IOManager.create_block()
            self.IOManager.add_wavepacket(self.parameters, timeslots=slots, blockid=bid)

        # Write some initial values to disk
        for packet in self.propagator.get_wavepackets():
            self.IOManager.save_wavepacket_description(packet.get_description())
            # Pi
            self.IOManager.save_wavepacket_parameters(packet.get_parameters(), timestep=0)
            # Basis shapes
            for shape in packet.get_basis_shape():
                self.IOManager.save_wavepacket_basisshapes(shape)
            # Coefficients
            self.IOManager.save_wavepacket_coefficients(packet.get_coefficients(), packet.get_basis_shape(), timestep=0)


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
                # TODO: Generalize for arbitrary number of wavepackets
                packets = self.propagator.get_wavepackets()
                assert len(packets) == 1

                for packet in packets:
                    # Pi
                    self.IOManager.save_wavepacket_parameters(packet.get_parameters(), timestep=i)
                    # Basis shapes (in case they changed!)
                    for shape in packet.get_basis_shape():
                        self.IOManager.save_wavepacket_basisshapes(shape)
                    # Coefficients
                    self.IOManager.save_wavepacket_coefficients(packet.get_coefficients(), packet.get_basis_shape(), timestep=i)


    def end_simulation(self):
        r"""Do the necessary cleanup after a simulation. For example request the
        :py:class:`IOManager` to write the data and close the output files.
        """
        self.IOManager.finalize()
