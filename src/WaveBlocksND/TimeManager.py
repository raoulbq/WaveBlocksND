"""The WaveBlocks Project

Provides several computation routines for
handling time and timesteps.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from scipy import floor

# TODO: Clean up and allow for calculation of all time related quantities given a subset

class TimeManager(object):
    """This class performs several computation with time, timesteps and so for.
    The important quantities here are:
    T  : the fixed simulation end time
    dt : the size of the timestep
    N  : the overall number of timesteps.
    t  : an unspecified time in the interval [0, T]
    n  : an unspecified timestep in the interval [0, N]
    The importtant relations that hold are:
    T = N * dt  and in analogy  t = n * dt
    There are also conversion routines for t and n.
    Additionally the class contains some routines for determining
    if and when to save data. But we do not touch any data in here.
    """

    def __init__(self, parameters):
        if parameters is None:
            parameters = {}

        if parameters.has_key("T") and parameters.has_key("dt"):
            self.set_T(parameters["T"])
            self.set_dt(parameters["dt"])
        else:
            raise KeyError("Parameters provide to little data to construct a 'TimeManager'.")

        if parameters.has_key("nsteps"):
            self.set_nsteps(parameters["nsteps"])
        else:
            self.set_nsteps(None)

        #: Interval for saving
        if parameters.has_key("write_nth"):
            self.set_interval(parameters["write_nth"])
        else:
            self.set_interval(1)

        #: List of timesteps when we have to save
        if parameters.has_key("save_at"):
            self.add_to_savelist(parameters["save_at"])
        else:
            self.savetimes = []


    def __str__(self):
        s  = "TimeManager configured with:\n"
        s += " Final time     T: " +str(self.T) +"\n"
        s += " Timestep size dt: " +str(self.dt) +"\n"
        s += " Interval        : " +str(self.interval) +"\n"
        s += " List            : " +str(self.savetimes) +"\n"
        return s


    def set_T(self, T):
        """Set the simulation endtime T.
        @param T: The simulation end time.
        """
        self.T = T


    def set_dt(self, dt):
        """Set the simulation timestep size dt.
        @param dt: The simulation timestep size.
        """
        self.dt = dt


    def set_nsteps(self, nsteps):
        """Set the number of timesteps the simulation runs.
        @param nsteps: The number timesteps we do.
        """
        self.nsteps = nsteps


    def set_interval(self, interval):
        """Set the inteval for saving results.
        @param interval: The interval at which we save simulation results.
        @note: A value of 0 means we never save data at any regular interval.
        """
        self.interval = interval


    def get_nsteps(self):
        if self.nsteps is None:
            self.nsteps = self.compute_number_timesteps(self)
        return self.nsteps


    def compute_number_timesteps(self):
        """Computes the number of time steps we will perform.
        """
        # This is independent from if, when and what data we save
        if self.nsteps is not None:
            return self.nsteps
        else:
            return int( floor(self.T / self.dt) )


    def compute_timestep(self, t):
        """Compute the timestep n from a time t such that t = n * dt holds.
        @param t: The time t of which we want to find the timestep number.
        @note: The user has to ensure that time is an integral multiple of dt.
        """
        stepo = t / self.dt
        step = round(stepo)

        if abs(stepo - step) > 10**-10:
            print("Warning: questionable rounding for timestep computation!")

        return int(step)


    def compute_time(self, n):
        """Compute the time t from a timestep n such that t = n * dt holds.
        @param n: The timestep n of which we want to find the corresponding time.
        """
        return 1.0 * n * self.dt


    def add_to_savelist(self, alist):
        """Add a list of times and/or timesteps to the list of times
        which determine when to save data.
        @param alist: A list with integers (interpreted as timesteps) and/or floats (interpreted as times)
        @note: The times and timesteps can be mixed and needn't to be given in monotone order.
        """
        timesteps = []

        # If the list is empty (global default), shortcut
        if len(alist) == 0:
            return

        # Integers are interpreted as timesteps, floats are interpreted as times (and converted to timesteps)
        for item in alist:
            if type(item) == int:
                timesteps.append(item)
            elif type(item) == float:
                timesteps.append( self.compute_timestep(item) )

        # Validate timesteps and check if n in [0,...,N]
        tmp = len(timesteps)
        nsteps = self.compute_number_timesteps()
        timesteps = [ i for i in timesteps if i > 0 and i <= nsteps ]

        if tmp != len(timesteps):
            print("Warning: Dropped some save timesteps due to invalidity!")

        # Assure unique elements, just silently remove duplicates
        oldlist = set(self.savetimes)
        newlist = set(timesteps)
        times = list(oldlist.union(newlist))
        # Sort in ascending order
        times.sort()
        # Write back
        self.savetimes = times


    # TODO: Save and savelist -> event and eventlist

    def compute_number_saves(self):
        """Compute the number of saves we will perform during the simulation. This
        can be used to determine how much space to allocate in the output files.
        """
        # We do not save at regular intervals
        if self.interval == 0:
            # Determine the number of saves resulting from saving at a regular interval is zero.
            n_si = 0
            # Determine the number of saves resulting from the savelist
            n_sl = len(self.savetimes)
        # We do save at regular intervals
        else:
            # Determine the number of saves resulting from saving at a regular interval
            n_ts = self.compute_number_timesteps()
            n_si = n_ts // self.interval
            # Determine the number of saves resulting from the savelist and
            # exclude the timesteps which coincide with the regular intervals.
            n_sl = len( [ i for i in self.savetimes if i % self.interval != 0 ] )

        # Total number of saves we will perform is given by the sum plus the initial value
        number_saves = 1 + n_si + n_sl

        return number_saves


    def must_save(self, n):
        """Determine if we have to save right now.
        @param n: The current timestep in question.
        """
        if self.interval == 1:
            # Save every timestep
            return True
        elif self.interval != 0  and  n % self.interval == 0:
            # Save every k-th timestep specified by the inetrval
            return True
        elif n in self.savetimes:
            # Save if the n is in the list of timesteps
            return True

        return False
