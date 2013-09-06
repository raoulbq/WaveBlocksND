"""The WaveBlocks Project

Provides several computation routines for
handling time and timesteps.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from scipy import floor

__all__ = ["TimeManager"]


class TimeManager(object):
    r"""This class performs several computations with time, timesteps and so forth.

    The important quantities here are:

    ============ ============== ======================================================
    Quantity     Parameter Name Description
    ============ ============== ======================================================
    :math:`T`    T              the fixed simulation end time
    :math:`\tau` dt             the size of the timestep
    :math:`N`    nsteps         the overall number of timesteps.
    :math:`t`                   an unspecified time in the interval :math:`[0, T]`
    :math:`n`                   an unspecified timestep in the interval :math:`[0, N]`
    ============ ============== ======================================================

    The important relations that hold are :math:`T = N \tau` and
    in analogy :math:`t = n \tau`. There are also conversion routines
    for :math:`t` and :math:`n`.

    The simulation parameters handed over to the constructor must contain at least
    two out of the three values :math:`T`, :math:`\tau` and :math:`N`. If all three
    are given, the user is responsible for compatible values.

    Additionally the class contains some routines for determining
    if and when to save data. But we do not touch any data in here.
    """

    def __init__(self, parameters):
        if parameters is None:
            parameters = {}

        # We need two out of three: T, dt and nsteps
        have_enough = 0

        if parameters.has_key("T"):
            self._T = parameters["T"]
            have_enough += 1
        else:
            self._T = None

        if parameters.has_key("dt"):
            self._dt = parameters["dt"]
            have_enough += 1
        else:
            self._dt = None

        if parameters.has_key("nsteps"):
            self._nsteps = parameters["nsteps"]
            have_enough += 1
        else:
            self._nsteps = None

        if have_enough < 2:
            raise KeyError("Parameters provide to little data to construct a 'TimeManager'.")

        if self._T is None:
            self._T = self.compute_endtime()

        if self._dt is None:
            self._dt = self.compute_timestep_size()

        if self._nsteps is None:
            self._nsteps = self.compute_number_timesteps()

        # Interval for saving
        self._interval = 1
        if parameters.has_key("write_nth"):
            self.set_interval(parameters["write_nth"])

        # List of timesteps when we have to save
        self._savetimes = []
        if parameters.has_key("save_at"):
            self.add_to_savelist(parameters["save_at"])


    def __str__(self):
        s  = "TimeManager configured with:\n"
        s += " Final time     T: " +str(self._T) +"\n"
        s += " Timestep size dt: " +str(self._dt) +"\n"
        s += " Number of steps : " +str(self._nsteps) +"\n"
        return s


    def set_T(self, T):
        r"""Set the simulation endtime :math:`T`.

        :param T: The simulation end time.
        """
        self._T = T


    def set_dt(self, dt):
        r"""Set the simulation timestep size :math:`\tau`.

        :param dt: The simulation timestep size.
        """
        self._dt = dt


    def set_nsteps(self, nsteps):
        r"""Set the number of timesteps the simulation runs.

        :param nsteps: The number :math:`n` timesteps we do.
        """
        self._nsteps = nsteps


    def get_T(self):
        r"""Set the simulation endtime :math:`T`.

        :returns: The endtime :math:`T`.
        """
        return self._T


    def get_dt(self):
        r"""Get the simulation timestep size :math:`\tau`.

        :returns: The timestep :math:`\tau`.
        """
        return self._dt


    def get_nsteps(self):
        r"""Get the number :math:`n` of timesteps the simulation runs.

        :returns: the number :math:`n` of timesteps.
        """
        return self._nsteps


    def compute_endtime(self):
        r"""Computes the simulation endtime :math:`T`.

        :returns: The endtime :math:`T`.
        """
        if self._T is not None:
            return self._T
        else:
            return self._nsteps * self._dt


    def compute_timestep_size(self):
        r"""Computes the simulation timestep size :math:`\tau`.

        :returns: The timestep :math:`\tau`.
        """
        if self._dt is not None:
            return self._dt
        else:
            return self._T / (1.0 * self._nsteps)


    def compute_number_timesteps(self):
        r"""Computes the number :math:`n` of time steps we will perform.

        :returns: the number :math:`n` of timesteps.
        """
        if self._nsteps is not None:
            return self._nsteps
        else:
            return int( floor(self._T / (1.0 * self._dt)) )


    def compute_timestep(self, t):
        r"""Compute the timestep :math:`n` from a time :math:`t` such that
        :math:`t = n \tau` holds.

        :param t: The time t of which we want to find the timestep number.
        :returns: The corresponding timestep :math:`n`.

        Note that the user has to ensure that time :math:`t` is an integral
        multiple of :math:`\tau`.
        """
        stepo = t / self._dt
        step = round(stepo)

        if abs(stepo - step) > 10**-10:
            print("Warning: questionable rounding for timestep computation!")

        return int(step)


    def compute_time(self, n):
        r"""Compute the time :math:`t` from a timestep :math:`n` such that
        :math:`t = n \tau` holds.

        :param n: The timestep n of which we want to find the corresponding time.
        :returns: The corresponding time :math:`t`.
        """
        return 1.0 * n * self._dt


    def set_interval(self, interval):
        r"""Set the interval for saving results.

        :param interval: The interval at which we save simulation results.

        Note that a value of ``0`` means we never save data at any regular interval.
        """
        self._interval = interval


    def add_to_savelist(self, alist):
        r"""Add a list of times and/or timesteps to the list of times
        which determine when to save data.

        :param alist: A list with integers (interpreted as timesteps)
                      and/or floats (interpreted as times)

        Note that the times and timesteps can be mixed and need not to be
        given in monotone order.
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
        timesteps = [ i for i in timesteps if i > 0 and i <= self._nsteps ]

        if tmp != len(timesteps):
            print("Warning: Dropped some save timesteps due to invalidity!")

        # Assure unique elements, just silently remove duplicates
        oldlist = set(self._savetimes)
        newlist = set(timesteps)
        times = list(oldlist.union(newlist))
        # Sort in ascending order
        times.sort()
        # Write back
        self._savetimes = times


    # TODO: Save and savelist -> event and eventlist

    def compute_number_saves(self):
        r"""Compute the number of saves we will perform during the simulation. This
        can be used to determine how much space to allocate in the output files.

        :returns: The number of times we will save something.
        """
        # We do not save at regular intervals
        if self._interval == 0:
            # Determine the number of saves resulting from saving at a regular interval is zero.
            n_si = 0
            # Determine the number of saves resulting from the savelist
            n_sl = len(self._savetimes)
        # We do save at regular intervals
        else:
            # Determine the number of saves resulting from saving at a regular interval
            n_si = self._nsteps // self._interval
            # Determine the number of saves resulting from the savelist and
            # exclude the timesteps which coincide with the regular intervals.
            n_sl = len( [ i for i in self._savetimes if i % self._interval != 0 ] )

        # Total number of saves we will perform is given by the sum plus the initial value
        number_saves = 1 + n_si + n_sl

        return number_saves


    def must_save(self, n):
        r"""Determine if we have to save right now.

        :param n: The current timestep in question.
        :returns: ``True`` or ``False``.
        """
        if self._interval == 1:
            # Save every timestep
            return True
        elif self._interval != 0  and  n % self._interval == 0:
            # Save every k-th timestep specified by the interval
            return True
        elif n in self._savetimes:
            # Save if the n is in the list of timesteps
            return True

        return False
