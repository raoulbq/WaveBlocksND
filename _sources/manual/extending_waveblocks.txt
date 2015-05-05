Extending ``WaveBlocks``
========================

In this chapter we describe the steps to extend the ``WaveBlocks`` code base with
respect to various central aspects. The most important aspect is probably the
possibility to add new potentials of whatever kind. The other topics include
the question of how to save more data and what it takes to implement new
simulation algorithms.


Structure of potential definitions
----------------------------------

Performing a simulation with a new potential that is not (yet) part of the
``PotentialLibrary`` is not difficult. All potentials are specified as a python
dict. An simple example of a potential definition could look like:

::

    quadratic = {}
    quadratic["potential"] = "1/2 * sigma * x**2"
    quadratic["defaults"] = {"sigma":"1/2"}

We see that this is nothing else than a python dict (here assigned to the variable
``quadratic``). This dict must have a key ``potential`` and the corresponding value
is the analytic potential formula. This formula must contain the variable ``x`` which
is the position space variable. It may optionally have an arbitrary number of
parameters, here for example the ``sigma``.

The dict may have a further key called ``defaults`` whose value is another dict
containing default values for the parameters. These values are used if you do
not specify a value for a parameter in the simulation configuration file. You
may omit the defaults here, then you always have to provide a value in the
simulation configuration. Values given in the simulation configuration always
precede default values.

The analytic potential formula is given as a string. This string is parsed by
the ``sympify`` function of ``Sympy``. You can use many of the usual mathematical
operations and functions, for example (fractional) powers (``**``), roots (``sqrt()``),
trigonometric and hyperbolic functions and their inverse functions (but remem-
ber, it's ``atan`` and not ``arctan`` etc) and also some constants like ``pi``.
Please refer to the ``Sympy`` manual for a more thorough description of ``sympify``.
The most important point is to make sure that the formula finally yields a scalar
expression in ``x``.

A more complicated potential could be given by the following python code:

::

    delta_gap = {}
    delta_gap["potential"] = [["1/2 * tanh(x)", "delta"         ],
                              ["delta",         "-1/2 * tanh(x)"]]

This potential has two energy levels and is given by a :math:`2\times 2` matrix.
The matrix is constructed as two nested python lists. Each entry is a single string
as above. Note also that this potential does not specify a default for ``delta``,
you have to assign a scalar float value to a variable called ``delta`` in the simulation
configuration. You may specify potentials with as many energy levels as you like.

A final word on the strings representing formulae and parameter values: any
string that `only` consists of a numerical constant my be replaced by a float. But
be careful with divisions, in python 2.x, ``1/2`` is ``0``!


Adding new potentials
---------------------

After we have seen the general structure of a potential definition in the last
section, let's now look at how to add and use a new potential. For new potentials
you have basically two options. Either place the potential definition in the
``PotentialLibrary.py`` file. Make sure your variable name (for example
``delta_gap``) is unique. If you decide to go this way, you can use the new potential
by referring to it in the simulation configuration as:

::

    potential = "delta_gap"

The other option is to place the `whole` potential definition inside the simulation
configuration file. Then this would look like:

::

    potential = {}
    potential["potential"] = [["1/2 * tanh(x)", "delta"         ],
                              ["delta",         "-1/2 * tanh(x)"]]

This time you must name the dict ``potential`` because that's the variable name
where the ``ParameterLoader`` looks for a potential. For the first time
it might be confusing that the dict as well as its main key both bear the name
``potential`` but this is correct. Of course you can specify default values here too,
but probably it has no real use case.

As a small side note, here is an example of a custom potential when we create the
configurations by the ``ConfigurationGenerator``. (This is only a small excerpt
of a metaconfiguration file.)

::

    # A custom potential without a default for " alpha "
    GP["potential"] = {}
    GP["potential[\"potential\"]"] = "\"cosh(alpha*x)\""
    # Different values for "alpha" in each simulation
    LP["alpha"] = [0.1*i for i in range(1,5)]

Please note the escaped quotes, it’s essential to do this correct here.


Compute and store more data
---------------------------

``IOM`` and ``IOM`` Plugins, how to teach the ``IOM`` to store more/other data.


New propagation algorithms
--------------------------

Implementing the ``SimulationLoop`` and ``Propagator`` interfaces
