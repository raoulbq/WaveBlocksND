A first glance
==============

Introduction
------------

The WaveBlocks project is a collection of reusable software components
providing many of the objects used in the study of semi-classical wavepackets.
Currently it's all about the time-dependent Schroedinger equation and the time
evolution of initial states.

One of the main goals is to provide a set of building blocks - hence the project's
name - that are well tested and reliable. The included features range from
very simple mathematical operations like specialised quadrature rules to basic
data structures for semi-classical wavepackets to more high-level simulation
algorithms and some non-standard plotting functions. Of course there are also
routines included for saving, managing and evaluating simulation results inflexible
manner. All of these components are put together in an easy to use and easy to
extend framework.

The whole project is written in the `python` programming language and a strong
emphasis is put on readable code and clean software design. Speed and efficiency
are, however, not the main concern for the moment.


Download
--------

The WaveBlocks project has its home at GitHub

https://github.com/raoulbq/WaveBlocksND

and the latest version can be found in the git
repository at the same place.

Check out the `master` branch for the main development version or one of the
tags for stable and released versions. There are also branches used for developing
special features before they are ready to get merged into the `master` branch.
Older stable versions are available as complete self-consistent tarballs.


Dependencies
------------

Probably the most difficult part of the installation is to get the dependencies
right. We need some more or less well known python packages that are not
installed by default. On any recent Linux distribution (for example Debian)
you can use the package manager to download and install all the dependencies.
First, make sure you run python 2.x and not python 3.x because some of
the following packages will not (yet) work with the latest python version. All
necessary dependencies are listed here together with a brief statement of why
we need the package.

* Numpy, available from http://www.numpy.org

  Numpy provides fast multidimensional arrays.

* Scipy, available from http://www.scipy.org

  Scipy interfaces to fast numerical subroutines (BLAS, LAPACK, FFTW).

* Sympy, available from http://sympy.org

  Sympy gives rise to (limited) symbolic calculation.

* Matplotlib, available from http://matplotlib.org

  Matplotlib is used for two-dimensional plotting.

* h5py, available from http://h5py.org

  H5py is the interface to the Hierarchical Data Format `hdf`.

* mayavi2, available from http://code.enthought.com/projects/mayavi/

  Mayavi2 is used for three-dimensional plotting.

The mayavi2 dependency is weak in the sense that the whole `WaveBlocks` code
runs fine without it and only the 3-dimensional plotting functions are not available.

The package numdifftools is already included in the program archive. You
should put it somewhere within your python path. (In the example instal-
lation from below, this would be ``~/python/numdifftools/``, just beside the
`WaveBlocks` directory). The package itself can be found at

http://code.google.com/p/numdifftools/

in case you want to look for a newer version. In general the latest version of
each software package should be used.


Installation
------------

In the following sections be sure not to mix up the name `WaveBlocks` that
stands for the whole project and the subdirectory `WaveBlocksND` which is the
python library and a small subset of the whole project.

Using a tarball
~~~~~~~~~~~~~~~

Installing the `WaveBlocks` code itself is trivial if you use one of the provided
tarballs. You just have to unpack the program archive and place the directories
``src/WaveBlocksND`` (which contains the library part of `WaveBlocks`) and
``src/numdifftools`` somewhere in your file system. Make sure the location is
within your `python path`, otherwise you'll have to adapt the environment variable
``PYTHONPATH``.

For example if you place these two directories below ``~/python`` then you have
to adapt the python path as follows:

::

    export PYTHONPATH="$PYTHONPATH:~/python"

You can write this line into your .bashrc file or any comparable file for your
default shell. (Otherwise this information is gone when you close the shell.)

From the repository
~~~~~~~~~~~~~~~~~~~

If you want to check out the source code from the repository, the necessary
steps are similar. First check out the code (we assume here you want to use
the `master` branch) into a new directory (here ``~/WB``)

::

    git clone https://github.com/raoulbq/WaveBlocksND.git ~/WB

The directory listing should now look like

::

    ls ~/WB/
    doc  examples  src

and we find the source code below src,

::

    ls ~/WB/src
    plotters  scripts  sh  tests  WaveBlocksND

Assume we want to install any custom python software locally below ``~/python``
so we create this directory and put all the things there. We start with the
`numdifftools` python package.

::

    mkdir ~/python
    cp -r ~/WB/src/numdifftools/ ~/python

We can just copy the files as they won't change in the git repository often. But
it's a wise idea not to copy ``~/WB/src/WaveBlocks`` but to instead use symbolic
links. (If you have multiple checkouts this is also a simple way to choose which
one will be used when you just type ``import WaveBlocksND`` at a python prompt.
This is similar to the alternatives framework in Debian.) You may just follow
the shell command given here for a working sample setup.

::

    ln -s ~/WB/src/WaveBlocksND WaveBlocksND

Finally we have to adapt the python path to include the directory ``~/python``
which can be done as follows:

::

    export PYTHONPATH="$PYTHONPATH:~/python"

You can write this last line into your ``.bashrc`` file or any comparable file
for your default shell. (Otherwise this information is gone when you close the shell.)

The scripts
~~~~~~~~~~~

The scripts (everything in ``src/scripts`` and ``src/plotters``)
that perform simulations, data evaluation and plotting can now be stored and
called from anywhere as these file are just plain python scripts that import the
``WaveBlocksND`` python module. It's best to put these scripts all together in a
directory where you plan to work and perform simulations.


Supported computing platforms
-----------------------------

The `WaveBlocksND` code should run on `Windows` and `Mac OS X` and the various
`BSD` variants too, provided that the required python dependencies are installed.
However, this has not yet been tested. The primary development platform is
`GNU/Linux` and in particular `Debian`.
