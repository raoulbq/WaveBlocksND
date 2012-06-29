r"""The WaveBlocks Project

Plot the evolution of the parameters :math:`Pi_i = (q, p, Q, P, S)`
of a homogeneous or inhomogeneous Hagedorn wavepacket during the
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs, squeeze, array
from numpy.linalg import norm, det
from matplotlib.pyplot import *

from WaveBlocksND import ComplexMath
from WaveBlocksND import IOManager

import GraphicsDefaults as GD


def read_all_datablocks(iom):
    r"""Read the data from all blocks that contain any usable data.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    """
    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        if iom.has_wavepacket(blockid=blockid):
            plot_parameters(read_data_homogeneous(iom, blockid=blockid), index=blockid)
        elif iom.has_inhomogwavepacket(blockid=blockid):
            plot_parameters(read_data_inhomogeneous(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: Not plotting wavepacket parameters in block '"+str(blockid)+"'!")


def read_data_homogeneous(iom, blockid=0):
    r"""
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_wavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    Pi = iom.load_wavepacket_parameters(blockid=blockid)
    qhist, phist, Qhist, Phist, Shist = Pi

    qhist = squeeze(array([ norm(qhist[i,:]) for i in xrange(qhist.shape[0]) ]))
    phist = squeeze(array([ norm(phist[i,:]) for i in xrange(phist.shape[0]) ]))
    Qhist = squeeze(array([ det(Qhist[i,:,:]) for i in xrange(Qhist.shape[0]) ]))
    Phist = squeeze(array([ det(Phist[i,:,:]) for i in xrange(Phist.shape[0]) ]))
    Shist = squeeze(Shist)

    return (time, [qhist], [phist], [Qhist], [Phist], [Shist])


def read_data_inhomogeneous(iom, blockid=0):
    r"""
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    Pis = iom.load_inhomogwavepacket_parameters(blockid=blockid)

    # List with Pi time evolutions
    Phist = []
    Qhist = []
    Shist = []
    phist = []
    qhist = []

    for q,p,Q,P,S in Pis:
        qhist.append(squeeze(q))
        phist.append(squeeze(p))
        Qhist.append(squeeze(Q))
        Phist.append(squeeze(P))
        Shist.append(squeeze(S))

    return (time, qhist, phist, Qhist, Phist, Shist)


def plot_parameters(data, index=0):
    r"""Plot the data parameters :math:`(q,p,Q,P,S)` over time.
    For each new `index` we start a new figure. This allows plotting
    several time evolutions to the same figure
    """
    print("Plotting the parameters of data block '"+str(index)+"'")

    timegrid, qhist, phist, Qhist, Phist, Shist = data

    # Plot the time evolution of the parameters q, p, Q, P and S
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in qhist:
        ax.plot(timegrid, real(item))
    ax.grid(True)
    ax.set_title(r"$\|q\|_2$")

    ax = fig.add_subplot(4,2,2)
    for item in phist:
        ax.plot(timegrid, real(item))
    ax.grid(True)
    ax.set_title(r"$\|p\|_2$")

    ax = fig.add_subplot(4,2,3)
    for item in Qhist:
        ax.plot(timegrid, real(item))
    ax.grid(True)
    ax.set_title(r"$\Re \det Q$")

    ax = fig.add_subplot(4,2,4)
    for item in Qhist:
        ax.plot(timegrid, imag(item))
    ax.grid(True)
    ax.set_title(r"$\Im \det Q$")

    ax = fig.add_subplot(4,2,5)
    for item in Phist:
        ax.plot(timegrid, real(item))
    ax.grid(True)
    ax.set_title(r"$\Re \det P$")

    ax = fig.add_subplot(4,2,6)
    for item in Phist:
        ax.plot(timegrid, imag(item))
    ax.grid(True)
    ax.set_title(r"$\Im \det P$")

    ax = fig.add_subplot(4,2,7)
    for item in Shist:
        ax.plot(timegrid, real(item))
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the time evolution of the parameters P, Q, S, p and q
    # This time plot abs/angle instead of real/imag
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in qhist:
        ax.plot(timegrid, real(item))
    ax.grid(True)
    ax.set_title(r"$\|q\|_2$")

    ax = fig.add_subplot(4,2,2)
    for item in phist:
        ax.plot(timegrid, real(item))
    ax.grid(True)
    ax.set_title(r"$\|p\|_2$")

    ax = fig.add_subplot(4,2,3)
    for item in Qhist:
        ax.plot(timegrid, abs(item))
    ax.grid(True)
    ax.set_title(r"$|\det Q|$")

    ax = fig.add_subplot(4,2,4)
    for item in Qhist:
        ax.plot(timegrid, ComplexMath.cont_angle(item))
    ax.grid(True)
    ax.set_title(r"$\arg \det Q$")

    ax = fig.add_subplot(4,2,5)
    for item in Phist:
        ax.plot(timegrid, abs(item))
    ax.grid(True)
    ax.set_title(r"$|\det P|$")

    ax = fig.add_subplot(4,2,6)
    for item in Phist:
        ax.plot(timegrid, ComplexMath.cont_angle(item))
    ax.grid(True)
    ax.set_title(r"$\arg \det P$")

    ax = fig.add_subplot(4,2,7)
    for item in Shist:
        ax.plot(timegrid, item, label=r"$S$")
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters_abs_ang_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the complex trajectory of the parameters P
    fig = figure()
    ax = fig.gca()
    for item in Phist:
        ax.plot(real(item), imag(item), "-o")
    ax.set_xlabel(r"$\Re \det P$")
    ax.set_ylabel(r"$\Im \det P$")
    ax.grid(True)
    ax.set_title(r"Trajectory of $\det P$")
    fig.savefig("wavepacket_parameters_trajectoryP_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the complex trajectory of the parameters Q
    fig = figure()
    ax = fig.gca()
    for item in Qhist:
        ax.plot(real(item), imag(item), "-o")
    ax.set_xlabel(r"$\Re \det Q$")
    ax.set_ylabel(r"$\Im \det Q$")
    ax.grid(True)
    ax.set_title(r"Trajectory of $\det Q$")
    fig.savefig("wavepacket_parameters_trajectoryQ_block"+str(index)+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Read the data and plot it, one plot for each data block.
    read_all_datablocks(iom)

    iom.finalize()
