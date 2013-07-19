"""The WaveBlocks Project

Plot the autocorrelations of the different wavepackets as well as the sum of all autocorrelations.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import max, real, imag, abs
from matplotlib.pyplot import *

from WaveBlocksND import IOManager
from WaveBlocksND.Plot import legend

import GraphicsDefaults as GD


def read_all_datablocks(iom):
    """Read the data from all blocks that contain any usable data.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    """
    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        if iom.has_autocorrelation(blockid=blockid):
            plot_autocorrelations(read_data(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: Not plotting autocorrelations in block '"+str(blockid)+"'!")


def read_data(iom, blockid=0):
    """
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    timegrid = iom.load_autocorrelation_timegrid(blockid=blockid)
    have_dt = iom.has_parameters()
    if have_dt:
        parameters = iom.load_parameters()
        time = timegrid * parameters["dt"]
    else:
        time = timegrid

    autocorrelations = iom.load_autocorrelation(blockid=blockid, split=True)

    autocorrelationsum = reduce(lambda x,y: x+y, autocorrelations)
    autocorrelations.append(autocorrelationsum)

    return (time, autocorrelations, have_dt)


def plot_autocorrelations(data, index=0):
    print("Plotting the autocorrelations of data block '"+str(index)+"'")

    timegrid, autocorrelations, have_dt = data

    if have_dt:
        xlbl=r"Time $t$"
    else:
        xlbl=r"Timesteps $n$"

    # Plot the autocorrelations
    fig = figure()
    ax = fig.gca()

    # Plot the autocorrelations of the individual wavepackets
    for i, datum in enumerate(autocorrelations[:-1]):
        label_i = r"$|\langle \Phi_"+str(i)+r"(0) | \Phi_"+str(i)+r"(t) \rangle|$"
        #ax.plot(timegrid, real(datum), label=label_i)
        #ax.plot(timegrid, imag(datum), label=label_i)
        ax.plot(timegrid, abs(datum), label=label_i)

    # Plot the sum of all autocorrelations
    ax.plot(timegrid, abs(autocorrelations[-1]), color=(1,0,0), label=r"$\sum_i {|\langle \Phi_i(0) | \Phi_i(t) \rangle|}$")

    ax.set_xlim(min(timegrid), max(timegrid))
    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_title(r"Autocorrelations of $\Psi$")
    legend(loc="upper right")
    ax.set_xlabel(xlbl)
    ax.set_ylim(0, 1.1)
    fig.savefig("autocorrelations_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the autocorrelations
    N = len(autocorrelations) -1
    fig = figure()

    # Plot the autocorrelations of the individual wavepackets
    for i, datum in enumerate(autocorrelations[:-1]):
        ax = fig.add_subplot(N, 1, i+1)
        label_ir = r"$\Re \langle \Phi_"+str(i)+r"(0) | \Phi_"+str(i)+r"(t) \rangle$"
        label_ii = r"$\Im \langle \Phi_"+str(i)+r"(0) | \Phi_"+str(i)+r"(t) \rangle$"
        label_im = r"$|\langle \Phi_"+str(i)+r"(0) | \Phi_"+str(i)+r"(t) \rangle|$"
        ax.plot(timegrid, real(datum), label=label_ir)
        ax.plot(timegrid, imag(datum), label=label_ii)
        ax.plot(timegrid, abs(datum), label=label_im)

        ax.set_xlim(min(timegrid), max(timegrid))
        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(xlbl)

        legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.set_title(r"Autocorrelations of $\Psi$")
    fig.savefig("autocorrelations_per_component_block"+str(index)+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read the file with the simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    read_all_datablocks(iom)

    iom.finalize()
