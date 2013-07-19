"""The WaveBlocks Project

Plot the norms of the different wavepackets as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import sqrt, max
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
        if iom.has_norm(blockid=blockid):
            plot_norms(read_data(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: Not plotting norms in block '"+str(blockid)+"'!")


def read_data(iom, blockid=0):
    """
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    timegrid = iom.load_norm_timegrid(blockid=blockid)
    have_dt = iom.has_parameters()
    if have_dt:
        parameters = iom.load_parameters()
        time = timegrid * parameters["dt"]
    else:
        time = timegrid

    norms = iom.load_norm(blockid=blockid, split=True)

    normsum = [ item**2 for item in norms ]
    normsum = reduce(lambda x,y: x+y, normsum)
    norms.append(sqrt(normsum))

    return (time, norms, have_dt)


def plot_norms(data, index=0):
    print("Plotting the norms of data block '"+str(index)+"'")

    timegrid, norms, have_dt = data

    if have_dt:
        xlbl=r"Time $t$"
    else:
        xlbl=r"Timesteps $n$"

    # Plot the norms
    fig = figure()
    ax = fig.gca()

    # Plot the norms of the individual wavepackets
    for i, datum in enumerate(norms[:-1]):
        label_i = r"$\| \Phi_"+str(i)+r" \|$"
        ax.plot(timegrid, datum, label=label_i)

    # Plot the sum of all norms
    ax.plot(timegrid, norms[-1], color=(1,0,0), label=r"${\sqrt{\sum_i {\| \Phi_i \|^2}}}$")

    ax.set_xlim(min(timegrid), max(timegrid))
    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_title(r"Norms of $\Psi$")
    legend(loc="outer right")
    ax.set_xlabel(xlbl)
    ax.set_ylim([0,1.1*max(norms[:-1])])
    fig.savefig("norms_block"+str(index)+GD.output_format)
    close(fig)


    fig = figure()
    ax = fig.gca()

    # Plot the squared norms of the individual wavepackets
    for i, datum in enumerate(norms[:-1]):
        label_i = r"$\| \Phi_"+str(i)+r" \|^2$"
        ax.plot(timegrid, datum**2, label=label_i)

    # Plot the squared sum of all norms
    ax.plot(timegrid, norms[-1]**2, color=(1,0,0), label=r"${\sum_i {\| \Phi_i \|^2}}$")

    ax.set_xlim(min(timegrid), max(timegrid))
    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_title(r"Squared norms of $\Psi$")
    legend(loc="outer right")
    ax.set_xlabel(xlbl)
    ax.set_ylim([0,1.1*max(norms[-1]**2)])
    fig.savefig("norms_sqr_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the difference from the theoretical norm
    fig = figure()
    ax = fig.gca()

    ax.plot(timegrid, abs(norms[-1][0] - norms[-1]), label=r"$\|\Psi\|_0 - \|\Psi\|_t$")

    ax.set_xlim(min(timegrid), max(timegrid))
    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_title(r"Drift of $\| \Psi \|$")
    legend(loc="outer right")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(r"$\|\Psi\|_0 - \|\Psi\|_t$")
    fig.savefig("norms_drift_block"+str(index)+GD.output_format)
    close(fig)


    fig = figure()
    ax = fig.gca()

    ax.semilogy(timegrid, abs(norms[-1][0] - norms[-1]), label=r"$\|\Psi\|_0 - \|\Psi\|_t$")

    ax.set_xlim(min(timegrid), max(timegrid))
    ax.grid(True)
    ax.set_title(r"Drift of $\| \Psi \|$")
    legend(loc="outer right")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(r"$\|\Psi\|_0 - \|\Psi\|_t$")
    fig.savefig("norms_drift_block"+str(index)+"_log"+GD.output_format)
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
