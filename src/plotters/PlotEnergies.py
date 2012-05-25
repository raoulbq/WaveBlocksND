"""The WaveBlocks Project

Plot the energies of the different wavepackets as well as the sum of all energies.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import abs
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
        if iom.has_energy(blockid=blockid):
            plot_energies(read_data(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: Not plotting energies in block '"+str(blockid)+"'!")


def read_data(iom, blockid=0):
    """
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_energy_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    ekin, epot = iom.load_energy(blockid=blockid, split=True)

    # Compute the sum of all energies
    ekinsum = reduce(lambda x,y: x+y, ekin)
    epotsum = reduce(lambda x,y: x+y, epot)

    ekin.append(ekinsum)
    epot.append(epotsum)

    return (time, ekin, epot)


def plot_energies(data, index=0):
    print("Plotting the energies of data block '"+str(index)+"'")

    timegrid, ekin, epot = data

    # Plot the energies
    fig = figure()
    ax = fig.gca()

    # Plot the kinetic energy of the individual wave packets
    for i, kin in enumerate(ekin[:-1]):
        ax.plot(timegrid, kin, label=r"$E^{kin}_"+str(i)+r"$")

    # Plot the potential energy of the individual wave packets
    for i, pot in enumerate(epot[:-1]):
        ax.plot(timegrid, pot, label=r"$E^{pot}_"+str(i)+r"$")

    # Plot the sum of kinetic and potential energy for all wave packets
    for i, (kin, pot) in enumerate(zip(ekin, epot)[:-1]):
        ax.plot(timegrid, kin + pot, label=r"$E^{kin}_"+str(i)+r"+E^{pot}_"+str(i)+r"$")

    # Plot sum of kinetic and sum of potential energy
    ax.plot(timegrid, ekin[-1], label=r"$\sum_i E^{kin}_i$")
    ax.plot(timegrid, epot[-1], label=r"$\sum_i E^{pot}_i$")

    # Plot the overall energy of all wave packets
    ax.plot(timegrid, ekin[-1] + epot[-1], label=r"$\sum_i E^{kin}_i + \sum_i E^{pot}_i$")

    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    legend(loc="outer right")
    ax.set_title(r"Energies of the wavepacket $\Psi$")

    fig.savefig("energies_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the energy drift
    e_orig = (ekin[-1]+epot[-1])[0]
    data = abs(e_orig-(ekin[-1]+epot[-1]))

    fig = figure()
    ax = fig.gca()

    ax.plot(timegrid, data, label=r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")

    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.set_title(r"Energy drift of the wavepacket $\Psi$")

    fig.savefig("energy_drift_block"+str(index)+"_lin"+GD.output_format)
    close(fig)


    fig = figure()
    ax = fig.gca()

    ax.semilogy(timegrid, data, label=r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.set_title(r"Energy drift of the wavepacket $\Psi$")

    fig.savefig("energy_drift_block"+str(index)+"_log"+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    read_all_datablocks(iom)

    iom.finalize()
