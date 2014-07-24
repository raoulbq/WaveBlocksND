"""The WaveBlocks Project

Plot the wavefunctions probability densities
for one-dimensional wavefunctions.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import angle, conj, real, imag, squeeze
from matplotlib.pyplot import *

from WaveBlocksND import ParameterLoader
from WaveBlocksND import BlockFactory
from WaveBlocksND import IOManager
from WaveBlocksND.Plot import plotcf

import GraphicsDefaults as GD


def plot_frames(PP, iom, blockid=0, view=None, plotphase=True, plotcomponents=False, plotabssqr=False, load=True, gridblockid=None, imgsize=(12,9)):
    """Plot the wave function for a series of timesteps.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param view: The aspect ratio.
    :param plotphase: Whether to plot the complex phase. (slow)
    :param plotcomponents: Whether to plot the real/imaginary parts..
    :param plotabssqr: Whether to plot the absolute value squared.
    """
    parameters = iom.load_parameters()

    if not parameters["dimension"] == 1:
        print("No wavefunction of one space dimensions, silent return!")
        return

    if PP is None:
        PP = parameters

    if load is True:
        print("Loading grid data from datablock 'global'")
        if gridblockid is None:
            gridblockid = blockid
        G = iom.load_grid(blockid=gridblockid)
        G = G.reshape((1, -1))
        grid = G
    else:
        print("Creating new grid")
        G = BlockFactory().create_grid(PP)
        grid = G.get_nodes(flat=True)

    timegrid = iom.load_wavefunction_timegrid(blockid=blockid)

    for step in timegrid:
        print(" Plotting frame of timestep # " + str(step))

        wave = iom.load_wavefunction(blockid=blockid, timestep=step)
        values = [ wave[j,...] for j in xrange(parameters["ncomponents"]) ]

        # Plot the probability densities projected to the eigenbasis
        fig = figure(figsize=imgsize)

        for index, component in enumerate(values):
            ax = fig.add_subplot(parameters["ncomponents"],1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

            if plotcomponents is True:
                ax.plot(squeeze(grid), real(component))
                ax.plot(squeeze(grid), imag(component))
                ax.set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
            if plotabssqr is True:
                ax.plot(squeeze(grid), component*conj(component))
                ax.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
            if plotphase is True:
                plotcf(squeeze(grid), angle(component), component*conj(component))
                ax.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

            ax.set_xlabel(r"$x$")

            # Set the aspect window
            if view is not None:
                ax.set_xlim(view[:2])
                ax.set_ylim(view[2:])

        if parameters.has_key("dt"):
            fig.suptitle(r"$\Psi$ at time $"+str(step*parameters["dt"])+r"$")
        else:
            fig.suptitle(r"$\Psi$")
        fig.savefig("wavefunction_block"+str(blockid)+"_"+ (7-len(str(step)))*"0"+str(step) +GD.output_format)
        close(fig)

    print(" Plotting frames finished")




if __name__ == "__main__":
    iom = IOManager()
    PL = ParameterLoader()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Read file with parameter data for grid
    try:
        PP = PL.load_from_file(sys.argv[2])
    except IndexError:
        PP = None

    # The axes rectangle that is plotted
    view = [-1.5, 1.5, 0.0, 10.0]

    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        print("Plotting frames of data block '"+str(blockid)+"'")
        # See if we have wavefunction values
        if iom.has_wavefunction(blockid=blockid):
            plot_frames(PP, iom, blockid=blockid, view=view, plotphase=True, plotcomponents=False, plotabssqr=False, load=False)
        else:
            print("Warning: Not plotting any wavefunctions in block '"+str(blockid)+"'!")

    iom.finalize()
