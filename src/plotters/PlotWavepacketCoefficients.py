"""The WaveBlocks Project

Plot the evolution of the coefficients $c_i$ of each component
of a homogeneous or inhomogeneous Hagedorn wavepacket during the
time propagation.

Note that this script will fail to produce correct results if the
basis shapes are adaptive and their mappings mu incompatible!

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs, angle, squeeze
from matplotlib.pyplot import *

from WaveBlocksND import IOManager
from WaveBlocksND import BlockFactory
from WaveBlocksND.Plot import plotcm

import GraphicsDefaults as GD


def read_all_datablocks(iom):
    r"""Read the data from all blocks that contain any usable data.

    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    """
    parameters = iom.load_parameters()

    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        if iom.has_wavepacket(blockid=blockid):
            plot_coefficients(parameters, read_data_homogeneous(iom, blockid=blockid), index=blockid)
        elif iom.has_inhomogwavepacket(blockid=blockid):
            plot_coefficients(parameters, read_data_inhomogeneous(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: Not plotting wavepacket coefficients in block '"+str(blockid)+"'!")


def read_data_homogeneous(iom, blockid=0):
    r"""
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_wavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    hashes, coeffs = iom.load_wavepacket_coefficients(blockid=blockid, get_hashes=True)

    return time, coeffs, map(squeeze, hashes)


def read_data_inhomogeneous(iom, blockid=0):
    r"""
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    hashes, coeffs = iom.load_inhomogwavepacket_coefficients(blockid=blockid, get_hashes=True)

    return time, coeffs, map(squeeze, hashes)


def plot_coefficients(parameters, data, index=0, imgsize=(10,20)):
    """
    :param parameters: A :py:class:`ParameterProvider` instance.
    :param timegrid: The timegrid that belongs to the coefficient values.
    :param coeffs: The coefficient values.
    :param imgsize: The size of the plot. For a large number of plotted
                    coefficients, we might have to increase this value.
    """
    print("Plotting the coefficients of data block '"+str(index)+"'")

    # Check if we have enough coefficients to plot
    timegrid, coeffs, hashes = data

    N = len(coeffs)

    # Reconstruct basis shapes
    BS = []
    for i in xrange(N):
        d = iom.load_wavepacket_basisshapes(the_hash=hashes[i][0])
        bs = BlockFactory().create_basis_shape(d)
        BS.append(bs)

    # Compute all multi indices available (union of all shapes)
    allvects = set([])

    for bs in BS:
        allvects.update(set([i for i in bs]))

    # And plot
    for vect in allvects:
        print(" Plotting coefficient "+str(vect)+" out of "+str(len(allvects)))

        fig = figure()

        for level in xrange(N):
            j = BS[level][vect]
            if j is not None:

                ax = fig.add_subplot(N,1,level+1)
                ax.plot(timegrid, real(coeffs[level][:,j]))
                ax.plot(timegrid, imag(coeffs[level][:,j]))
                ax.plot(timegrid, abs(coeffs[level][:,j]))
                ax.grid(True)
                ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"$c^"+str(level)+r"$")

        fig.suptitle(r"$\underline{k} = "+str(vect)+r"$")
        fig.savefig("coefficient_k"+str(vect)+"_block"+str(index)+GD.output_format)
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
