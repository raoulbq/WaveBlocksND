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
from numpy import abs, angle
from matplotlib.pyplot import *

from WaveBlocksND import IOManager
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

    return time, coeffs


def read_data_inhomogeneous(iom, blockid=0):
    r"""
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    hashes, coeffs = iom.load_inhomogwavepacket_coefficients(blockid=blockid, get_hashes=True)

    return time, coeffs


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
    timegrid, coeffs = data

    # Plot for each of the N levels
    for jndex, coeff in enumerate(coeffs):

        # Scale the image to roughly fit the data shape
        v, u = coeff.shape
        imags = (imgsize[0], int(10*v / (1.0*u)))
        if imags[1] < 10000:
            imgsize = imags

        fig = figure(figsize=imgsize)
        ax = gca()
        ax.matshow(abs(coeff))
        ax.set_xlabel("k")
        ax.set_ylabel("t")
        ax.set_title(r"Coefficients $c_k^"+str(jndex)+"$")
        fig.savefig("wavepacket_coefficients_map_abs_component"+str(jndex)+"_block"+str(index)+GD.output_format)
        close(fig)

        fig = figure(figsize=imgsize)
        ax = gca()
        ax.matshow(angle(coeff))
        ax.set_xlabel("k")
        ax.set_ylabel("t")
        ax.set_title(r"Coefficients $c_k^"+str(jndex)+"$")
        fig.savefig("wavepacket_coefficients_map_angle_component"+str(jndex)+"_block"+str(index)+GD.output_format)
        close(fig)

        fig = figure(figsize=imgsize)
        ax = gca()
        plotcm(coeff, angle(coeff), abs(coeff), darken=False, axes=ax)
        ax.set_xlabel("k")
        ax.set_ylabel("t")
        ax.set_title(r"Coefficients $c_k^"+str(jndex)+"$")
        fig.savefig("wavepacket_coefficients_map_cm_component"+str(jndex)+"_block"+str(index)+GD.output_format)
        close(fig)

        fig = figure(figsize=imgsize)
        ax = gca()
        plotcm(coeff, angle(coeff), abs(coeff), darken=True, axes=ax)
        ax.set_xlabel("k")
        ax.set_ylabel("t")
        ax.set_title(r"Coefficients $c_k^"+str(jndex)+"$")
        fig.savefig("wavepacket_coefficients_map_cm_darken_component"+str(jndex)+"_block"+str(index)+GD.output_format)
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
