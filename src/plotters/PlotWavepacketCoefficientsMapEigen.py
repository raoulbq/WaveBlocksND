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
from numpy import abs, angle, transpose, hstack
from matplotlib.pyplot import *

from WaveBlocksND import IOManager
from WaveBlocksND import BlockFactory
from WaveBlocksND import BasisTransformationHAWP

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

    # The potential used
    Potential = BlockFactory().create_potential(parameters)

    # Basis transformator
    BT = BasisTransformationHAWP(Potential)

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes(blockid=blockid)
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_wavepacket_description(blockid=blockid)
    HAWP = BlockFactory().create_wavepacket(descr)

    BT.set_matrix_builder(HAWP.get_quadrature())

    # Store the resulting coefficients here
    CI = [ [] for i in xrange(HAWP.get_number_components()) ]

    # Iterate over all timesteps, this is an *expensive* transformation
    for i, step in enumerate(timegrid):
        print(" Computing eigentransformation at timestep "+str(step))
        # Retrieve simulation data
        params = iom.load_wavepacket_parameters(timestep=step, blockid=blockid)
        hashes, coeffs = iom.load_wavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWP.set_parameters(params)
        HAWP.set_basis_shape([ BS[int(ha)] for ha in hashes ])
        HAWP.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        BT.transform_to_eigen(HAWP)

        for index, item in enumerate(HAWP.get_coefficients()):
            CI[index].append(item)

    CI = [ transpose(hstack(item)) for item in CI ]

    return time, CI


def read_data_inhomogeneous(iom, blockid=0):
    r"""
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    # The potential used
    Potential = BlockFactory().create_potential(parameters)

    # Basis transformator
    BT = BasisTransformationHAWP(Potential)

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes(blockid=blockid)
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BlockFactory().create_basis_shape(descr)

    # Initialize a Hagedorn wavepacket with the data
    descr = iom.load_inhomogwavepacket_description(blockid=blockid)
    HAWP = BlockFactory().create_wavepacket(descr)

    BT.set_matrix_builder(HAWP.get_quadrature())

    # Store the resulting coefficients here
    CI = [ [] for i in xrange(HAWP.get_number_components()) ]

    # Iterate over all timesteps, this is an *expensive* transformation
    for i, step in enumerate(timegrid):
        print(" Computing eigentransformation at timestep "+str(step))
        # Retrieve simulation data
        params = iom.load_inhomogwavepacket_parameters(timestep=step, blockid=blockid)
        hashes, coeffs = iom.load_inhomogwavepacket_coefficients(timestep=step, get_hashes=True, blockid=blockid)

        # Configure the wavepacket
        HAWP.set_parameters(params)
        HAWP.set_basis_shape([ BS[int(ha)] for ha in hashes ])
        HAWP.set_coefficients(coeffs)

        # Transform to the eigenbasis.
        BT.transform_to_eigen(HAWP)

        for index, item in enumerate(HAWP.get_coefficients()):
            CI[index].append(item)

    CI = [ transpose(hstack(item)) for item in CI ]

    return time, CI


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
        fig.savefig("wavepacket_coefficients_map_eigen_abs_component"+str(jndex)+"_block"+str(index)+GD.output_format)
        close(fig)

        fig = figure(figsize=imgsize)
        ax = gca()
        ax.matshow(angle(coeff))
        ax.set_xlabel("k")
        ax.set_ylabel("t")
        ax.set_title(r"Coefficients $c_k^"+str(jndex)+"$")
        fig.savefig("wavepacket_coefficients_map_eigen_angle_component"+str(jndex)+"_block"+str(index)+GD.output_format)
        close(fig)

        fig = figure(figsize=imgsize)
        ax = gca()
        plotcm(coeff, angle(coeff), abs(coeff), darken=False, axes=ax)
        ax.set_xlabel("k")
        ax.set_ylabel("t")
        ax.set_title(r"Coefficients $c_k^"+str(jndex)+"$")
        fig.savefig("wavepacket_coefficients_map_eigen_cm_component"+str(jndex)+"_block"+str(index)+GD.output_format)
        close(fig)

        fig = figure(figsize=imgsize)
        ax = gca()
        plotcm(coeff, angle(coeff), abs(coeff), darken=True, axes=ax)
        ax.set_xlabel("k")
        ax.set_ylabel("t")
        ax.set_title(r"Coefficients $c_k^"+str(jndex)+"$")
        fig.savefig("wavepacket_coefficients_map_eigen_cm_darken_component"+str(jndex)+"_block"+str(index)+GD.output_format)
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
