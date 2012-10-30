"""The WaveBlocks Project

Plot the wavepackets probability densities
for two-dimensional wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, squeeze
from matplotlib.pyplot import *

from WaveBlocksND import BlockFactory
from WaveBlocksND import BasisTransformationHAWP
from WaveBlocksND import IOManager

from WaveBlocksND.Plot import plotcf2d


def plot_frames(iom, blockid=0, load=False, limits=None):

    parameters = iom.load_parameters()
    BF = BlockFactory()

    if not parameters["dimension"] == 2:
        print("No wavepacket of two space dimensions, silent return!")
        return

    if load is True:
        # TODO: Implement reshaping
        raise NotImplementedError("Loading of 2D grids is not implemented")
        #G = iom.load_grid(blockid=blockid)
        #G = grid.reshape((1, -1))
    else:
        G = BF.create_grid(parameters)

    u, v = map(squeeze, G.get_axes())

    V = BF.create_potential(parameters)
    BT = BasisTransformationHAWP(V)

    wpd = iom.load_wavepacket_description()
    HAWP = BF.create_wavepacket(wpd)

    # Basis shapes
    BS_descr = iom.load_wavepacket_basisshapes()
    BS = {}
    for ahash, descr in BS_descr.iteritems():
        BS[ahash] = BF.create_basis_shape(descr)

    timegrid = iom.load_wavepacket_timegrid(blockid=blockid)

    N = HAWP.get_number_components()

    for step in timegrid:
        print(" Plotting frame of timestep # " + str(step))

        hi, ci = iom.load_wavepacket_coefficients(timestep=step, get_hashes=True)
        Pi = iom.load_wavepacket_parameters(timestep=step)

        HAWP.set_parameters(Pi)
        HAWP.set_basis_shapes([ BS[int(ha)] for ha in hi ])
        HAWP.set_coefficients(ci)

        # Transform the values to the eigenbasis
        if parameters["algorithm"] == "fourier":
            BT.transform_to_eigen(WF)
        else:
            pass

        psi = HAWP.evaluate_at(G, prefactor=True, component=0)

        fig = figure()

        for level in xrange(N):
            z = psi[level]
            z = z.reshape(G.get_number_nodes())

            subplot(N,1,level+1)
            #plotcm(z.reshape(G.get_number_nodes()), darken=0.3)
            plotcf2d(u, v, z, darken=0.3, limits=limits)

        savefig("wavepacket_level_"+str(level)+"_timestep_"+(5-len(str(step)))*"0"+str(step)+".png")
        close(fig)

    print(" Plotting frames finished")



if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # The axes rectangle that is plotted
    #view = [-3, 3, -3, 3]
    view = None

    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        print("Plotting frames of data block '"+str(blockid)+"'")
        # See if we have wavepacket values
        if iom.has_wavepacket(blockid=blockid):
            plot_frames(iom, blockid=blockid, limits=view)
        else:
            print("Warning: Not plotting any wavepackets in block '"+str(blockid)+"'!")

    iom.finalize()
