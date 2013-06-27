"""The WaveBlocks Project

This file contains the class which represents a homogeneous Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import complexfloating, squeeze, real, ones, zeros_like, conj, sum, argsort
from scipy.optimize import fmin
from scipy.linalg import sqrtm, inv, eigh

from WaveBlocksND import BlockFactory
from WaveBlocksND import ParameterLoader
from WaveBlocksND import GradientHAWP
from WaveBlocksND import IOManager


def compute_eigenstate(params):
    r"""
    Special variables necessary in configuration:

    * eigenstate_of_level (default: 0)
    * states_indices (default: [0])
    """
    D = params["dimension"]

    if params.has_key("eigenstate_of_level"):
        N = params["eigenstate_of_level"]
    else:
        # Upper-most potential surface
        N = 0

    # Create output file now, in case this fails we did not waste computations
    IOM = IOManager()
    IOM.create_file(params, filename="eigenstates.hdf5")
    gid = IOM.create_group()

    BF = BlockFactory()
    # Create the potential
    V = BF.create_potential(params)
    V.calculate_local_quadratic()

    # Minimize the potential to find q0
    f = lambda x: real((squeeze(V.evaluate_at(x)[N])))
    # Start with an offset because exact 0.0 values can give
    # issues, especially with the Hessian evaluation. This way
    # the minimizer will always stay away from zero a tiny bit.
    # The current starting point can give issues if the potential
    # is stationary at the point (2, ..., 2) but that is less likely.
    x0 = 2.0*ones(D)
    q0 = fmin(f, x0, xtol=1e-12)
    q0 = q0.reshape((D,1))

    # We are at the minimum with no momentum
    p0 = zeros_like(q0)

    # Compute spreads now
    # Q_0 = H^(-1/4)
    H = V.evaluate_hessian_at(q0)
    Q0 = inv(sqrtm(sqrtm(H)))
    # Take P_00 = i Q_0^(-1)
    P0 = 1.0j * inv(Q0)

    #
    print(70*"-")
    print("Parameter values are:")
    print("---------------------")
    print(" q0:")
    print(str(q0))
    print(" p0:")
    print(str(p0))
    print(" Q0:")
    print(str(Q0))
    print(" P0:")
    print(str(P0))
    # Consistency check
    print(" consistency:")
    print(str(conj(Q0)*P0 - conj(P0)*Q0))
    print(70*"-")

    # Next find the new coefficients c'
    HAWP = BF.create_wavepacket(params["hawp_template"])

    # Set the parameter values
    Pi = HAWP.get_parameters()
    Pi[0] = q0
    Pi[1] = p0
    Pi[2] = Q0
    Pi[3] = P0
    HAWP.set_parameters(Pi)

    # Next compute the matrix M_ij = <phi_i | T + V | phi_j>
    # The potential part
    HQ = BF.create_inner_product(params["innerproduct"])

    opV = lambda x, q, entry: V.evaluate_at(x, entry=entry)
    MV = HQ.build_matrix(HAWP, operator=opV)

    # The kinetic part
    MT = zeros_like(MV, dtype=complexfloating)
    GR = GradientHAWP()
    BS = HAWP.get_basis_shapes(N)

    vects = {}
    for i in BS:
        z = zeros_like(HAWP.get_coefficient_vector(), dtype=complexfloating)
        HAWP.set_coefficient_vector(z)
        HAWP.set_coefficient(N, i, 1.0)
        Kn, cnew = GR.apply_gradient(HAWP, N)
        vects[i] = cnew

    for j in BS:
        for k in BS:
            cj = vects[j]
            ck = vects[k]
            entry = 0.5 * squeeze(sum(conj(cj) * ck))
            MT[BS[j], BS[k]] = entry

    # Find eigenvalues and eigenvectors of the whole matrix
    M = MT + MV
    ew, ev = eigh(M)
    ind = argsort(ew)

    # Build the requested energy levels and states
    if params.has_key("eigenstates_indices"):
        states = params["eigenstates_indices"]
    else:
        # Groundstate only
        states = [0]

    BS = HAWP.get_basis_shapes(component=0)

    KEY = ("q","p","Q","P","S","adQ")

    print(70*"-")
    for state in states:
        if state > BS.get_basis_size():
            print("Warning: can not compute energy level "+state+" with basis size of "+str(BS))
            continue

        index = ind[state]

        coeffs = ev[:,index]
        energy = ew[index]

        print("Level: "+str(state))
        print("Energy: "+str(energy))
        print("Coefficients: \n")
        print(str(coeffs))
        print(70*"-")

        HAWP.set_coefficient_vector(coeffs.reshape((-1, 1)))

        # Save all the wavepacket data
        bid = IOM.create_block(groupid=gid)
        IOM.add_wavepacket(params, blockid=bid, key=KEY)
        IOM.save_wavepacket_description(HAWP.get_description(), blockid=bid)
        for shape in HAWP.get_basis_shapes():
            IOM.save_wavepacket_basisshapes(shape, blockid=bid)
        IOM.save_wavepacket_parameters(HAWP.get_parameters(key=KEY), timestep=0, blockid=bid, key=KEY)
        IOM.save_wavepacket_coefficients(HAWP.get_coefficients(), HAWP.get_basis_shapes(), timestep=0, blockid=bid)

    IOM.finalize()




if __name__ == "__main__":

    # Read the path for the configuration file we use for this simulation.
    try:
        parametersfile = sys.argv[1]
    except IndexError:
        raise ValueError("No configuration file given!")

    print("Using configuration from file: " + parametersfile)

    # Set up the parameter provider singleton
    PA = ParameterLoader().load_from_file(parametersfile)
    compute_eigenstate(PA)

    print("Eigenstate computation finished")
