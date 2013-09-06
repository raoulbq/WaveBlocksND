"""The WaveBlocks Project

Plot all potentials and put the definitions into a rest file.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2013 R. Bourquin
@license: Modified BSD License
"""

from sympy import *
from numpy import *
from matplotlib.pyplot import *
try:
    # New Mayavi version
    from mayavi import mlab
except ImportError:
    # Older Mayavi versions
    from enthought.mayavi import mlab

from WaveBlocksND import BlockFactory
import WaveBlocksND.PotentialLibrary as PL

# Extract raw information from potential library
# Never do this like here!
potentialnames = filter(lambda i: not i.startswith("_"), dir(PL))
potentialdefs = [ PL.__dict__[p] for p in potentialnames ]
for p, pn in zip(potentialdefs, potentialnames):
    p["name"] = pn

def potential_sorter(pota, potb):
    if len(pota["variables"]) < len(potb["variables"]):
        return -1
    elif len(pota["variables"]) > len(potb["variables"]):
        return 1
    else:
        if pota["number_levels"] < potb["number_levels"]:
            return -1
        elif pota["number_levels"] > potb["number_levels"]:
            return 1
        else:
            return 0

potentials = sorted(potentialdefs, cmp=potential_sorter)

file = open("potentials.rst", "wb")

header = """\
Ready made Potentials
---------------------

The following sections contain some potentials that are implemented in the potential
library. The plots show the eigenvalues or energy surfaces. Some potentials
have additional parameters, the default values for these are also Name.

"""

file.write(header.encode("UTF-8"))

params = {"delta":0.2, "delta1":0.2, "delta2":0.2}

# Grids
N = 5000
x = linspace(-5,5,N)

Nx = 200
Ny = 200
x2 = linspace(-5, 5, Nx)
y2 = linspace(-5, 5, Ny)
X2, Y2 = meshgrid(x2, y2)
G = vstack([X2.reshape(1,-1), Y2.reshape(1, -1)])

for potdef in potentials:
    print("Potential is: " + potdef["name"])

    potvars = map(latex, map(sympify, potdef["variables"]))
    potvars = map(lambda x: ":math:`"+str(x)+"`", potvars)
    potvars = reduce(lambda a,b: a+", "+b, potvars)

    if type(potdef["potential"]) == str:
        potformula = latex(sympify(potdef["potential"]))
    else:
        potformula = latex(Matrix(sympify(potdef["potential"])))

    if potdef.has_key("defaults"):
        potdefaults = potdef["defaults"]
    else:
        potdefaults = {}

    # Create the potential "the right way"
    params["potential"] = potdef
    P = BlockFactory().create_potential(params)

    if len(potdef["variables"]) == 1:
        # Plot the potential
        values = P.evaluate_eigenvalues_at(x)

        figure(figsize=(4,3))
        for value in values:
            plot(squeeze(x), squeeze(value))
        grid(True)
        xlabel(r"$x$")
        ylabel(r"$\lambda_i\left(x\right)$")
        xlim(min(x), max(x))
        savefig(potdef["name"] + ".png")

    elif len(potdef["variables"]) == 2:
        values = P.evaluate_eigenvalues_at(G)

        f = mlab.figure()
        for value in values:
            mlab.surf(x2, y2, real(value.reshape(Nx, Ny)))
        mlab.savefig(potdef["name"] + ".png")
        mlab.close(f)

    # The latex code
    ls = []
    l = "Potential ``" + str(potdef["name"]) + "``\n"
    ls.append(l)
    ls.append((len(l)-1)*"^")
    ls.append("\n\n")
    ls.append("* Formula: :math:`V(x) = " + potformula + "`\n")
    ls.append("\n")
    ls.append("* Variables: " + potvars + "\n")
    ls.append("\n")

    if len(potdefaults) > 0:
        ls.append("* Default values:\n")
        ls.append("\n")

        for key, value, in potdefaults.iteritems():
           ls.append("  * :math:`" + latex(sympify(key)) + " = " + str(value) + "`\n")

    ls.append("\n")

    if len(potdef["variables"]) < 3:
        ls.append(".. image:: fig/" + potdef["name"] + ".png\n\n")

    ls = reduce(lambda x,y: x+y, ls)
    ls.encode("UTF-8")
    file.write(ls)

file.close()
