"""The WaveBlocks Project

This file contains some ready made potentials in several variables
and with several separate energy levels. This is a pure data file
without any code. To load the potentials, use the :py:class:`PotentialFactory`.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

####################################################################
# Potentials in one dimension and with one energy level (D=1, N=1) #
####################################################################

# Free particle
free_particle = {}
free_particle["variables"] = ["x"]
free_particle["potential"] = "c"
free_particle["defaults"] = {"c":"0"}

# Simple harmonic potential 1D
quadratic = {}
quadratic["variables"] = ["x"]
quadratic["potential"] = "1/2 * sigma * x**2"
quadratic["defaults"] = {"sigma":"1/2"}

# Perturbed harmonic potential
pert_quadratic = {}
pert_quadratic["variables"] = ["x"]
pert_quadratic["potential"] = "1/2 * sigma * x**2 + 1/2 * delta**2 * x**2"
pert_quadratic["defaults"] = {"sigma":0.05, "delta":0.2}

# A simple fourth order anharmonic potential
quartic = {}
quartic["variables"] = ["x"]
quartic["potential"] = "1/4 * sigma * x**4"
quartic["defaults"] = {"sigma":0.05}

# A potential consisting of a cosine wave
cos_osc = {}
cos_osc["variables"] = ["x"]
cos_osc["potential"] = "a * (1 - cos(b*x))"
cos_osc["defaults"] = {"a":0.07, "b":1.0}

# A potential consisting of a hyperbolic cosine
cosh_osc = {}
cosh_osc["variables"] = ["x"]
cosh_osc["potential"] = "a * cosh(b * x)"
cosh_osc["defaults"] = {"a":"1", "b":"1"}

# The Morse potential
morse = {}
morse["variables"] = ["x"]
morse["potential"] = "D * (1 - exp(-a*(x-x0)))**2"
morse["defaults"] = {"D":3.0, "a":0.3, "x0":0.0}

# A double well potential
double_well = {}
double_well["variables"] = ["x"]
double_well["potential"] = "sigma * (x**2 - 1)**2"
double_well["defaults"] = {"sigma":1.0}

# The Eckart potential
eckart = {}
eckart["variables"] = ["x"]
eckart["potential"] = "sigma * cosh(x/a)**(-2)"
eckart["defaults"] = {"sigma":100*3.8088*10**(-4), "a":1.0/(2.0*0.52918)}

# A smooth unitstep like wall
wall = {}
wall["variables"] = ["x"]
wall["potential"] = "atan(sigma*x) + pi/2"
wall["defaults"] = {"sigma":10.0}

# A narrow 'V'-like potential
v_shape = {}
v_shape["variables"] = ["x"]
v_shape["potential"] = "1/2 * sqrt(tanh(x)**2+4*delta**2)"
v_shape["defaults"] = {"delta":0.2}


#####################################################################
# Potentials in two dimensions and with one energy level (D=2, N=1) #
#####################################################################

# Simple harmonic potential 2D
quadratic_2d = {}
quadratic_2d["variables"] = ["x", "y"]
quadratic_2d["potential"] = "1/2 * (sigmax * x**2 + sigmay * y**2)"
quadratic_2d["defaults"] = {"sigmax":"1/2", "sigmay":"1/2"}

# A potential consisting of a cosine wave part in 2D
cos_osc_2d = {}
cos_osc_2d["variables"] = ["x", "y"]
cos_osc_2d["potential"] = "ax * (1 - cos(bx*x)) + ay * (1 - cos(by*y))"
cos_osc_2d["defaults"] = {"ax":"1", "bx":"1", "ay":"1", "by":"1"}

# A potential consisting of a hyperbolic cosine
cosh_osc_2d = {}
cosh_osc_2d["variables"] = ["x", "y"]
cosh_osc_2d["potential"] = "a * cosh(b * sqrt(x**2+y**2))"
cosh_osc_2d["defaults"] = {"a":"1", "b":"1"}

# A potential consisting of a hyperbolic cosine
corral_rotsym_2d = {}
corral_rotsym_2d["variables"] = ["x", "y"]
corral_rotsym_2d["potential"] = "atan(sigma*(sqrt(x**2+y**2) - R)) + pi/2"
corral_rotsym_2d["defaults"] = {"sigma":"10", "R":"8"}

# A potential consisting of circular pit of radius R
circle_pit_2d = {}
circle_pit_2d["variables"] = ["x", "y"]
circle_pit_2d["potential"] = "atan(sigma*(sqrt(x**2+y**2) - R)) + pi/2"
circle_pit_2d["defaults"] = {"sigma":"10", "R":"8"}

# A potential consisting of a ring like corral
corral_ring = {}
corral_ring["variables"] = ["x", "y"]
corral_ring["potential"] = "sqrt(delta**2 + tanh(sqrt(x**2 + y**2) - R)**2*tanh(sqrt(x**2 + y**2) + R)**2)/2"
corral_ring["defaults"] = {"delta":"1/32", "R":"3"}




#######################################################################
# Potentials in three dimensions and with one energy level (D=3, N=1) #
#######################################################################

# Simple harmonic potential 3D
quadratic_3d = {}
quadratic_3d["variables"] = ["x", "y", "z"]
quadratic_3d["potential"] = "1/2 * (sigmax * x**2 + sigmay * y**2 + sigmaz * z**2)"
quadratic_3d["defaults"] = {"sigmax":"1/2", "sigmay":"1/2", "sigmaz":"1/2"}




#####################################################################
# Potentials in one dimension and with two energy levels (D=1, N=2) #
#####################################################################

# Double harmonic potential for two components
two_quadratic = {}
two_quadratic["variables"] = ["x"]
two_quadratic["potential"] = [["1/2*sigma*x**2", "0"             ],
                              ["0",              "1/2*sigma*x**2"]]
two_quadratic["defaults"] = {"sigma":0.05}

# Double quartic anharmonic potential for two components
two_quartic = {}
two_quartic["variables"] = ["x"]
two_quartic["potential"] = [["1/4*sigma*x**4", "0"             ],
                            ["0",              "1/8*sigma*x**4"]]
two_quartic["defaults"] = {"sigma":"1"}

# A potential with a single avoided crossing
delta_gap = {}
delta_gap["variables"] = ["x"]
delta_gap["potential"] = [["1/2 * tanh(x)", "delta"         ],
                          ["delta",         "-1/2 * tanh(x)"]]

# Diagonalized single avoided crossing
delta_gap_diag = {}
delta_gap["variables"] = ["x"]
delta_gap_diag["potential"] = [["sqrt(delta**2 + tanh(x)**2/4)", "0"                             ],
                               ["0",                             "-sqrt(delta**2 + tanh(x)**2/4)"]]

# A potential with two avoided crossings in series
two_crossings = {}
two_crossings["variables"] = ["x"]
two_crossings["potential"] = [["tanh(x-rho)*tanh(x+rho)/2", "delta/2"                   ],
                              ["delta/2",                   "-tanh(x-rho)*tanh(x+rho)/2"]]
two_crossings["defaults"] = {"rho":3.0}


######################################################################
# Potentials in two dimensions and with two energy levels (D=2, N=2) #
######################################################################

delta_gap_rotsym = {}
delta_gap_rotsym["variables"] = ["x", "y"]
delta_gap_rotsym["potential"] = [["tanh(sqrt(x**2 + y**2))/2",                      "delta"],
                                 ["delta"                    , "-tanh(sqrt(x**2 + y**2))/2"]]

conic = {}
conic["variables"] = ["x", "y"]
conic["potential"] = [["x",  "y"],
                      ["y", "-x"]]

conic_gap = {}
conic_gap["variables"] = ["x", "y"]
conic_gap["potential"] = [["x",           "y + I*delta"],
                          ["y - I*delta", "-x"         ]]


########################################################################
# Potentials in three dimensions and with two energy levels (D=3, N=2) #
########################################################################




#######################################################################
# Potentials in one dimension and with three energy levels (D=1, N=3) #
#######################################################################

# Decoupled harmonic potentials for three components
three_quadratic = {}
three_quadratic["variables"] = ["x"]
three_quadratic["potential"] = [["1/2 * sigma * x**2", "0",                  "0"                 ],
                                ["0",                  "1/2 * sigma * x**2", "0"                 ],
                                ["0",                  "0",                  "1/2 * sigma * x**2"]]
three_quadratic["defaults"] = {"sigma":0.05}

# A potential with three energy levels and multiple crossings
three_levels = {}
three_levels["variables"] = ["x"]
three_levels["potential"] = [["tanh(x+rho) + tanh(x-rho)", "delta1",       "delta2"         ],
                             ["delta1",                    "-tanh(x+rho)", "0"              ],
                             ["delta2",                    "0",            "1 - tanh(x-rho)"]]
three_levels["defaults"] = {"rho":3.0}


########################################################################
# Potentials in two dimensions and with three energy levels (D=2, N=3) #
########################################################################


##########################################################################
# Potentials in three dimensions and with three energy levels (D=3, N=3) #
##########################################################################




######################################################################
# Potentials in one dimension and with four energy levels (D=1, N=4) #
######################################################################

# Decoupled harmonic potentials for four components
four_quadratic = {}
four_quadratic["variables"] = ["x"]
four_quadratic["potential"] = [["1/2 * sigma * x**2", "0",                  "0",                  "0"                 ],
                               ["0",                  "1/2 * sigma * x**2", "0",                  "0"                 ],
                               ["0",                  "0",                  "1/2 * sigma * x**2", "0"                 ],
                               ["0",                  "0",                  "0",                  "1/2 * sigma * x**2"]]
four_quadratic["defaults"] = {"sigma":0.05}

# Harmonic and higher order anharmonic potentials for four components
four_powers = {}
four_powers["variables"] = ["x"]
four_powers["potential"] = [["1/2 * sigma * x**2", "0",                  "0",                  "0"                 ],
                            ["0",                  "1/4 * sigma * x**4", "0",                  "0"                 ],
                            ["0",                  "0",                  "1/6 * sigma * x**6", "0"                 ],
                            ["0",                  "0",                  "0",                  "1/8 * sigma * x**8"]]
four_powers["defaults"] = {"sigma":0.05}




######################################################################
# Potentials in one dimension and with five energy levels (D=1, N=5) #
######################################################################

# Decoupled harmonic potential for five components
five_quadratic = {}
five_quadratic["variables"] = ["x"]
five_quadratic["potential"] = [["1/2 * sigma * x**2", "0",                  "0",                  "0",                  "0"                 ],
                               ["0",                  "1/2 * sigma * x**2", "0",                  "0",                  "0"                 ],
                               ["0",                  "0",                  "1/2 * sigma * x**2", "0",                  "0"                 ],
                               ["0",                  "0",                  "0",                  "1/2 * sigma * x**2", "0"                 ],
                               ["0",                  "0",                  "0",                  "0",                  "1/2 * sigma * x**2"]]
five_quadratic["defaults"] = {"sigma":0.05}
