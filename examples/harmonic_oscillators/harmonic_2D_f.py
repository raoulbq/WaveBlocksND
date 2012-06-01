algorithm = "fourier"

T = 12
dt = 0.01

dimension = 2
ncomponents = 1

eps = 0.1

potential = "quadratic_2d"

# The grid of our simulation domain
limits = [(-6.283185307179586, 6.283185307179586), (-6.283185307179586, 6.283185307179586)]
number_nodes = [512, 512]

# The parameter set of the initial wavepacket
Q = [[1.0, 0.0],
     [0.0, 1.0]]

P = [[1.0j, 0.0 ],
     [0.0,  1.0j]]

q = [[-3.0],
     [ 0.0]]

p = [[0.0],
     [0.5]]

S = [[0.0]]

# What it takes to specify a wavepacket!
wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : 2,
    "ncomponents": 1,
    "eps" : 0.1,
    "Pi" : [q,p,Q,P,S],
    "basis_shapes" : [{
        "type" : "HyperbolicCutShape",
        "K" : 4,
        "dimension" : 2
    }],
    "coefficients" : [[ ((0,0), 1.0) ]],
    "quadrature" : {
        "type" : "HomogeneousQuadrature",
	'qr': {
            'type': 'TensorProductQR',
            'dimension': 2,
            'qr_rules': [{'dimension': 1, 'order': 8, 'type': 'GaussHermiteQR'},
                         {'dimension': 1, 'order': 8, 'type': 'GaussHermiteQR'}],
        }
    }
}

# Which wavepackets are initial values
initvals = [ wp0 ]

# How often do we write data to disk
# WARNING: Produces about 1GB of data!
write_nth = 5
