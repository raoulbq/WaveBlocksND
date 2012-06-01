algorithm = "hagedorn"

T = 10
dt = 0.01

dimension = 2
ncomponents = 1

eps = 0.1

potential = {}
potential["potential"] = "0.25 * y**2"
potential["variables"] = ["x", "y"]

# The parameter set of the initial wavepacket
Q = [[1.0, 0.0],
     [0.0, 1.0]]

P = [[1.0j, 0.0 ],
     [0.0,  1.0j]]

q = [[0.0],
     [1.0]]

p = [[0.0],
     [0.0]]

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
            'qr_rules': [
		{'dimension': 1, 'order': 8, 'type': 'GaussHermiteQR'},
		{'dimension': 1, 'order': 8, 'type': 'GaussHermiteQR'}],
        }
    }
}

# Which wavepackets are initial values
initvals = [ wp0 ]

leading_component = 0

# How often do we write data to disk
write_nth = 1

matrix_exponential = "pade"
