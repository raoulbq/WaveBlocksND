algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T = 12
dt = 0.01

dimension = 2
ncomponents = 1

eps = 0.1

potential = "cosh_osc_2d"

# The parameter set of the initial wavepacket
Q = [[1.0, 0.0],
     [0.0, 1.0]]

P = [[1.0j, 0.0 ],
     [0.0,  1.0j]]

q = [[-1.0],
     [ 0.0]]

p = [[0.0],
     [0.1]]

S = [[0.0]]

# What it takes to specify a wavepacket!
wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : 2,
    "ncomponents": 1,
    "eps" : eps,
    "Pi" : [q,p,Q,P,S],
    "basis_shapes" : [{
        "type" : "HyperbolicCutShape",
        "K" : 12,
        "dimension" : 2
    }],
    "coefficients" : [[ ((0,0), 1.0) ]],
    "innerproduct" : {
        "type" : "HomogeneousInnerProduct",
        "delegate" : {
            "type" : "DirectHomogeneousQuadrature",
            'qr': {
                'type': 'TensorProductQR',
                'dimension': 2,
                'qr_rules': [{'dimension': 1, 'order': 40, 'type': 'GaussHermiteQR'},
                             {'dimension': 1, 'order': 40, 'type': 'GaussHermiteQR'}],
            }
        }
    }
}

# Which wavepackets are initial values
initvals = [ wp0 ]

leading_component = 0

# How often do we write data to disk
write_nth = 5

matrix_exponential = "pade"
