algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T =
dt =

dimension = 2
ncomponents = 1

eps =

potential =

# The parameter set of the initial wavepacket
Q = [[1.0, 0.0],
     [0.0, 1.0]]

P = [[1.0j, 0.0 ],
     [0.0,  1.0j]]

q = [[-1.0],
     [ 0.0]]

p = [[0.0],
     [0.0]]

S = [[0.0]]

leading_component = 0

# How often do we write data to disk
write_nth = 1

matrix_exponential = "pade"

# What it takes to specify a wavepacket!
wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : 2,
    "ncomponents": 1,
    "eps" : eps,
    "Pi" : [q,p,Q,P,S],
    "basis_shapes" : [{
        "type" : "HyperbolicCutShape",
        "K" : 8,
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
                'qr_rules': [{'dimension': 1, 'order': 24, 'type': 'GaussHermiteQR'},
                             {'dimension': 1, 'order': 24, 'type': 'GaussHermiteQR'}],
            }
        }
    }
}

# Which wavepackets are initial values
initvals = [ wp0 ]
