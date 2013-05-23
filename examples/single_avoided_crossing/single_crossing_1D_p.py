algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T = 10
dt = 0.01

dimension = 1
ncomponents = 2

eps = 0.2
delta = eps

potential = "delta_gap"

leading_component = 0

# The parameter set of the initial wavepacket
Q = [[1.0-5.0j]]
P = [[1.0j]]
q = [[-5.0]]
p = [[ 1.0]]
S = [[0.0]]

# What it takes to specify a wavepacket!
wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : dimension,
    "ncomponents": ncomponents,
    "eps" : eps,
    "Pi" : [q,p,Q,P,S],
    "basis_shapes" : [{
            "type" : "HyperbolicCutShape",
            "K" : 64,
            "dimension" : 1
            },
            {
            "type" : "HyperbolicCutShape",
            "K" : 64,
            "dimension" : 1
            }],
    "coefficients" : [[ ((0,), 1.0) ],
                      [ ((0,), 0.0) ]
                      ],
    "innerproduct" : {
        "type" : "HomogeneousInnerProduct",
        "delegate" : {
            "type" : "DirectHomogeneousQuadrature",
            'qr': {
                'type': 'TensorProductQR',
                'dimension': 1,
                'qr_rules': [{'dimension': 1, 'order': 68, 'type': 'GaussHermiteQR'}]
            }
        }
    }
}

# Which wavepackets are initial values
initvals = [ wp0 ]

# How often do we write data to disk
write_nth = 2

matrix_exponential = "pade"
