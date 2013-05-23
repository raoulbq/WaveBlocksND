algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T = 12
dt = 0.01

dimension = 1
ncomponents = 1

eps = 0.24

potential = "quadratic"
sigma = 0.5

# The parameter set of the initial wavepacket
# Parameter values computed by 'ComputeGroundstate.py'
Q = [[1.18920712 + 0.0j]]
P = [[0.0 + 0.84089642j]]
q = [[0.0]]
p = [[0.0]]
S = [[0.0]]

# What it takes to specify a wavepacket!
wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : 1,
    "ncomponents": 1,
    "eps" : eps,
    "Pi" : [q,p,Q,P,S],
    "basis_shapes" : [{
        "type" : "HyperbolicCutShape",
        "K" : 10,
        "dimension" : 1
    }],
    "coefficients" : [[ ((0,), 1.0) ]],
    "innerproduct" : {
        "type" : "HomogeneousInnerProduct",
        "delegate" : {
            "type" : "DirectHomogeneousQuadrature",
            'qr': {
                'type': 'TensorProductQR',
                'dimension': 1,
                'qr_rules': [{'dimension': 1, 'order': 14, 'type': 'GaussHermiteQR'}]
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
