algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T = 12
dt = 0.01

dimension = 2
ncomponents = 1

eps = 1.0

potential = "quadratic_2d"

# The parameter set of the initial wavepacket
# Parameter values computed by 'ComputeGroundstate.py'
Q = [[ 1.18920712+0.j,  0.00000000+0.j],
     [ 0.00000000+0.j,  1.18920712+0.j]]

P = [[ 0.+0.84089642j,  0.+0.j        ],
     [ 0.+0.j        ,  0.+0.84089642j]]

q = [[ 9.90555981e-14],
     [-4.51334304e-13]]

p = [[0.0],
     [0.0]]

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
        "K" : 4,
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
                'qr_rules': [{'dimension': 1, 'order': 8, 'type': 'GaussHermiteQR'},
                             {'dimension': 1, 'order': 8, 'type': 'GaussHermiteQR'}],
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
