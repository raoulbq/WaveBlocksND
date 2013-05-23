algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

ICH3 = {}
ICH3["variables"] = ["x", "y", "z"]
ICH3["potential"] = [["C1 + a0*x+a1*y+a2*z + (al0**2*x**2 + al1**2*y**2 + al2**2*z**2)/2 + C2 + b0*x+b1*y+b2*z",        "be*z"],
                     ["be*z",        "C1 + a0*x+a1*y+a2*z + (al0**2*x**2 + al1**2*y**2 + al2**2*z**2)/2 - C2 - b0*x+b1*y+b2*z"]]
ICH3["defaults"] = {
    "C1" : 4.39,
    "C2" : -0.45,
    "a0" : -0.521,
    "a1" : 0.081,
    "a2" : 0.0,
    "b0" : 0.698,
    "b1" : -0.467,
    "b2" : 0.0,
    "al0" : 1.703,
    "al1" : 1.0,
    "al2" : 1.595,
    "be" : 1.216
    }

potential = ICH3

dimension = 3
ncomponents = 2
leading_component = 0

# hbar = 0.074
eps = 0.27203

# Time [0, 500] femto seconds, 200'00 timesteps
#T = 28
T = 0.1
dt = 0.00014

# Parameter values: sqrt(alpha_j)
sa1 = 1.305
sa2 = 1.0
sa3 = 1.2629

Q = [[1.0/sa1, 0.0,     0.0    ],
     [0.0,     1.0/sa2, 0.0    ],
     [0.0,     0.0,     1.0/sa3]]

P = [[1.0j/sa1, 0.0,      0.0     ],
     [0.0,      1.0j/sa2, 0.0     ],
     [0.0,      0.0,      1.0j/sa3]]

q = [[0.0],
     [0.0],
     [0.0]]

p = [[0.0],
     [0.0],
     [0.0]]

S = [[0.0]]

bsize = 8

wp0 = {
    'type' : 'HagedornWavepacket',
    'dimension' : dimension,
    'ncomponents': ncomponents,
    'eps' : eps,
    'Pi' : [q,p,Q,P,S],
    'basis_shapes' : [
        {
            'type' : 'HyperbolicCutShape',
            'K' : bsize,
            'dimension' : dimension
        },
        {
            'type' : 'HyperbolicCutShape',
            'K' : bsize,
            'dimension' : dimension
        }
    ],
    'coefficients' : [[ ((0,0,0), 1.0) ],
                      [ ((0,0,0), 0.0) ]],
    "innerproduct" : {
        "type" : "HomogeneousInnerProduct",
        "delegate" : {
            "type" : "DirectHomogeneousQuadrature",
            'qr': {
                'type': 'TensorProductQR',
                'dimension': 3,
                'qr_rules': [{'dimension': 1, 'order': 24, 'type': 'GaussHermiteQR'},
                             {'dimension': 1, 'order': 24, 'type': 'GaussHermiteQR'},
                             {'dimension': 1, 'order': 24, 'type': 'GaussHermiteQR'}],
            }
        }
    }
}

initvals = [ wp0 ]

matrix_exponential = "arnoldi"
write_nth = 10
