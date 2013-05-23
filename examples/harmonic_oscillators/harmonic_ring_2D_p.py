algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T = 10
dt = 0.01

dimension = 2
ncomponents = 1

eps = 0.1

# A ring like potential
potential = {}
potential["potential"] = [["(sqrt(x**2+y**2) -R)**2"]]
potential["variables"] = ["x", "y"]

R = 4

# The parameter set of the initial wavepacket
Q = [[1.0, 0.0],
     [0.0, 1.0]]

P = [[1.0j, 0.0 ],
     [0.0,  1.0j]]

q = [[ 4.2],
     [ 0.0]]

p = [[0.0],
     [0.2]]

S = [[0.0]]

# What it takes to specify a wavepacket!
wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : dimension,
    "ncomponents": 1,
    "eps" : eps,
    "Pi" : [q,p,Q,P,S],
    "basis_shapes" : [{
        "type" : "HyperbolicCutShape",
        "K" : 16,
        "dimension" : dimension
    }],
    "coefficients" : [[ ((0,0), 1.0) ]],
    "innerproduct" : {
        "type" : "HomogeneousInnerProduct",
        "delegate" : {
            "type" : "DirectHomogeneousQuadrature",
            'qr': {
                'type': 'TensorProductQR',
                'dimension': dimension,
                'qr_rules': [{'dimension': 1, 'order': 20, 'type': 'GaussHermiteQR'},
                             {'dimension': 1, 'order': 20, 'type': 'GaussHermiteQR'}],
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
