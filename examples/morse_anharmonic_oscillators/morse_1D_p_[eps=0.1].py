from WaveBlocksND import LoadFromFile as LFF

algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T = 12
dt = 0.01

dimension = 1
ncomponents = 1

eps = 0.1

potential ="morse_zero_2"
l = 1.0
x0 = 0.0

# The parameter set of the initial wavepacket
PI, C = LFF.load_from_file("groundstate.hdf5")

# What it takes to specify a wavepacket!
wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : 1,
    "ncomponents": 1,
    "eps" : eps,
    "Pi" : PI,
    "basis_shapes" : [{
        "type" : "HyperbolicCutShape",
        "K" : 32,
        "dimension" : 1
    }],
# Coefficient values computed by 'ComputeGroundstate.py'
    "coefficients" : C,
    "innerproduct" : {
        "type" : "HomogeneousInnerProduct",
        "delegate" : {
            "type" : "DirectHomogeneousQuadrature",
            'qr': {
                'type': 'TensorProductQR',
                'dimension': 1,
                'qr_rules': [{'dimension': 1, 'order': 36, 'type': 'GaussHermiteQR'}]
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
