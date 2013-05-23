dimension = 1
ncomponents = 1

potential = "morse_zero_2"
l = 1.0
x0 = 0.0

eigenstate_of_level = 0

eps = 0.1

hawp_template = {
    "type" : "HagedornWavepacket",
    "dimension" : dimension,
    "ncomponents": 1,
    "eps" : eps,
    "basis_shapes" : [{
            "type" : "HyperbolicCutShape",
            "K" : 32,
            "dimension" : 1
            }]
    }

innerproduct = {
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
