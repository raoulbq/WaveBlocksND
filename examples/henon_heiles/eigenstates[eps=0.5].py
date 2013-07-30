dimension = 2
ncomponents = 1

potential = "henon_heiles"

eigenstate_of_level = 0

eigenstates_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

eps = 0.5

hawp_template = {
    "type" : "HagedornWavepacket",
    "dimension" : dimension,
    "ncomponents": 1,
    "eps" : eps,
    "basis_shapes" : [{
            "type" : "HyperCubicShape",
            "limits" : [10,10],
            }]
    }

innerproduct = {
    "type" : "HomogeneousInnerProduct",
    "delegate" : {
        "type" : "DirectHomogeneousQuadrature",
        'qr': {
            'type': 'TensorProductQR',
            'dimension': dimension,
            'qr_rules': [{'dimension': 1, 'order': 15, 'type': 'GaussHermiteQR'},
                         {'dimension': 1, 'order': 15, 'type': 'GaussHermiteQR'}]
        }
    }
}
