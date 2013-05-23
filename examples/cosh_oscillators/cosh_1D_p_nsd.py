algorithm = "hagedorn"
propagator = "semiclassical"
splitting_method = "Y4"

T = 12
dt = 0.001

dimension = 1
ncomponents = 1

eps = 0.01

potential = "cosh_osc"

# The parameter set of the initial wavepacket
Q = [[1.0]]
P = [[1.0j]]
q = [[1.0]]
p = [[0.0]]
S = [[0.0]]

wp0 = {
    "type" : "HagedornWavepacket",
    "dimension" : 1,
    "ncomponents": 1,
    "eps" : 0.01,
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
            "type" : "NSDInhomogeneous",
            'qr': {
                'type': 'GaussLaguerreQR',
                'order': 5,
                'a': -0.5
            }
        }
    }
}

initvals = [ wp0 ]

leading_component = 0

write_nth = 5

matrix_exponential = "pade"


observables = {
    "autocorrelation" : {
        "innerproduct" : {
            "type" : "InhomogeneousInnerProduct",
            "delegate" : {
                "type" : "NSDInhomogeneous",
                'qr': {
                    'type': 'GaussLaguerreQR',
                    'order': 5,
                    'a': -0.5
                }
            }
        }
    }
}
