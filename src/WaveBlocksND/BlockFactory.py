"""The WaveBlocks Project

This file contains a the block factory.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

from copy import deepcopy

__all__ = ["BlockFactory"]


class BlockFactory(object):
    """A factory to create instances of various classes
    based on a simple description ``dict``.
    """

    def __init__(self):

        # Load different factory methods
        import GridFactory
        self.__dict__["create_grid"] = GridFactory.create_grid

        import PotentialFactory
        self.__dict__["create_potential"] = PotentialFactory.create_potential

        import MatrixExponentialFactory
        self.__dict__["create_matrixexponential"] = MatrixExponentialFactory.create_matrixexponential


    # TODO: Consider "local" vs "global" description dicts
    # TODO: Consider putting defaults into "GlobalDefaults"


    def create_basis_shape(self, description):
        try:
            bs_type = description["type"]
        except:
            # Default setting
            bs_type = "HyperCubicShape"

        if bs_type == "HyperCubicShape":
            from HyperCubicShape import HyperCubicShape
            limits = description["limits"]
            BS = HyperCubicShape(limits)

        elif bs_type == "HyperbolicCutShape":
            from HyperbolicCutShape import HyperbolicCutShape
            K = description["K"]
            D = description["dimension"]
            BS = HyperbolicCutShape(D, K)

        else:
            raise ValueError("Unknown basis shape type "+str(bs_type))

        return BS


    def create_wavepacket(self, description):

        wp_type = description["type"]

        if wp_type == "HagedornWavepacket":
            from HagedornWavepacket import HagedornWavepacket

            # Initialize a packet
            WP = HagedornWavepacket(description["dimension"],
                                    description["ncomponents"],
                                    description["eps"])

            # Set parameters
            if description.has_key("Pi"):
                Pi = description["Pi"]
                WP.set_parameters(Pi)

            # Configure basis shapes
            if description.has_key("basis_shapes"):
                for component, shapedescr in enumerate(description["basis_shapes"]):
                    BS = self.create_basis_shape(shapedescr)
                    WP.set_basis_shapes(BS, component=component)

            # Set coefficients
            if description.has_key("coefficients"):
                for component, data in enumerate(description["coefficients"]):
                    BS = WP.get_basis_shapes(component=component)
                    for index, value in data:
                        if BS.contains(index):
                            WP.set_coefficient(component, index, value)
                        else:
                            print("Warning: dropped coefficient with index "+str(index))

            # And the quadrature
            if description.has_key("quadrature"):
                QE = self.create_quadrature(description["quadrature"])
                WP.set_quadrature(QE)
            else:
                print("Warning: no quadrature specified!")

        elif wp_type == "HagedornWavepacketInhomogeneous":
            from HagedornWavepacketInhomogeneous import HagedornWavepacketInhomogeneous

            # Initialize a packet
            WP = HagedornWavepacketInhomogeneous(description["dimension"],
                                    description["ncomponents"],
                                    description["eps"])

            # Set parameters
            if description.has_key("Pi"):
                Pi = description["Pi"]
                WP.set_parameters(Pi)

            # Configure basis shapes
            if description.has_key("basis_shapes"):
                for component, shapedescr in enumerate(description["basis_shapes"]):
                    BS = self.create_basis_shape(shapedescr)
                    WP.set_basis_shapes(BS, component=component)

            # Set coefficients
            if description.has_key("coefficients"):
                for component, data in enumerate(description["coefficients"]):
                    for index, value in data:
                        WP.set_coefficient(component, index, value)

            # And the quadrature
            if description.has_key("quadrature"):
                QE = self.create_quadrature(description["quadrature"])
                WP.set_quadrature(QE)
            else:
                print("Warning: no quadrature specified!")

        else:
            raise ValueError("Unknown wavepacket type "+str(wp_type))

        return WP


    def create_quadrature(self, description):
        try:
            qe_type = description["type"]
        except:
            # Default setting
            qe_type = "InhomogeneousQuadrature"

        # TODO: Maybe denest QR initialization?
        if qe_type == "HomogeneousQuadrature":
            from HomogeneousQuadrature import HomogeneousQuadrature
            QR = self.create_quadrature_rule(description["qr"])
            QE = HomogeneousQuadrature(QR)

        elif qe_type == "InhomogeneousQuadrature":
            from InhomogeneousQuadrature import InhomogeneousQuadrature
            QR = self.create_quadrature_rule(description["qr"])
            QE = InhomogeneousQuadrature(QR)

        else:
            raise ValueError("Unknown basis shape type "+str(qe_type))

        return QE


    def create_quadrature_rule(self, description):
        qr_type = description["type"]

        if description.has_key("options"):
            op = deepcopy(description["options"])
        else:
            # Per default, adapt qr to follow dynamics
            op = {"transform":True}

        if qr_type == "GaussHermiteQR":
            from GaussHermiteQR import GaussHermiteQR
            order = description["order"]
            assert type(order) == int
            QR = GaussHermiteQR(order, options=op)

        elif qr_type == "TrapezoidalQR":
            from TrapezoidalQR import TrapezoidalQR
            left = description["left"]
            right = description["right"]
            order = description["order"]
            assert type(order) == int
            QR = TrapezoidalQR(left, right, order, options=op)

        elif qr_type == "TensorProductQR":
            from TensorProductQR import TensorProductQR
            # Iteratively create all quadrature rules necessary
            qrs = [ self.create_quadrature_rule(desc) for desc in description["qr_rules"] ]
            QR = TensorProductQR(qrs, options=op)

        return QR
