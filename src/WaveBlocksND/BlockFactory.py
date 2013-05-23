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

            # And the inner product
            if description.has_key("innerproduct"):
                IP = self.create_inner_product(description["innerproduct"])
                WP.set_innerproduct(IP)
            else:
                print("Warning: no inner product specified!")

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
            if description.has_key("innerproduct"):
                IP = self.create_inner_product(description["innerproduct"])
                WP.set_innerproduct(IP)
            else:
                print("Warning: no inner product specified!")

        else:
            raise ValueError("Unknown wavepacket type "+str(wp_type))

        return WP


    def create_inner_product(self, description):
        #
        try:
            ip_type = description["type"]
        except:
            # Default setting
            print("Warning: Using fall-back inner product of type 'InhomogeneousInnerProduct'!")
            # TODO: Maybe change to homogeneous one?
            ip_type = "InhomogeneousInnerProduct"

        if ip_type == "HomogeneousInnerProduct":
            from HomogeneousInnerProduct import HomogeneousInnerProduct
            QE = self.create_quadrature(description["delegate"])
            IP = HomogeneousInnerProduct(QE)

        elif ip_type == "InhomogeneousInnerProduct":
            from InhomogeneousInnerProduct import InhomogeneousInnerProduct
            QE = self.create_quadrature(description["delegate"])
            IP = InhomogeneousInnerProduct(QE)

        else:
            raise ValueError("Unknown inner product type "+str(ip_type))

        return IP


    def create_quadrature(self, description):
        # NOTE: The difference between Quadrature and InnerProduct here
        #       is for backward compatibility but will be removed soon.
        try:
            qe_type = description["type"]
        except:
            # Default setting
            qe_type = "DirectInhomogeneousQuadrature"

        # TODO: Maybe denest QR initialization?
        if qe_type == "DirectHomogeneousQuadrature":
            from DirectHomogeneousQuadrature import DirectHomogeneousQuadrature
            QR = self.create_quadrature_rule(description["qr"])
            QE = DirectHomogeneousQuadrature(QR)

        elif qe_type == "DirectInhomogeneousQuadrature":
            from DirectInhomogeneousQuadrature import DirectInhomogeneousQuadrature
            QR = self.create_quadrature_rule(description["qr"])
            QE = DirectInhomogeneousQuadrature(QR)

        elif qe_type == "NSDInhomogeneous":
            from NSDInhomogeneous import NSDInhomogeneous
            QR = self.create_quadrature_rule(description["qr"])
            QE = NSDInhomogeneous(QR)

        elif qe_type == "SymbolicIntegral":
            from SymbolicIntegral import SymbolicIntegral
            QE = SymbolicIntegral()

        else:
            raise ValueError("Unknown quadrature type "+str(qe_type))

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

        elif qr_type == "GaussLaguerreQR":
            from GaussLaguerreQR import GaussLaguerreQR
            order = description["order"]
            a = description["a"]
            assert type(order) == int
            QR = GaussLaguerreQR(order, a=a, options=op)

        elif qr_type == "TensorProductQR":
            from TensorProductQR import TensorProductQR
            # Iteratively create all quadrature rules necessary
            qrs = [ self.create_quadrature_rule(desc) for desc in description["qr_rules"] ]
            QR = TensorProductQR(qrs, options=op)

        else:
            raise ValueError("Unknown quadrature rule type "+str(qr_type))

        return QR
