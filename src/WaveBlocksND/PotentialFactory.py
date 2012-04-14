"""The WaveBlocks Project

This file contains a simple factory for MatrixPotential instances. The exact
subtype of the instance is derived from the potentials' symbolic expression.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sympy

__all__ = ["PotentialFactory"]


class PotentialFactory(object):
    """A factory for :py:class:`MatrixPotential` instances. We decide which subclass of the
    abstract base class :py:class:`MatrixPotential` to instantiate according to the size
    of the potential's matrix. For a :math:`1 \\times 1` matrix we can use the class
    :py:class:`MatrixPotential1S` which implements simplified scalar symbolic calculations.
    In the case of a :math:`2 \\times 2` matrix we use the class :py:class:`MatrixPotential2S`
    that implements the full symbolic calculations for matrices. And for matrices of size
    bigger than :math:`2 \\times 2` symbolic calculations are unfeasible and we have to
    fall back to pure numerical methods implemented in :py:class:`MatrixPotentialMS`.
    """

    def __init__(self):
        pass


    def create_potential(self, parameters):
        """The method that creates a :py:class:`MatrixPotential` instance and decides
        which subclass to instantiate depending on the given potential expression.

        :param parameters: A :py:class:`ParameterProvider` instance with all necessary
                           parameters (at least a ``potential`` entry).

        :return: An adequate :py:class:`MatrixPotential` instance.

        :raises: :py:class:`ValueError` In case of various input error, f.e. if the potential can
                 not be found or if the potential matrix is not square etc.
        """
        # The potential reference given in the parameter provider.
        # This may be a string which is the common name of the potential
        # or a full potential description. In the first case we try
        # to find the referenced potential in the potential library
        # while in the second one, we can omit this step.
        potential_reference = parameters["potential"]

        if type(potential_reference) == str:
            # Try to load the potential from the library
            import PotentialLibrary as PL
            if PL.__dict__.has_key(potential_reference):
                potential_description = PL.__dict__[potential_reference]
            else:
                raise ValueError("Unknown potential " + potential_reference + " requested from library.")
        elif type(potential_reference) == dict:
            # The potential reference given in the parameter provider was a full description
            potential_description = potential_reference
        else:
            raise ValueError("Invalid potential reference.")

        # The symbolic expression strings of the potential
        pot = potential_description["potential"]

        # Potential is just one level, wrap it into a matrix
        if type(pot) == str:
            pot = [[pot]]

        if not all([type(i) == list for i in pot]):
            raise ValueError("Invalid potential format!")

        # Sympify the expression strings for each entry of the potential matrix
        potmatrix = [ [ sympy.sympify(item) for item in row ] for row in pot ]

        # Get the default parameters, if any
        if potential_description.has_key("defaults"):
            default_params = potential_description["defaults"]
        else:
            default_params = {}

        # Build the potential matrix with known values substituted for symbolic constants
        final_matrix = []
        free_symbols = []

        for row in potmatrix:
            cur_row = []
            for item in row:
                # Get atoms, but only symbols
                symbols = item.atoms(sympy.Symbol)
                values = {}

                # Search symbols and find a value for each one
                for atom in symbols:
                    if parameters.has_key(atom.name):
                        # A value is given by the parameter provider
                        val = parameters[atom.name]
                    elif default_params.has_key(atom.name):
                        # We do have a default value
                        val = default_params[atom.name]
                    else:
                        # No default value found either! This could be an input
                        # error in the potential definition but we rather interpret this
                        # case as a free parameter, for example the first space coordinate x
                        free_symbols.append(atom)
                        continue

                    # Sympify in case the values was specified as a string
                    values[atom.name] = sympy.sympify(val)

                # Substitute the values for the symbols
                # Remember expressions are immutable
                item = item.subs(values)

                # Try to simplify, but may fail
                try:
                    item = sympy.simplify(item)
                except:
                    pass

                # And insert into the final potential matrix
                cur_row.append(item)
            # Finished the current row
            final_matrix.append(cur_row)

        # Create a real sympy matrix instance
        potential_matrix = sympy.Matrix(final_matrix)
        # potential_matrix is set now


        # Check if the matrix is square
        if not potential_matrix.is_square:
            raise ValueError("Potential matrix is not square!")

        # Read of the number of components
        if potential_description.has_key("number_components"):
            nc_given = potential_description["number_components"]
            nc_found = potential_matrix.rows
            if nc_given != nc_found:
                raise ValueError("Inconsistent number of energy levels!")
            nc = nc_given
        else:
            nc = potential_matrix.rows
        # nc is set now

        # Now it gets really UGLY ... :-(

        # should also check parameters["potential"] !!

        # Get the variables used for the space axes
        if not potential_description.has_key("variables"):
            found_free_variables = sorted(set(free_symbols))
            unused_free_variables = []

            if potential_description.has_key("dimension"):
                # The dimension is explicitely specified
                dimension_given = potential_description["dimension"]
                dimension_found = len(found_free_variables)

                if dimension_found > dimension_given:
                    raise ValueError("Too many free variables!")
                elif dimension_found < dimension_given:
                    # Create additional free variables
                    unused_free_variables = [ sympy.Dummy("x_"+str(i)) for i in xrange(dimension_found,dimension_given) ]

                assert len(found_free_variables) + len(unused_free_variables) == dimension_given
                dimension = dimension_given
            else:
                dimension = len(found_free_variables)

            free_variables = found_free_variables

        else:
            given_free_variables = set(map(sympy.sympify, potential_description["variables"]))
            found_free_variables = set(free_symbols)
            unused_free_variables = []

            if potential_description.has_key("dimension"):
                # The dimension is explicitely specified
                dimension = potential_description["dimension"]
                assert dimension == len(given_free_variables)
            else:
                dimension = len(given_free_variables)

            if len(found_free_variables) > dimension:
                raise ValueError("Too many free variables!")
            elif len(found_free_variables) < dimension:
                assert given_free_variables.issuperset(found_free_variables)
                unused_free_variables = given_free_variables.difference(found_free_variables)

            free_variables = found_free_variables

        # dimension, free_vars and unused_free_vars set now

        if len(free_variables) == 0 and len(unused_free_variables) == 0:
            raise ValueError("Could not construct potential, not enough data given!")

        # Debug:
        print("===================")
        print("Space dimension is:")
        print(dimension)
        print("Free variables are:")
        print(free_variables)
        print("Unused variables are:")
        print(unused_free_variables)
        print("Potential matrix is:")
        print(potential_matrix)
        print("Number of levels:")
        print(nc)

        # Hack for unique symbols
        free_symbols = list(set(free_symbols))

        # Create instances of MatrixPotential*
        if nc == 1:
            from MatrixPotential1S import MatrixPotential1S
            potential = MatrixPotential1S(potential_matrix, free_symbols)
        elif nc == 2:
            from MatrixPotential2S import MatrixPotential2S
            potential = MatrixPotential2S(potential_matrix, free_symbols)
        else:
            #from MatrixPotentialMS import MatrixPotentialMS
            pass

        return potential
