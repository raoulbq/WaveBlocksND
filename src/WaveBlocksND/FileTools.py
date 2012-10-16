"""The WaveBlocks Project

This file contains various functions for finding and retrieving
the files that contain parameter settings and simulation results.
Note: The terms 'path' and 'ID' are used as synonyms here. Each simulation
ID is just the basename of the path or the configuration file.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import os

import GlobalDefaults as GD


def get_result_dirs(path):
    r"""Lists all simulations (IDs) that can be found under the given path.

    :param path: The filesystem path under which we search for simulations.
    :return: A list of simulation IDs.
    """
    dirs = [ os.path.join(path, adir) for adir in os.listdir(path) if os.path.isdir(adir) ]
    return dirs


def get_parameters_file(path):
    r"""Search for a configuration file containing the simulation parameters under a given path.
    Note that in case there are more than one .py file under the given path we just return the
    first one found!

    :parameter path: The path under which we search for a configuration file.
    :return: The path (filename) of the configuration file.
    """
    parameters_file = None

    for afile in os.listdir(path):
        if afile.endswith(".py"):
            parameters_file = afile
            break

    if parameters_file is None:
        raise IOError("No configuration .py file found!")

    parameters_file = os.path.join(path, parameters_file)
    return parameters_file


def get_results_file(path, fileext=GD.ext_resultdatafile):
    r"""Search for a file containing the simulation results under a given path.
    Note that in case there are more than one .hdf5 file under the given path
    we just return the first one found!

    :param path: The path under which we search for a output file.
    :return: The path (filename) of the output file.
    """
    results_file = None

    for afile in os.listdir(path):
        if os.path.isfile(afile) and afile.endswith(fileext):
            results_file = afile
            break

    if results_file is None:
        raise IOError("No results .hdf5 file found!")

    results_file = os.path.join(path, results_file)
    return results_file


def get_number_simulations(path):
    r"""Get the number of simulations at hand below the given path.

    :param path: The path under which we search for a output file.
    :return: The number of simulations result directories.
    """
    ids = get_result_dirs(path)
    return len(ids)


def name_contains(name, pattern):
    r"""Checks if a simulation ID contains a given pattern.

    :param name: The full simulation ID.
    :param pattern: The pattern in question.
    :return: A boolean answer.
    """
    return pattern in name


def gather_all(stringlist, pattern):
    r"""Collects all simulation IDs which contain a specific pattern from a given list.

    :param stringlist: A list with the simulation IDs.
    :param pattern: The pattern.
    :return: A list of simulation IDs that contain the given pattern.
    """
    gathered = [ s for s in stringlist if name_contains(s, pattern) ]
    return gathered


def compare_by(namea, nameb, pattern, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel, as_string=True):
    r"""Compare two simulation IDs with respect to a (numerical) value in the ID.

    :param namea: The first name in the comparison.
    :param nameb: The second name in the comparison.
    :param pattern: The pattern whose (numerical) value is used for sorting.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :param as_string: Determines if the values for ``pattern`` get converted to floats.
    :return: A boolean answer if the IDs are the same w.r.t the pattern.
    """
    part1 = namea.partition(ldel + pattern + mdel)
    part2 = nameb.partition(ldel + pattern + mdel)

    part1 = part1[0:2] + part1[2].partition(rdel)
    part2 = part2[0:2] + part2[2].partition(rdel)

    # Convert values to float for comparisons
    if not as_string:
        part1 = part1[0:2] + (float(part1[2]),) + part1[3:]
        part2 = part2[0:2] + (float(part2[2]),) + part2[3:]

    return part1[2] == part2[2]


def group_by(stringlist, pattern, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel, as_string=True):
    r"""Groups simulation IDs with respect to a pattern.

    :param stringlist: A list with the simulation IDs.
    :param pattern: The pattern used for grouping.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :param as_string: Determines if the values for ``pattern`` get converted to floats. Not used here.
    :return: A list of groups of simulation IDs.
    """
    tmp = [ s.partition(ldel + pattern + mdel) for s in stringlist ]
    tmp = [ s[0:2] + s[2].partition(rdel) for s in tmp ]

    distinct_vals = set([ s[2] for s in tmp ])

    groups = [ [] for i in xrange(len(distinct_vals)) ]

    for item in tmp:
        for index, val in enumerate(distinct_vals):
            # Yes, we compare the strings here and not the floats
            # to avoid representation errors when converting to floats.
            if item[2] == val:
                # Concatenate the fragments again
                groups[index].append( reduce(lambda x,y: x+y, item) )
                break

    return groups


def intersect_by(lista, listb, pattern, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel, as_string=True):
    r"""Find the intersection of two lists containing simulation IDs.

    :param lista: A first list with the simulation IDs.
    :param listb: A second list with the simulation IDs.
    :param pattern: The pattern whose numerical value is used for sorting.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :param as_string: Determines if the values for ``pattern`` get converted to floats.
    :return: A sorted list of simulation IDs.
    """
    result = []

    for namea in lista:
        for nameb in listb:
            if compare_by(namea, nameb, pattern, ldel=ldel, mdel=mdel, rdel=rdel, as_string=as_string):
                result.append(namea)
                result.append(nameb)

    # Remove possible duplicates
    result = list(set(result))
    return result


def sort_by(stringlist, pattern, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel, as_string=False):
    r"""Sorts simulation IDs with respect to a (numerical) value in the ID.

    :param stringlist: A list with the simulation IDs.
    :param pattern: The pattern whose (numerical) value is used for sorting.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :param as_string: Determines if the values for ``pattern`` get converted to floats.
    :return: A sorted list of simulation IDs.
    """
    tmp = [ s.partition(ldel + pattern + mdel) for s in stringlist ]
    tmp = [ s[0:2] + s[2].partition(rdel) for s in tmp ]

    if not as_string:
        # Convert to float and append numeric values to the splitted IDs
        tmp = [ s + (float(s[2]),) for s in tmp ]
    else:
        # Use string in comparison, allows sorting
        tmp = [ s + (s[2],) for s in tmp ]

    # Define a costum order relation
    def compare(x,y):
        if x[-1] > y[-1]:
            return 1
        elif y[-1] > x[-1]:
            return -1
        else:
            return 0

    # Sort w.r.t. the numerical value
    tmp.sort(cmp=compare)

    # Remove numeric value and concatenate the fragments again
    f = lambda x,y: x+y
    sorted_list = [ reduce(f, s[:-1]) for s in tmp ]

    return sorted_list


def get_max_by(stringlist, pattern, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel, as_string=False):
    r"""Get the maximum of a list with simulation IDs with respect to a (numerical) value in the ID.
    This is just a simple convenience function so that the user needs not to remember if the
    sort order is ascending or descending which plays no role for iteration.

    :param stringlist: A list with the simulation IDs.
    :param pattern: The pattern whose (numerical) value is used for sorting.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :param as_string: Determines if the values for ``pattern`` get converted to floats.
    :return: A sorted list of simulation IDs.
    """
    sortedlist = sort_by(stringlist, pattern, ldel=ldel, mdel=mdel, rdel=rdel, as_string=as_string)
    return sortedlist[-1]


def get_min_by(stringlist, pattern, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel, as_string=False):
    r"""Get the minimum of a list with simulation IDs with respect to a (numerical) value in the ID.
    This is just a simple convenience function so that the user needs not to remember if the
    sort order is ascending or descending which plays no role for iteration.

    :param stringlist: A list with the simulation IDs.
    :param pattern: The pattern whose (numerical) value is used for sorting.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :param as_string: Determines if the values for ``pattern`` get converted to floats.
    :return: A sorted list of simulation IDs.
    """
    sortedlist = sort_by(stringlist, pattern, ldel=ldel, mdel=mdel, rdel=rdel, as_string=as_string)
    return sortedlist[0]


def get_items(name):
    r"""Get a list of all the ``key=value`` items in the name.

    :param name: The name from which to get the items.
    :return: A list of all ``key=value`` items present in the ``name``.
    """
    parts = name.split(ldel)
    items = [ ldel+p for p in parts if rdel in p ]
    return items


def get_item(name, pattern):
    r"""Get a single ``key=value`` item out of the name.
    The ``pattern`` specifies the ``key`` part.

    :param name: The name from which to get the item.
    :param pattern: The pattern whose value is used for ``key``.
    :return: A (list of the) item(s) whose ``key`` part matches ``pattern``.
    """
    items = get_items(name)
    result = filter(lambda item: pattern in item, items)
    if len(result) == 1:
        result = result[0]
    return result


def get_by_item(stringlist, item):
    r"""Get a filtered list of simulation IDs containing a given ``key=value`` item.

    :param stringlist: A list with the simulation IDs.
    :param item: The ``key=value`` item to search for.
    :return: A list of simulation IDs.
    """
    result = [ name for name in stringlist if item in name ]
    return result


def get_value(item, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel):
    r"""Get the ``value`` part of a given ``key=value`` pair.

    :param item: The ``key=value`` item.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :return: The ``value`` part of the given item.
    """
    rightpart = item.partition(mdel)
    value = rightpart[-1].partition(rdel)
    return value[0]
    

def get_by_value(stringlist, pattern, value, ldel=GD.kvp_ldel, mdel=GD.kvp_mdel, rdel=GD.kvp_rdel):
    r"""Get a filtered list of simulation IDs by specifying the ``key``
    and ``value`` of some ``key=value`` pair.

    :param stringlist: A list with the simulation IDs.
    :param pattern: The pattern that is used for ``key`.
    :param value: The value that is used for ``value``.
    :param ldel: Left delimiter of the pattern.
    :param mdel: Middle delimiter of the pattern.
    :param rdel: Right delimiter of the pattern.
    :return: A list of IDs that contain a ``key`` with given ``value``.
    """
    result = []

    for name in stringlist:
        item = get_item(name, pattern)

        if not type(item) == str:
            print("Warning: ambiguous pattern "+pattern)
            item = item[0]

        val = get_value(item, ldel=ldel, mdel=mdel, rdel=rdel)

        if val == value:
            result.append(name)

    return result
