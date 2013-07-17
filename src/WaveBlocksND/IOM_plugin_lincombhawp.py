"""The WaveBlocks Project

IOM plugin providing functions for handling
linear combinations of Hagedorn wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

import pickle
import numpy as np


def add_lincombhawp(self, parameters, timeslots=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Add storage for the linear combination of Hagedorn wavepackets.

    :param parameters: An :py:class:`ParameterProvider` instance with at
                       least the keys ``dimension`` and ``ncomponents``.
    :param timeslots: The number of time slots we need. Can be ``None``
                      to get automatically growing datasets.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    N = parameters["ncomponents"]
    D = parameters["dimension"]

    # TODO: Handle multi-component packets
    assert N == 1

    # TODO: Consider an 'assume_constant' option
    #       None -> False -> lcsize dynamic, int -> lcsize fix
    # Could maybe avoid many resize operations

    # The overall group containing all lincombwp data
    grp_lc = self._srf[self._prefixb+str(blockid)].require_group("lincombhawp")

    # The group for storing the basis shapes
    grp_wpbs = grp_lc.create_group("basisshapes")
    # The group for storing the parameter set Pi
    grp_wppi = grp_lc.create_group("Pi")
    # The group for storing the coefficients
    grp_wpci = grp_lc.create_group("wp_coefficients")

    # TODO: Consider merging both cases

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_tg_lc = grp_lc.create_dataset("timegrid_lc_coefficients", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_tg_wp = grp_lc.create_dataset("timegrid_wp_parameters", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_tg_wc = grp_lc.create_dataset("timegrid_wp_coefficients", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_lcsize = grp_lc.create_dataset("lincomb_size", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        # Linear combination coefficients
        daset_ci = grp_lc.create_dataset("lc_coefficients", (0, 0), dtype=np.complexfloating, chunks=True, maxshape=(None,None))
        # Linear combination wavepackets
        daset_bs = grp_lc.create_dataset("basis_shapes_hashes", (0, 0, N), dtype=np.integer, chunks=True, maxshape=(None,None,N))
        daset_bsi = grp_lc.create_dataset("basis_sizes", (0, 0, N), dtype=np.integer, chunks=True, maxshape=(None,None,N))
        # Wavepackets parameters
        if "q" in key and not "q" in grp_wppi.keys():
            daset_q = grp_wppi.create_dataset("q", (0, 0, D), dtype=np.complexfloating, chunks=True, maxshape=(None,None,D))
        if "p" in key and not "p" in grp_wppi.keys():
            daset_p = grp_wppi.create_dataset("p", (0, 0, D), dtype=np.complexfloating, chunks=True, maxshape=(None,None,D))
        if "Q" in key and not "Q" in grp_wppi.keys():
            daset_Q = grp_wppi.create_dataset("Q", (0, 0, D, D), dtype=np.complexfloating, chunks=True, maxshape=(None,None,D,D))
        if "P" in key and not "P" in grp_wppi.keys():
            daset_P = grp_wppi.create_dataset("P", (0, 0, D, D), dtype=np.complexfloating, chunks=True, maxshape=(None,None,D,D))
        if "S" in key and not "S" in grp_wppi.keys():
            daset_S = grp_wppi.create_dataset("S", (0, 0, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,None,1))
        # Wavepackets coefficients
        for i in xrange(N):
            daset_c_i = grp_wpci.create_dataset("c_"+str(i), (0, 0, 0), dtype=np.complexfloating, chunks=True, maxshape=(None,None,None))
    # else:
    #     # User specified how much space is necessary.
    #     daset_tg_c = grp_lc.create_dataset("timegrid_coefficients", (timeslots,), dtype=np.integer)
    #     daset_tg_p = grp_lc.create_dataset("timegrid_packets", (timeslots,), dtype=np.integer)
    #     daset_lcsize = grp_lc.create_dataset("lincomb_size", (timeslots,), dtype=np.integer)
    #     # Coefficients
    #     daset_ci = grp_lc.create_dataset("coefficients", (timeslots, 0), dtype=np.complexfloating, chunks=True, maxshape=(timeslots,None))
    #     # Packet IDs
    #     daset_refs = grp_lc.create_dataset("packet_refs", (timeslots, 0), dtype=np.dtype((str,32)), chunks=True, maxshape=(timeslots,None))

    #     # Mark all steps as invalid
    #     daset_tg_c[...] = -1.0
    #     daset_tg_p[...] = -1.0

    # Attach pointer to timegrid
    daset_tg_lc.attrs["pointer"] = 0
    grp_wppi.attrs["pointer"] = 0
    grp_wpci.attrs["pointer"] = 0


def delete_lincombhawp(self, blockid=0):
    r"""Remove the stored linear combination.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/lincombhawp"]
    except KeyError:
        pass


def has_lincombhawp(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return "lincombhawp" in self._srf[self._prefixb+str(blockid)].keys()


def save_lincombhawp_description(self, descr, blockid=0):
    r"""Save the description of this linear combination.

    :param descr: The description.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp"
    # Save the description
    for key, value in descr.iteritems():
        # Store all the values as pickled strings because hdf can
        # only store strings or ndarrays as attributes.
        self._srf[pathd].attrs[key] = pickle.dumps(value)


def save_lincombhawp_coefficients(self, coefficients, timestep, blockid=0):
    r"""Save the coefficients of the linear combination to a file.

    :param coefficients: The coefficients of the linear combination of wavepackets.
    :type coefficients: A single, suitable ``ndarray``.
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_lc_coefficients"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombhawp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp/lc_coefficients"

    timeslot = self._srf[pathtg].attrs["pointer"]

    # Write the data
    self.must_resize(pathlcs, timeslot)
    J = np.size(coefficients)
    self._srf[pathlcs][timeslot] = J
    self.must_resize(pathd, timeslot)
    if not J == 0:
        self.must_resize(pathd, J-1, axis=1)
        self._srf[pathd][timeslot,:J] = np.squeeze(coefficients)

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathtg].attrs["pointer"] += 1


def save_lincombhawp_wavepacket_parameters(self, parameters, timestep, blockid=0, key=("q","p","Q","P","S")):
    r"""Save the parameter set :math:`\Pi` of the Hagedorn wavepacket :math:`\Psi` to a file.

    :param parameters: The parameter set of the Hagedorn wavepacket.
    :type parameters: A ``list`` containing the (five) ``ndarrays`` like :math:`(q,p,Q,P,S)`
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_wp_parameters"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombhawp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp/Pi/"
    timeslot = self._srf[pathd].attrs["pointer"]

    # TODO: This an assumption based on data layout and stable
    J = parameters[0].shape[0]

    # Write the basis size
    self.must_resize(pathlcs, timeslot)
    self._srf[pathlcs][timeslot] = J

    # Write the parameters
    for key, item in zip(key, parameters):
        self.must_resize(pathd+key, timeslot)
        self.must_resize(pathd+key, J-1, axis=1)
        self._srf[pathd+key][timeslot,:J,...] = item

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_lincombhawp_wavepacket_coefficients(self, coefficients, basissizes, basisshapeshashes, timestep=None, blockid=0):
    r"""Save the coefficients of the Hagedorn wavepacket to a file.
    Warning: we do only save tha hash of the basis shapes here!
    You have to save the basis shape with the corresponding function too.

    :param coefficients: The coefficients of the Hagedorn wavepacket.
    :type coefficients: A ``list`` with :math:`N` suitable ``ndarrays``.
    :param basisshapes: The corresponding basis shapes of the Hagedorn wavepacket.
    :type basisshapes: A ``list`` with :math:`N` :py:class:`BasisShape` subclass instances.
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_wp_coefficients"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombhawp/lincomb_size"
    pathbsi = "/"+self._prefixb+str(blockid)+"/lincombhawp/basis_sizes"
    pathbsh = "/"+self._prefixb+str(blockid)+"/lincombhawp/basis_shapes_hashes"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp/wp_coefficients/"

    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the lincomb size
    basissizes = np.atleast_1d(basissizes)
    J = basissizes.shape[0]

    self.must_resize(pathlcs, timeslot)
    self._srf[pathlcs][timeslot] = J

    # Write all basis sizes
    self.must_resize(pathbsi, timeslot)
    self.must_resize(pathbsi, J-1, axis=1)
    self._srf[pathbsi][timeslot,:J,0] = basissizes

    # Write basis shape hashes
    basisshapeshashes = np.atleast_1d(basisshapeshashes)
    self.must_resize(pathbsh, timeslot)
    self.must_resize(pathbsh, J-1, axis=1)
    self._srf[pathbsh][timeslot,:J,0] = basisshapeshashes

    # Write the wavepackets coefficients data
    coefficients = np.atleast_2d(coefficients)
    j, k = coefficients.shape
    # TODO: Allow wavepackets with multiple components
    index = 0
    pathc = pathd+"c_"+str(index)
    # Do we have to resize due to changed number of packets or coefficients
    self.must_resize(pathc, timeslot)
    self.must_resize(pathc, j-1, axis=1)
    self.must_resize(pathc, k-1, axis=2)
    self._srf[pathc][timeslot,:j,:k] = coefficients

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_lincombhawp_wavepacket_basisshapes(self, basisshapes, blockid=0):
    r"""Save the basis shapes of the linear combination of
    Hagedorn wavepacket to a file.

    :param basisshapes: A list of the basis shapes of the linear combination.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp/basisshapes/"

    for basisshape in basisshapes:
        ha = hash(basisshape)
        name = "basis_shape_"+str(ha)

        # Chech if we already stored this basis shape
        if not name in self._srf[pathd].keys():
            # TODO: Consider storing all hashes in one big dataset
            # Create new data set
            daset = self._srf[pathd].create_dataset("basis_shape_"+str(ha), (1,), dtype=np.integer)
            daset[0] = ha

            # Save the description
            descr = basisshape.get_description()
            for key, value in descr.iteritems():
                # Store all the values as pickled strings because hdf can
                # only store strings or ndarrays as attributes.
                daset.attrs[key] = pickle.dumps(value)


def load_lincombhawp_description(self, blockid=0):
    r"""Load the description of this linear combination.

    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp"

    # Load and return all descriptions available
    descr = {}
    for key, value in self._srf[pathd].attrs.iteritems():
        descr[key] = pickle.loads(value)
    return descr


def load_lincombhawp_timegrid(self, blockid=0, key=("coeffs", "packets")):
    r"""Load the timegrid of this linear combination.

    :param blockid: The ID of the data block to operate on.
    :param key: Specify which linear combination timegrids to load. All are independent.
    :type key: Tuple of valid identifier strings that are ``ceoffs`` and ``packets``.
               Default is ``("coeffs", "packets")``.
    """
    tg = []
    for item in key:
        if item == "coeffs":
            pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_lc_coefficients"
            tg.append(self._srf[pathtg][:])
        elif item == "packets":
            pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_lc_packets"
            tg.append(self._srf[pathtg][:])

    if len(tg) == 1:
        return tg[0]
    else:
        return tuple(tg)


def load_lincombhawp_size(self, timestep=None, blockid=0):
    r"""Load the size (number of packets) of this linear combination.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_lc_coefficients"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombhawp/lincomb_size"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self._srf[pathlcs][index]
    else:
        index = slice(None)
        return self._srf[pathlcs][index]


def load_lincombhawp_coefficients(self, timestep=None, blockid=0):
    r"""Load the coefficients of this linear combination.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_lc_coefficients"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombhawp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp/lc_coefficients"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        J = self._srf[pathlcs][index]
        return self._srf[pathd][index,:J]
    else:
        index = slice(None)
        return self._srf[pathd][index,:]


def load_lincombhawp_wavepacket_parameters(self, timestep=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Load the wavepacket parameters.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which parameters to load. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/lincombhawp/timegrid_wp_parameters"
    pathlcs = "/"+self._prefixb+str(blockid)+"/lincombhawp/lincomb_size"
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp/Pi/"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        J = self._srf[pathlcs][index]
        params = tuple([ self._srf[pathd+k][index,:J,...] for k in key ])
    else:
        params = tuple([ self._srf[pathd+k][:,:,...] for k in key ])

    return params


# def load_wavepacket_coefficients(self, timestep=None, get_hashes=False, component=None, blockid=0):
#     r"""Load the wavepacket coefficients.

#     :param timestep: Load only the data of this timestep.
#     :param get_hashes: Return the corresponding basis shape hashes.
#     :param component: Load only data from this component.
#     :param blockid: The ID of the data block to operate on.
#     """
#     pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
#     pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_shape_hash"
#     pathbsi = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_size"
#     pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/coefficients/"

#     if timestep is not None:
#         index = self.find_timestep_index(pathtg, timestep)
#     else:
#         index = slice(None)

#     # Number components
#     N = len(self._srf[pathd].keys())

#     # Single component requested
#     if component is not None:
#         components = [component]
#     else:
#         components = xrange(N)

#     # Load the hash data
#     if get_hashes is True:
#         hashes = self._srf[pathbs][index,...]
#         hashes = np.hsplit(hashes, N)
#         # Only a single wanted
#         if component is not None:
#             hashes = hashes[component]

#     # Load the coefficient data
#     data = []
#     if timestep is not None:
#         for i in components:
#             size = self._srf[pathbsi][index,i]
#             data.append(self._srf[pathd+"c_"+str(i)][index,:size])
#     else:
#         for i in components:
#             data.append(self._srf[pathd+"c_"+str(i)][index,...])

#     # TODO: Consider unpacking data for single components

#     if get_hashes is True:
#         return (hashes, data)
#     else:
#         return data


def load_wavepacket_basisshapes(self, the_hash=None, blockid=0):
    r"""Load the basis shapes by hash.

    :param the_hash: The hash of the basis shape whose description we want to load.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/lincombhawp/basisshapes/"

    if the_hash is None:
        # Load and return all descriptions available
        descrs = {}
        for ahash in self._srf[pathd].keys():
            # TODO: What data exactly do we want to return?
            descr = {}
            for key, value in self._srf[pathd+ahash].attrs.iteritems():
                descr[key] = pickle.loads(value)
            # 'ahash' is "basis_shape_..." and we want only the "..." part
            descrs[int(ahash[12:])] = descr
        return descrs
    else:
        the_hash = int(the_hash)
        name = "basis_shape_"+str(the_hash)
        # Chech if we already stored this basis shape
        if name in self._srf[pathd].keys():
            # TODO: What data exactly do we want to return?
            descr = {}
            for key, value in self._srf[pathd+name].attrs.iteritems():
                descr[key] = pickle.loads(value)
            return descr
        else:
            raise IndexError("No basis shape with given hash "+str(hash))


#
# The following two methods are only for convenience and are NOT particularly efficient.
#


# def load_lincombwp(self, timestep, blockid=0):
#     r"""Load a linear combination at a given timestep and return a fully configured
#     :py:class:`LinearCombinationOfWPs` instance. This method just calls some other
#     :py:class:`IOManager` methods in the correct order. It is included only for
#     convenience and is not particularly efficient.

#     :param timestep: The timestep :math:`n` we load the wavepacket.
#     :param blockid: The ID of the data block to operate on.
#     :return: A :py:class:`LinearCombinationOfWPs` instance.
#     """
#     from LinearCombinationOfWPs import LinearCombinationOfWPs

#     descr = self.load_lincombwp_description(blockid=blockid)

#     J = self.load_lincombwp_size(timestep=timestep, blockid=blockid)
#     if J == 0:
#         return None

#     # Load the data
#     c = self.load_lincombwp_coefficients(timestep=timestep, blockid=blockid)
#     psi = self.load_lincombwp_wavepackets(timestep=timestep, blockid=blockid)

#     # Assemble the linear combination
#     LC = LinearCombinationOfWPs(descr["dimension"], descr["ncomponents"])
#     LC.add_wavepackets(psi, c)

#     return LC


# def save_lincombwp(self, lincomb, timestep, blockid=0):
#     r"""Save a linear combination of general wavepackets at a given timestep and read
#     all data to save from the :py:class:`LinearCombinationOfWPs` instance provided. This
#     method just calls some other :py:class:`IOManager` methods in the correct order.
#     It is included only for convenience and is not particularly efficient. We assume
#     the linear combination is already set up with the correct :py:meth:`add_lincombwp`
#     method call.

#     :param lincomb: The :py:class:`LinearCombinationOfWPs` instance we want to save.
#     :param timestep: The timestep :math:`n` at which we save the linear combination.
#     :param blockid: The ID of the data block to operate on.
#     """
#     # Description
#     self.save_lincombwp_description(lincomb.get_description(), blockid=blockid)
#     # Wavepackets
#     self.save_lincombwp_wavepackets(lincomb.get_wavepackets(), timestep=timestep, blockid=blockid)
#     # Coefficients
#     self.save_lincombwp_coefficients(lincomb.get_coefficients(), timestep=timestep, blockid=blockid)
