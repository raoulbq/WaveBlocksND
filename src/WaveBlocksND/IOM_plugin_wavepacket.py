"""The WaveBlocks Project

IOM plugin providing functions for handling
homogeneous Hagedorn wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import pickle
import numpy as np


def add_wavepacket(self, parameters, timeslots=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Add storage for the homogeneous wavepackets.

    :param parameters: An :py:class:`ParameterProvider` instance with at
                       least the keys ``dimension`` and ``ncomponents``.
    :param timeslots: The number of time slots we need. Can be set to ``None``
                      to get automatically growing datasets.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    N = parameters["ncomponents"]
    D = parameters["dimension"]

    if timeslots is None:
        T = 0
        Ts = None
    else:
        T = timeslots
        Ts = timeslots

    # The overall group containing all wavepacket data
    grp_wp = self._srf[self._prefixb+str(blockid)].require_group("wavepacket")
    # The group for storing the basis shapes
    grp_wp.create_group("basisshapes")
    # The group for storing the parameter set Pi
    grp_pi = grp_wp.create_group("Pi")
    # The group for storing the coefficients
    grp_ci = grp_wp.create_group("coefficients")
    # Create the dataset with appropriate parameters
    grp_wp.create_dataset("timegrid", (T,), dtype=np.integer, chunks=True, maxshape=(Ts,), fillvalue=-1)
    grp_wp.create_dataset("basis_shape_hash", (T, N), dtype=np.integer, chunks=True, maxshape=(Ts,N))
    grp_wp.create_dataset("basis_size", (T, N), dtype=np.integer, chunks=True, maxshape=(Ts,N))
    # Parameters
    if "q" in key and not "q" in grp_pi.keys():
        grp_pi.create_dataset("q", (T,D,1), dtype=np.complexfloating, chunks=True, maxshape=(Ts,D,1))
    if "p" in key and not "p" in grp_pi.keys():
        grp_pi.create_dataset("p", (T,D,1), dtype=np.complexfloating, chunks=True, maxshape=(Ts,D,1))
    if "Q" in key and not "Q" in grp_pi.keys():
        grp_pi.create_dataset("Q", (T,D,D), dtype=np.complexfloating, chunks=True, maxshape=(Ts,D,D))
    if "P" in key and not "P" in grp_pi.keys():
        grp_pi.create_dataset("P", (T,D,D), dtype=np.complexfloating, chunks=True, maxshape=(Ts,D,D))
    if "S" in key and not "S" in grp_pi.keys():
        grp_pi.create_dataset("S", (T,1,1), dtype=np.complexfloating, chunks=True, maxshape=(Ts,1,1))
    if "adQ" in key and not "adQ" in grp_pi.keys():
        grp_pi.create_dataset("adQ", (T,1,1), dtype=np.complexfloating, chunks=True, maxshape=(Ts,1,1))
    # Coefficients
    for i in xrange(N):
        grp_ci.create_dataset("c_"+str(i), (T,1), dtype=np.complexfloating, chunks=(1,8), maxshape=(Ts,None))

    # Attach pointer to data instead timegrid
    grp_pi.attrs["pointer"] = 0
    grp_ci.attrs["pointer"] = 0


def delete_wavepacket(self, blockid=0):
    r"""Remove the stored wavepackets.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavepacket"]
    except KeyError:
        pass


def has_wavepacket(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    """
    return "wavepacket" in self._srf[self._prefixb+str(blockid)].keys()


def save_wavepacket_description(self, descr, blockid=0):
    r"""Save the description of this wavepacket.

    :param descr: The description.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket"
    # Save the description
    for key, value in descr.iteritems():
        # Store all the values as pickled strings because hdf can
        # only store strings or ndarrays as attributes.
        self._srf[pathd].attrs[key] = pickle.dumps(value)


def save_wavepacket_parameters(self, parameters, timestep=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Save the parameter set :math:`\Pi` of the Hagedorn wavepacket :math:`\Psi` to a file.

    :param parameters: The parameter set of the Hagedorn wavepacket.
    :type parameters: A ``list`` containing the (five) ``ndarrays`` like :math:`(q,p,Q,P,S)`
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/Pi/"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    for key, item in zip(key, parameters):
        self.must_resize(pathd+key, timeslot)
        self._srf[pathd+key][timeslot,:,:] = item

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_wavepacket_coefficients(self, coefficients, basisshapes, timestep=None, blockid=0):
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
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_shape_hash"
    pathbsi = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_size"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/coefficients/"

    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathbs, timeslot)
    self.must_resize(pathbsi, timeslot)
    for index, (bs,ci) in enumerate(zip(basisshapes, coefficients)):
        self.must_resize(pathd+"c_"+str(index), timeslot)
        size = bs.get_basis_size()
        # Do we have to resize due to changed number of coefficients
        self.must_resize(pathd+"c_"+str(index), size-1, axis=1)
        self._srf[pathbsi][timeslot,index] = size
        self._srf[pathbs][timeslot,index] = hash(bs)
        self._srf[pathd+"c_"+str(index)][timeslot,:size] = np.squeeze(ci)

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_wavepacket_basisshapes(self, basisshape, blockid=0):
    r"""Save the basis shapes of the Hagedorn wavepacket to a file.

    :param basisshape: The basis shape of the Hagedorn wavepacket.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/basisshapes/"

    ha = hash(basisshape)
    name = "basis_shape_"+str(ha)

    # Chech if we already stored this basis shape
    if not name in self._srf[pathd].keys():
        # Create new data set
        daset = self._srf[pathd].create_dataset("basis_shape_"+str(ha), (1,), dtype=np.integer)
        daset[0] = ha

        # Save the description
        descr = basisshape.get_description()
        for key, value in descr.iteritems():
            # Store all the values as pickled strings because hdf can
            # only store strings or ndarrays as attributes.
            daset.attrs[key] = pickle.dumps(value)

        # TODO: Consider to save the mapping. Do we want or need this?


def load_wavepacket_description(self, blockid=0):
    r"""Load the wavepacket description.

    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket"

    # Load and return all descriptions available
    descr = {}
    for key, value in self._srf[pathd].attrs.iteritems():
        descr[key] = pickle.loads(value)
    return descr


def load_wavepacket_timegrid(self, blockid=0):
    r"""Load the wavepacket timegrid.

    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    return self._srf[pathtg][:]


def load_wavepacket_parameters(self, timestep=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Load the wavepacket parameters.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which parameters to load. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/Pi/"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        params = tuple([ self._srf[pathd+k][index,:,:] for k in key ])
    else:
        params = tuple([ self._srf[pathd+k][...,:,:] for k in key ])

    return params


def load_wavepacket_coefficients(self, timestep=None, get_hashes=False, component=None, blockid=0):
    r"""Load the wavepacket coefficients.

    :param timestep: Load only the data of this timestep.
    :param get_hashes: Return the corresponding basis shape hashes.
    :param component: Load only data from this component.
    :param blockid: The ID of the data block to operate on.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_shape_hash"
    pathbsi = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_size"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/coefficients/"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
    else:
        index = slice(None)

    # Number components
    N = len(self._srf[pathd].keys())

    # Single component requested
    if component is not None:
        components = [component]
    else:
        components = xrange(N)

    # Load the hash data
    if get_hashes is True:
        hashes = self._srf[pathbs][index,...]
        hashes = np.hsplit(hashes, N)
        # Only a single wanted
        if component is not None:
            hashes = hashes[component]

    # Load the coefficient data
    data = []
    if timestep is not None:
        for i in components:
            size = self._srf[pathbsi][index,i]
            data.append(self._srf[pathd+"c_"+str(i)][index,:size])
    else:
        for i in components:
            data.append(self._srf[pathd+"c_"+str(i)][index,...])

    # TODO: Consider unpacking data for single components

    if get_hashes is True:
        return (hashes, data)
    else:
        return data


def load_wavepacket_basisshapes(self, the_hash=None, blockid=0):
    r"""Load the basis shapes by hash.

    :param the_hash: The hash of the basis shape whose description we want to load.
    :param blockid: The ID of the data block to operate on.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/basisshapes/"

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


def load_wavepacket(self, timestep, blockid=0, key=("q","p","Q","P","S")):
    r"""Load a wavepacket at a given timestep and return a fully configured instance.
    This method just calls some other :py:class:`IOManager` methods in the correct
    order. It is included only for convenience and is not particularly efficient.

    :param timestep: The timestep :math:`n` we load the wavepacket.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    :param blockid: The ID of the data block to operate on.
    :return: A :py:class:`HagedornWavepacket` instance.
    """
    from BlockFactory import BlockFactory
    BF = BlockFactory()

    descr = self.load_wavepacket_description(blockid=blockid)
    HAWP = BF.create_wavepacket(descr)

    # Parameters and coefficients
    Pi = self.load_wavepacket_parameters(timestep=timestep, blockid=blockid, key=key)
    hashes, coeffs = self.load_wavepacket_coefficients(timestep=timestep, get_hashes=True, blockid=blockid)

    # Basis shapes
    Ks = []
    for ahash in hashes:
        K_descr = self.load_wavepacket_basisshapes(the_hash=ahash, blockid=blockid)
        Ks.append(BF.create_basis_shape(K_descr))

    # Configure the wavepacket
    HAWP.set_parameters(Pi, key=key)
    HAWP.set_basis_shapes(Ks)
    HAWP.set_coefficients(coeffs)

    return HAWP


def save_wavepacket(self, packet, timestep, blockid=None, key=("q","p","Q","P","S")):
    r"""Save a wavepacket at a given timestep and read all data to save from the
    :py:class:`HagedornWavepacket` instance provided. This method just calls some
    other :py:class:`IOManager` methods in the correct order. It is included only
    for convenience and is not particularly efficient. We assume the wavepacket
    is already set up with the correct :py:meth:`add_wavepacket` method call.

    :param packet: The :py:class:`HagedornWavepacket` instance we want to save.
    :param timestep: The timestep :math:`n` at which we save the wavepacket.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    :param blockid: The ID of the data block to operate on.
    """
    # Description
    self.save_wavepacket_description(packet.get_description(), blockid=blockid)
    # Pi
    self.save_wavepacket_parameters(packet.get_parameters(key=key), timestep=timestep, blockid=blockid, key=key)
    # Basis shapes
    for shape in packet.get_basis_shapes():
        self.save_wavepacket_basisshapes(shape, blockid=blockid)
    # Coefficients
    self.save_wavepacket_coefficients(packet.get_coefficients(), packet.get_basis_shapes(), timestep=timestep, blockid=blockid)
