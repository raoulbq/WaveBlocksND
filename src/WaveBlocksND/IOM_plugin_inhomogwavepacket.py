"""The WaveBlocks Project

IOM plugin providing functions for handling
homogeneous Hagedorn wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012, 2013 R. Bourquin
@license: Modified BSD License
"""

import pickle
import numpy as np


def add_inhomogwavepacket(self, parameters, timeslots=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Add storage for the inhomogeneous wavepackets.

    :param parameters: An :py:class:`ParameterProvider` instance with at
                       least the keys ``dimension`` and ``ncomponents``.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S", "adQ")``.
    """
    N = parameters["ncomponents"]
    D = parameters["dimension"]

    # The overall group containing all wavepacket data
    grp_wp = self._srf[self._prefixb+str(blockid)].require_group("wavepacket_inhomog")
    # The group for storing the basis shapes
    grp_bs = grp_wp.create_group("basisshapes")
    # The group for storing the parameter set Pi
    grp_pi = grp_wp.create_group("Pi")
    grp_pi.attrs["number_parameters"] = len(key)
    # The group for storing the coefficients
    grp_ci = grp_wp.create_group("coefficients")

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_tg = grp_wp.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_bs = grp_wp.create_dataset("basis_shape_hash", (0, N), dtype=np.integer, chunks=True, maxshape=(None,N))
        daset_bsi = grp_wp.create_dataset("basis_size", (0, N), dtype=np.integer, chunks=True, maxshape=(None,N))
        for i in xrange(N):
            if "q" in key and not "q" in grp_pi.keys():
                daset_q_i = grp_pi.create_dataset("q_"+str(i), (0, D, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,D,1))
            if "p" in key and not "p" in grp_pi.keys():
                daset_p_i = grp_pi.create_dataset("p_"+str(i), (0, D, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,D,1))
            if "Q" in key and not "Q" in grp_pi.keys():
                daset_Q_i = grp_pi.create_dataset("Q_"+str(i), (0, D, D), dtype=np.complexfloating, chunks=True, maxshape=(None,D,D))
            if "P" in key and not "P" in grp_pi.keys():
                daset_P_i = grp_pi.create_dataset("P_"+str(i), (0, D, D), dtype=np.complexfloating, chunks=True, maxshape=(None,D,D))
            if "S" in key and not "S" in grp_pi.keys():
                daset_S_i = grp_pi.create_dataset("S_"+str(i), (0, 1, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,1,1))
            if "adQ" in key and not "adQ" in grp_pi.keys():
                daset_adQ_i = grp_pi.create_dataset("adQ_"+str(i), (0, 1, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,1,1))
        for i in xrange(N):
            daset_c_i = grp_ci.create_dataset("c_"+str(i), (0, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,None))
    else:
        # User specified how much space is necessary.
        daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), dtype=np.integer)
        daset_bs = grp_wp.create_dataset("basis_shape_hash", (timeslots, N), dtype=np.integer)
        daset_bsi = grp_wp.create_dataset("basis_size", (timeslots, N), dtype=np.integer)
        for i in xrange(N):
            if "q" in key and not "q" in grp_pi.keys():
                daset_q_i = grp_pi.create_dataset("q_"+str(i), (timeslots, D, 1), dtype=np.complexfloating)
            if "p" in key and not "p" in grp_pi.keys():
                daset_p_i = grp_pi.create_dataset("p_"+str(i), (timeslots, D, 1), dtype=np.complexfloating)
            if "Q" in key and not "Q" in grp_pi.keys():
                daset_Q_i = grp_pi.create_dataset("Q_"+str(i), (timeslots, D, D), dtype=np.complexfloating)
            if "P" in key and not "P" in grp_pi.keys():
                daset_P_i = grp_pi.create_dataset("P_"+str(i), (timeslots, D, D), dtype=np.complexfloating)
            if "S" in key and not "S" in grp_pi.keys():
                daset_S_i = grp_pi.create_dataset("S_"+str(i), (timeslots, 1, 1), dtype=np.complexfloating)
            if "adQ" in key and not "adQ" in grp_pi.keys():
                daset_adQ_i = grp_pi.create_dataset("adQ_"+str(i), (timeslots, 1, 1), dtype=np.complexfloating)
        for i in xrange(N):
            daset_c_i = grp_ci.create_dataset("c_"+str(i), (timeslots, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,None))

        # Mark all steps as invalid
        daset_tg[...] = -1.0

    # Attach pointer to data instead timegrid
    grp_pi.attrs["pointer"] = 0
    grp_ci.attrs["pointer"] = 0


def delete_inhomogwavepacket(self, blockid=0):
    r"""Remove the stored wavepackets.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavepacket_inhomog"]
    except KeyError:
        pass


def has_inhomogwavepacket(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.
    """
    return "wavepacket_inhomog" in self._srf[self._prefixb+str(blockid)].keys()


def save_inhomogwavepacket_description(self, descr, blockid=0):
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog"
    # Save the description
    for key, value in descr.iteritems():
        # Store all the values as pickled strings because hdf can
        # only store strings or ndarrays as attributes.
        self._srf[pathd].attrs[key] = pickle.dumps(value)


def save_inhomogwavepacket_parameters(self, parameters, timestep=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Save the parameter set :math:`\Pi` of the Hagedorn wavepacket :math:`\Psi` to a file.

    :param parameters: The parameter set of the Hagedorn wavepacket.
    :type parameters: A ``list`` containing the five ``ndarrays`` like :math:`(q,p,Q,P,S)`
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/Pi/"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    for i, piset in enumerate(parameters):
        for k, item in zip(key, piset):
            self.must_resize(pathd+k+"_"+str(i), timeslot)
            self._srf[pathd+k+"_"+str(i)][timeslot,:,:] = item

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_inhomogwavepacket_coefficients(self, coefficients, basisshapes, timestep=None, blockid=0):
    r"""Save the coefficients of the Hagedorn wavepacket to a file.
    Warning: we do only save the hash of the basis shapes here!
    You have to save the basis shape with the corresponding function too.

    :param coefficients: The coefficients of the Hagedorn wavepacket.
    :type coefficients: A ``list`` with :math:`N` suitable ``ndarrays``.
    :param basisshapes: The corresponding basis shapes of the Hagedorn wavepacket.
    :type basisshapes: A ``list`` with :math:`N` :py:class:`BasisShape` subclass instances.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basis_shape_hash"
    pathbsi = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basis_size"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/coefficients/"

    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathbs, timeslot)
    self.must_resize(pathbsi, timeslot)
    for index, (bs,ci) in enumerate(zip(basisshapes, coefficients)):
        self.must_resize(pathd+"c_"+str(index), timeslot)
        size = bs.get_basis_size()
        # Do we have to resize due to changed number of coefficients
        if self._srf[pathd+"c_"+str(index)].shape[1] < size:
            self._srf[pathd+"c_"+str(index)].resize(size, axis=1)
        self._srf[pathbsi][timeslot,index] = size
        self._srf[pathbs][timeslot,index] = hash(bs)
        self._srf[pathd+"c_"+str(index)][timeslot,:size] = np.squeeze(ci)

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_inhomogwavepacket_basisshapes(self, basisshape, blockid=0):
    r"""Save the basis shapes of the Hagedorn wavepacket to a file.

    :param coefficients: The basis shapes of the Hagedorn wavepacket.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basisshapes/"

    ha = hash(basisshape)
    name = "basis_shape_"+str(ha)

    # Check if we already stored this basis shape
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


def load_inhomogwavepacket_description(self, blockid=0):
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog"

    # Load and return all descriptions available
    descr = {}
    for key, value in self._srf[pathd].attrs.iteritems():
        descr[key] = pickle.loads(value)
    return descr


def load_inhomogwavepacket_timegrid(self, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    return self._srf[pathtg][:]


def load_inhomogwavepacket_parameters(self, timestep=None, component=None, blockid=0, key=("q","p","Q","P","S")):
    r"""Load the wavepacket parameters.

    :param timestep: Load only the data of this timestep.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which parameters to load. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/Pi/"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)

    data = []
    for i in xrange(len(self._srf[pathd].keys()) // int(self._srf[pathd].attrs["number_parameters"])):
        if timestep is not None:
            data.append( tuple([ self._srf[pathd+k+"_"+str(i)][index,:,:] for k in key ]) )
        else:
            data.append( tuple([ self._srf[pathd+k+"_"+str(i)][...,:,:] for k in key ]) )
    return tuple(data)


def load_inhomogwavepacket_coefficients(self, timestep=None, get_hashes=False, component=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basis_shape_hash"
    pathbsi = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basis_size"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/coefficients/"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
    else:
        index = slice(None)

    if get_hashes is True:
        hashes = self._srf[pathbs][index,...]
        # Number of components
        N = self._srf[pathbs].shape[1]
        hashes = np.hsplit(hashes, N)

    data = []
    for i in xrange(len(self._srf[pathd].keys())):
        if timestep is not None:
            size = self._srf[pathbsi][index,i]
            data.append( self._srf[pathd+"c_"+str(i)][index,:size] )
        else:
            data.append( self._srf[pathd+"c_"+str(i)][index,...] )

    if get_hashes is True:
        return (hashes, data)
    else:
        return data


def load_inhomogwavepacket_basisshapes(self, the_hash=None, blockid=0):
    r"""Load the basis shapes by hash.
    """
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basisshapes/"

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
        # Be sure the hash is a plain number and not something
        # else like a numpy array with one element.
        the_hash = int(the_hash)
        name = "basis_shape_"+str(the_hash)
        # Check if we already stored this basis shape
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


def load_inhomogwavepacket(self, timestep, blockid=0, key=("q","p","Q","P","S")):
    r"""Load a wavepacket at a given timestep and return a fully configured instance.
    This method just calls some other :py:class:`IOManager` methods in the correct order.
    It is included only for convenience and is not particularly efficient.

    :param timestep: The timestep :math:`n` at which we load the wavepacket.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    :param blockid: The ID of the data block to operate on.
    :return: A :py:class:`HagedornWavepacketInhomogeneous` instance.
    """
    from BlockFactory import BlockFactory
    BF = BlockFactory()

    descr = self.load_inhomogwavepacket_description(blockid=blockid)
    HAWP = BF.create_wavepacket(descr)

    # Parameters and coefficients
    Pi = self.load_inhomogwavepacket_parameters(timestep=timestep, blockid=blockid, key=key)
    hashes, coeffs = self.load_inhomogwavepacket_coefficients(timestep=timestep, get_hashes=True, blockid=blockid)

    # Basis shapes
    Ks = []
    for ahash in hashes:
        K_descr = self.load_inhomogwavepacket_basisshapes(the_hash=ahash, blockid=blockid)
        Ks.append(BF.create_basis_shape(K_descr))

    # Configure the wavepacket
    HAWP.set_parameters(Pi, key=key)
    HAWP.set_basis_shapes(Ks)
    HAWP.set_coefficients(coeffs)

    return HAWP


def save_inhomogwavepacket(self, packet, timestep, blockid=None, key=("q","p","Q","P","S")):
    r"""Save a wavepacket at a given timestep and read all data to save from the
    :py:class:`HagedornWavepacketInhomogeneous` instance provided. This method just
    calls some other :py:class:`IOManager` methods in the correct order. It is included
    only for convenience and is not particularly efficient. We assume the wavepacket is
    already set up with the correct :py:meth:`add_inhomogwavepacket` method call.

    :param packet: The :py:class:`HagedornWavepacketInhomogeneous` instance we want to save.
    :param timestep: The timestep :math:`n` at which we save the wavepacket.
    :param key: Specify which parameters to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``q``, ``p``, ``Q``, ``P``, ``S`` and ``adQ``.
               Default is ``("q", "p", "Q", "P", "S")``.
    :param blockid: The ID of the data block to operate on.
    """
    # Description
    self.save_inhomogwavepacket_description(packet.get_description(), blockid=blockid)
    # Pi
    self.save_inhomogwavepacket_parameters(packet.get_parameters(key=key), timestep=timestep, blockid=blockid, key=key)
    # Basis shapes
    for shape in packet.get_basis_shapes():
        self.save_inhomogwavepacket_basisshapes(shape, blockid=blockid)
    # Coefficients
    self.save_inhomogwavepacket_coefficients(packet.get_coefficients(), packet.get_basis_shapes(), timestep=timestep, blockid=blockid)
