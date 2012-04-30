"""The WaveBlocks Project

IOM plugin providing functions for handling
homogeneous Hagedorn wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import pickle
import numpy as np


def add_wavepacket(self, parameters, timeslots=None, blockid=0):
    r"""Add storage for the homogeneous wavepackets.

    :param parameters: An :py:class:`ParameterProvider` instance with at least the keys ``dimension``, ``basis_size`` and ``ncomponents``.
    """
    # If we run with an adaptive basis size, then we must make the data tensor size maximal
    # TODO: Ugly, improve:
    if parameters.has_key("max_basis_size"):
        bs = parameters["max_basis_size"]
    else:
        bs = np.max(parameters["basis_size"])

    N = parameters["ncomponents"]
    D = parameters["dimension"]

    # The overall group containing all wavepacket data
    grp_wp = self._srf[self._prefixb+str(blockid)].require_group("wavepacket")
    # The group for storing the basis shapes
    grp_bs = grp_wp.create_group("basisshapes")
    # The group for storing the parameter set Pi
    grp_pi = grp_wp.create_group("Pi")
    # The group for storing the coefficients
    grp_ci = grp_wp.create_group("coefficients")

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_tg = grp_wp.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_bs = grp_wp.create_dataset("basis_shape_hash", (0, N), dtype=np.integer, chunks=True, maxshape=(None,N))
        daset_bsi = grp_wp.create_dataset("basis_size", (0, N), dtype=np.integer, chunks=True, maxshape=(None,N))
        daset_q = grp_pi.create_dataset("q", (0, D, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,D,1))
        daset_p = grp_pi.create_dataset("p", (0, D, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,D,1))
        daset_Q = grp_pi.create_dataset("Q", (0, D, D), dtype=np.complexfloating, chunks=True, maxshape=(None,D,D))
        daset_P = grp_pi.create_dataset("P", (0, D, D), dtype=np.complexfloating, chunks=True, maxshape=(None,D,D))
        daset_S = grp_pi.create_dataset("S", (0, 1, 1), dtype=np.complexfloating, chunks=True, maxshape=(None,1,1))
        for i in xrange(N):
            daset_c_i = grp_ci.create_dataset("c_"+str(i), (0, bs), dtype=np.complexfloating, chunks=True, maxshape=(None,bs))
    else:
        # User specified how much space is necessary.
        daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), dtype=np.integer)
        daset_bs = grp_wp.create_dataset("basis_shape_hash", (timeslots, N), dtype=np.integer)
        daset_bsi = grp_wp.create_dataset("basis_size", (timeslots, N), dtype=np.integer)
        daset_q = grp_pi.create_dataset("q", (timeslots, D, 1), dtype=np.complexfloating)
        daset_p = grp_pi.create_dataset("p", (timeslots, D, 1), dtype=np.complexfloating)
        daset_Q = grp_pi.create_dataset("Q", (timeslots, D, D), dtype=np.complexfloating)
        daset_P = grp_pi.create_dataset("P", (timeslots, D, D), dtype=np.complexfloating)
        daset_S = grp_pi.create_dataset("S", (timeslots, 1, 1), dtype=np.complexfloating)
        for i in xrange(N):
            daset_c_i = grp_ci.create_dataset("c_"+str(i), (timeslots, bs), dtype=np.complexfloating)

    # Attach pointer to data instead timegrid
    grp_pi.attrs["pointer"] = 0
    grp_ci.attrs["pointer"] = 0


def delete_wavepacket(self, blockid=0):
    r"""Remove the stored wavepackets.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavepacket"]
    except KeyError:
        pass


def has_wavepacket(self, blockid=0):
    r"""Ask if the specified data block has the desired data tensor.
    """
    return "wavepacket" in self._srf[self._prefixb+str(blockid)].keys()


def save_wavepacket_description(self, descr, blockid=0):
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket"
    # Save the description
    for key, value in descr.iteritems():
        # Store all the values as pickled strings because hdf can
        # only store strings or ndarrays as attributes.
        self._srf[pathd].attrs[key] = pickle.dumps(value)


def save_wavepacket_parameters(self, parameters, timestep=None, blockid=0):
    r"""Save the parameter set :math:`\Pi` of the Hagedorn wavepacket :math:`\Psi` to a file.

    :param parameters: The parameter set of the Hagedorn wavepacket.
    :type parameters: A ``list`` containing the five ``ndarrays`` like :math:`(q,p,Q,P,S)`
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/Pi/"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    for key, item in zip(("q","p","Q","P","S"), parameters):
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

    :param coefficients: The basis shapes of the Hagedorn wavepacket.
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
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket"

    # Load and return all descriptions available
    descr = {}
    for key, value in self._srf[pathd].attrs.iteritems():
        descr[key] = pickle.loads(value)
    return descr


def load_wavepacket_timegrid(self, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    return self._srf[pathtg][:]


def load_wavepacket_parameters(self, timestep=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/Pi/"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        params = tuple([ self._srf[pathd+key][index,:,:] for key in ("q","p","Q","P","S") ])
    else:
        params = tuple([ self._srf[pathd+key][...,:,:] for key in ("q","p","Q","P","S") ])

    return params


def load_wavepacket_coefficients(self, timestep=None, get_hashes=False, component=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_shape_hash"
    pathbsi = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_size"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/coefficients/"

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
            size = self._srf[pathbsi][timestep,i]
            data.append( self._srf[pathd+"c_"+str(i)][index,:size] )
        else:
            data.append( self._srf[pathd+"c_"+str(i)][index,...] )

    if get_hashes is True:
        return (hashes, data)
    else:
        return data


def load_wavepacket_basisshapes(self, the_hash=None, blockid=0):
    r"""Load the basis shapes by hash.
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
