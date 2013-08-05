"""The WaveBlocks Project

IOM plugin providing functions for handling various
overlap matrices of linear combinations of general
wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_overlaplcwp(self, parameters, timeslots=None, matrixsize=None, blockid=0, key=("ov", "ovkin", "ovpot")):
    r"""Add storage for various overlap matrices. We can store one matrix type
    per keys.

    ========= ======
    Key name  Matrix
    ========= ======
    ``ov``    :math:`\langle\Upsilon | \Upsilon\rangle`
    ``ovkin`` :math:`\langle\Upsilon | T | \Upsilon\rangle`
    ``ovpot`` :math:`\langle\Upsilon | V(\underline{x}) | \Upsilon\rangle`
    ========= ======

    Note that 'strange' errors occur if we later try to load or save
    matrices for a type we did not initialise with this function.

    :param parameters: A :py:class:`ParameterProvider` instance. It can
                       be empty and is not used at the moment.
    :param timeslots: The number of time slots we need. Can be set to ``None``
                      to get automatically growing datasets.
    :param matrixsize: The (maximal) size of each of the overlap matrices. If specified
                       this remains fixed for all timeslots. Can be set to ``None`` (default)
                       to get automatically growing datasets.
    :type matrixsize: Pair of integers or ``None``.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which overlap matrices to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``ov``, ``ovkin`` and ``ovpot``.
               Default is ``("ov", "ovkin", "ovpot")``.
    """
    valid_keys = ("ov", "ovkin", "ovpot")

    # Create the dataset with appropriate parameters
    grp_ov = self._srf[self._prefixb+str(blockid)].create_group("overlaplcwp")

    if timeslots is None:
        T = 0
        Ts = None
    else:
        T = timeslots
        Ts = timeslots

    if matrixsize is None:
        Jr = 0
        Jc = 0
        Jrs = None
        Jcs = None
    else:
        Jr, Jc = matrixsize
        Jrs, Jcs = matrixsize

    for k in key:
        if not k in valid_keys:
            raise ValueError("Unknown key value "+str(k))

        name = k[2:]

        daset_tg = grp_ov.create_dataset("timegrid"+name, (T,), dtype=np.integer, chunks=True, maxshape=(Ts,), fillvalue=-1)
        daset_shape = grp_ov.create_dataset("shape"+name, (T,2), dtype=np.integer, chunks=True, maxshape=(Ts,2))
        daset_ov = grp_ov.create_dataset("overlap"+name, (T,Jr,Jc), dtype=np.complexfloating, chunks=True, maxshape=(Ts,Jrs,Jcs))

        daset_tg.attrs["pointer"] = 0


def delete_overlaplcwp(self, blockid=0):
    r"""Remove the stored overlap matrices.

    :param blockid: The ID of the data block to operate on.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/overlaplcwp"]
    except KeyError:
        pass


def has_overlaplcwp(self, blockid=0, key=("ov", "ovkin", "ovpot")):
    r"""Ask if the specified data block has the desired data tensor.

    :param blockid: The ID of the data block to operate on.
    :param key: Specify which overlap matrices to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``ov``, ``ovkin`` and ``ovpot``.
               Default is ``("ov", "ovkin", "ovpot")``.
    """
    r = True
    r &= ("overlaplcwp" in self._srf[self._prefixb+str(blockid)].keys())

    if r and "ov" in key:
        r &= ("overlap" in self._srf[self._prefixb+str(blockid)]["overlaplcwp"].keys())
    if r and "ovpot" in key:
        r &= ("overlappot" in self._srf[self._prefixb+str(blockid)]["overlaplcwp"].keys())
    if r and "ovkin" in key:
        r &= ("overlapkin" in self._srf[self._prefixb+str(blockid)]["overlaplcwp"].keys())

    return r


def save_overlaplcwp(self, data, timestep=None, blockid=0, key=("ov", "ovkin", "ovpot")):
    r"""Save overlap matrices of linear combinations of general wavepackets.
    In principle this function also supports non-square matrices.

    :param data: The data matrices to save.
    :type data: A list of :py:class:`ndarray` entries.
    :param timestep: The timestep at which we save the data.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which overlap matrices to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``ov``, ``ovkin`` and ``ovpot``.
               Default is ``("ov", "ovkin", "ovpot")``.
    """
    for item, datum in zip(key, data):
        if item == "ov":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegrid"
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shape"
            pathd  = "/"+self._prefixb+str(blockid)+"/overlaplcwp/overlap"
        elif item == "ovkin":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegridkin"
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shapekin"
            pathd  = "/"+self._prefixb+str(blockid)+"/overlaplcwp/overlapkin"
        elif item == "ovpot":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegridpot"
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shapepot"
            pathd  = "/"+self._prefixb+str(blockid)+"/overlaplcwp/overlappot"
        else:
            raise ValueError("Unknown key value "+str(item))

        timeslot = self._srf[pathtg].attrs["pointer"]

        # Write the data
        self.must_resize(pathd, timeslot)
        data = np.atleast_2d(np.squeeze(data))
        rows, cols = data.shape
        self.must_resize(pathd, rows-1, axis=1)
        self.must_resize(pathd, cols-1, axis=2)
        self._srf[pathd][timeslot,:rows,:cols] = data
        self.must_resize(pathsh, timeslot)
        self._srf[pathsh][timeslot,:] = np.array([rows,cols])

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self._srf[pathtg][timeslot] = timestep

        # Update the pointer
        self._srf[pathtg].attrs["pointer"] += 1


def load_overlaplcwp_timegrid(self, blockid=0, key=("ov", "ovkin", "ovpot")):
    r"""Load the timegrid corresponding to the overlap matrices specified.

    :param blockid: The ID of the data block to operate on.
    :param key: Specify which overlap matrices to load. All are independent.
    :type key: Tuple of valid identifier strings that are ``ov``, ``ovkin`` and ``ovpot``.
               Default is ``("ov", "ovkin", "ovpot")``.
    :return: A list of :py:class:`ndarray` each having one column.
    """
    tg = []
    for item in key:
        if item == "ov":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegrid"
            tg.append(self._srf[pathtg][:])
        elif item == "ovkin":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegridkin"
            tg.append(self._srf[pathtg][:])
        elif item == "ovpot":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegridpot"
            tg.append(self._srf[pathtg][:])
        else:
            raise ValueError("Unknown key value "+str(item))

    if len(tg) == 1:
        print(tg)
        return tg[0]
    else:
        return tuple(tg)


def load_overlaplcwp_shape(self, blockid=0, key=("ov", "ovkin", "ovpot")):
    r"""Load the shape of the overlap matrices specified.

    :param blockid: The ID of the data block to operate on.
    :param key: Specify which overlap matrices to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``ov``, ``ovkin`` and ``ovpot``.
               Default is ``("ov", "ovkin", "ovpot")``.
    :return: A list of :py:class:`ndarray` each having two columns.
    """
    tg = []
    for item in key:
        if item == "ov":
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shape"
            tg.append(self._srf[pathsh][:])
        elif item == "ovkin":
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shapekin"
            tg.append(self._srf[pathsh][:])
        elif item == "ovpot":
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shapepot"
            tg.append(self._srf[pathsh][:])
        else:
            raise ValueError("Unknown key value "+str(item))

    if len(tg) == 1:
        print(tg)
        return tg[0]
    else:
        return tuple(tg)


def load_overlaplcwp(self, timestep=None, blockid=0, key=("ov", "ovkin", "ovpot")):
    r"""Load overlap matrices of linear combinations of general wavepackets.

    :param timestep: Load only the data of this timestep.
    :param split: Split the data array into one array for each component.
    :param blockid: The ID of the data block to operate on.
    :param key: Specify which overlap matrices to save. All are independent.
    :type key: Tuple of valid identifier strings that are ``ov``, ``ovkin`` and ``ovpot``.
               Default is ``("ov", "ovkin", "ovpot")``.
    :return: A list of :py:class:`ndarray` items. Their shapes depend on the
             exact value of the above arguments.
    """
    result = []

    for item in key:
        if item == "ov":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegrid"
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shape"
            pathd = "/"+self._prefixb+str(blockid)+"/overlaplcwp/overlap"
        elif item == "ovkin":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegridkin"
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shapekin"
            pathd = "/"+self._prefixb+str(blockid)+"/overlaplcwp/overlapkin"
        elif item == "ovpot":
            pathtg = "/"+self._prefixb+str(blockid)+"/overlaplcwp/timegridpot"
            pathsh = "/"+self._prefixb+str(blockid)+"/overlaplcwp/shapepot"
            pathd = "/"+self._prefixb+str(blockid)+"/overlaplcwp/overlappot"
        else:
            raise ValueError("Unknown key value "+str(item))

        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            shape = self._srf[pathsh][index,:]
            datum = self._srf[pathd][index,:shape[0],:shape[1]]
        else:
            datum = self._srf[pathd][:,:,:]

        result.append(datum)

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)
