#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2018
# Unrolr
#
# Utils functions
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


from __future__ import print_function

import os
import sys

import h5py
import pyopencl as cl
import numpy as np

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2018, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def read_dataset(fname, dname, start=0, stop=-1, skip=1):
    """Read dataset from HDF5 file."""
    data = None

    try:
        with h5py.File(fname, 'r') as f:
            if stop == -1:
                return f[dname][start::skip,]
            else:
                return f[dname][start:stop:skip,]
    except IOError:
        print("Error: cannot find file %s." % fname)

    return data


def save_dataset(fname, dname, data):
    """Save dataset to HDF5 file."""
    with h5py.File(fname, 'w') as w:
        try:
            dset = w.create_dataset(dname, (data.shape[0], data.shape[1]))
            dset[:] = data
        except:
            pass

        w.flush()


def is_opencl_env_defined():
    """Check if OpenCL env. variable is defined."""
    variable_name = "PYOPENCL_CTX"
    if os.environ.get(variable_name):
        return True
    else:
        return False


def max_conformations_from_dataset(fname, dname):
    """Get maximum number of conformations that can fit
    into the memory of the selected OpenCL device and
    also the step/interval """
    if not is_opencl_env_defined():
        print("Error: The environnment variable PYOPENCL_CTX is not defined.")
        print("Tip: python -c \"import pyopencl as cl; cl.create_some_context()\"")
        sys.exit(1)

    ctx = cl.create_some_context()
    max_size = int(ctx.devices[0].max_mem_alloc_size)

    try:
        with h5py.File(fname, 'r') as f:
            bytes_size = f[dname].dtype.itemsize
            n_conf, n_dim = f[dname].shape
            data_size = bytes_size * n_conf * n_dim
    except IOError:
        print("Error: cannot find file %s." % fname)

    if data_size > max_size:
        """ Return the first interval that produces a dataset
        with a size inferior than max_size """
        for i in range(1, n_conf):
            if n_conf % i == 0:
                tmp_size = (n_conf / i) * n_dim * bytes_size
                if tmp_size <= max_size:
                    return (n_conf / i, i)

        # Return None if we didn't find anything
        return (None, None)
    else:
        return (data_shape[0], 1)
