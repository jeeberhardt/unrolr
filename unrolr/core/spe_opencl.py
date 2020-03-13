#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2020
# Unrolr
#
# pSPE OpenCL
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


import os
import sys

import numpy as np
import pyopencl as cl

from ..utils import path_module

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2020, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def _read_kernel_file():
    path = path_module('unrolr')
    fname = os.path.join(path, 'core/kernel.cl')

    with open(fname) as f:
        kernel = f.read()

    return kernel


def _spe_opencl(r, d, r_neighbor, metric="dihedral", n_iter=10000, learning_rate=1., verbose=0):
    """
    The OpenCL implementation pSPE (dihedral_distance/intermolecular_distance)
    
    Args:
        r (ndarray): n-dimensional dataset (rows: frame; columns: angle/intramolecular distance)
        d (ndarray): projected embedding in low dim space
        r_neighbor (float): neighbor radius cutoff
        metric (str): distance metric (choices: dihedral or intramolecular) (default: dihedral)
        n_iter (int): number of optimization iteration (default: 10000)
        learning_rate (float): learning rate, aka computational temperature (default: 1)
        verbose (int): turn on:off verbose (default: False)

    """
    alpha = learning_rate / float(n_iter)
    freq_progression = n_iter / 100.
    r = np.ascontiguousarray(r, dtype=np.float32)
    d = np.ascontiguousarray(d.T, dtype=np.float32)

    # Create context and queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Compile kernel
    kernel = _read_kernel_file()
    program = cl.Program(ctx, kernel).build()

    # Send dihedral angles to CPU/GPU
    r_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=r)
    # Allocate space on CPU/GPU to store rij and dij
    tmp = np.zeros((r.shape[0],), dtype=np.float32)
    rij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, tmp.nbytes)
    dij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, tmp.nbytes)
    # Send initial embedding to the CPU/GPU
    d_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=d)

    for i in range(0, n_iter + 1):
        if i % freq_progression == 0 and verbose:
            percentage = float(i) / float(n_iter) * 100.
            sys.stdout.write("\rUnrolr Optimization         : %8.3f %%" % percentage)
            sys.stdout.flush()

        # Choose random embedding (pivot)
        pivot = np.int32(np.random.randint(r.shape[0]))

        if metric == 'dihedral':
            # Compute dihedral distances
            program.dihedral_distance(queue, (r.shape[0],), None, r_buf, rij_buf, 
                                      pivot, np.int32(r.shape[1])).wait()
        elif metric == 'intramolecular':
            # Compute intramolecular distance
            program.intramolecular_distance(queue, (r.shape[0],), None, r_buf, rij_buf, 
                                            pivot, np.int32(r.shape[1])).wait()

        # Compute euclidean distances
        program.euclidean_distance(queue, (d.shape[1],), None, d_buf, 
                                   dij_buf, pivot, np.int32(d.shape[1]), 
                                   np.int32(d.shape[0])).wait()
        # Stochastic Proximity Embbeding
        program.spe(queue, d.shape, None, rij_buf, dij_buf, d_buf, pivot, 
                    np.int32(d.shape[1]), np.float32(r_neighbor), 
                    np.float32(learning_rate)).wait()

        learning_rate -= alpha

    # Get the last embedding d
    cl.enqueue_copy(queue, d, d_buf)

    if verbose:
        print()

    # Return final embedding
    return d.T


def _evaluate_embedding_opencl(r, d, r_neighbor, metric="dihedral", epsilon=1e-4):
    """
    Dirty OpenCL function to evaluate the final embedding by calculating the stress and correlation
    
    Args:
        r (ndarray): n-dimensional dataset (rows: frame; columns: angle/intramolecular distance)
        d (ndarray): the final projected embedding in low dim space
        r_neighbor (float): neighbor radius cutoff
        metric (str): distance metric (choices: dihedral or intramolecular) (default: dihedral)
        epsilon (float): convergence criteria when computing final stress and correlation (default: 1e-4)

    """
    tmp_correl = []
    tmp_sij_sum = 0.0
    tmp_rij_sum = 0.0
    old_stress = 999.
    old_correl = 999.
    correlation = None
    stress = None

    r = np.ascontiguousarray(r, dtype=np.float32)
    d = np.ascontiguousarray(d.T, dtype=np.float32)

    # Creation du contexte et de la queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # On compile le kernel
    kernel = _read_kernel_file()
    program = cl.Program(ctx, kernel).build()

    # Send dihedral angles and embedding on CPU/GPU
    e_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=d)
    r_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=r)

    # Allocate memory
    rij = np.zeros((r.shape[0],), dtype=np.float32)
    dij = np.zeros((r.shape[0],), dtype=np.float32)
    rij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, rij.nbytes)
    dij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, dij.nbytes)

    sij = np.zeros((r.shape[0],), dtype=np.float32)
    sij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, sij.nbytes)

    while True:
        # Choose random conformation as pivot
        pivot = np.int32(np.random.randint(r.shape[0]))

        if metric == 'dihedral':
            # Dihedral distances
            program.dihedral_distance(queue, (r.shape[0],), None, r_buf, rij_buf,
                                       pivot, np.int32(r.shape[1])).wait()
        elif metric == 'intramolecular':
            # Dihedral distances
            program.intramolecular_distance(queue, (r.shape[0],), None, r_buf, rij_buf,
                                            pivot, np.int32(r.shape[1])).wait()

        # Euclidean distances
        program.euclidean_distance(queue, (d.shape[1],), None, e_buf,
                                    dij_buf, pivot, np.int32(d.shape[1]),
                                    np.int32(d.shape[0])).wait()
        # Compute part of stress
        program.stress(queue, (r.shape[0],), None, rij_buf, dij_buf, sij_buf,
                       np.float32(r_neighbor)).wait()

        # Get rij, dij and sij
        cl.enqueue_copy(queue, rij, rij_buf)
        cl.enqueue_copy(queue, dij, dij_buf)
        cl.enqueue_copy(queue, sij, sij_buf)

        # Compute current correlation
        tmp = (np.dot(rij.T, dij) / rij.shape[0]) - (np.mean(rij) * np.mean(dij))
        tmp_correl.append(tmp / (np.std(rij) * np.std(dij)))
        correlation = np.mean(tmp_correl)

        # Compute current stress
        tmp_sij_sum += np.nansum(sij)
        tmp_rij_sum += np.sum(rij)
        stress = tmp_sij_sum / tmp_rij_sum

        # Test for convergence
        if (np.abs(old_stress - stress) < epsilon) and (np.abs(old_correl - correlation) < epsilon):
            break

        old_stress = stress
        old_correl = correlation

    return correlation, stress
