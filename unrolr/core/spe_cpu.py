#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2020
# Unrolr
#
# pSPE CPU
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


import os
import sys

import numpy as np
from scipy.spatial.distance import cdist

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2020, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def _spe_dihedral(r, d, r_neighbor, n_iter=10000, learning_rate=1, verbose=0):
    """
    The CPU implementation of pSPE with dihedral_distance

    Args:
        r (ndarray): n-dimensional dataset (rows: frame; columns: angle/intramolecular distance)
        d (ndarray): projected embedding in low dim space
        r_neighbor (float): neighbor radius cutoff
        n_iter (int): number of optimization iteration (default: 10000)
        learning_rate (float): learning rate, aka computational temperature (default: 1)
        verbose (int): turn on:off verbose (default: False)

    """
    alpha = float(learning_rate) / float(n_iter)
    freq_progression = float(n_iter) / 100.
    epsilon = 1e-10
    l = 1. / r.shape[1]

    for c in range(0, n_iter + 1):
        if c % freq_progression == 0 and verbose:
            percentage = float(c) / float(n_iter) * 100.
            sys.stdout.write("\rUnrolr Optimization         : %8.3f %%" % percentage)
            sys.stdout.flush()

        # Choose random embedding (pivot)
        i = np.random.randint(r.shape[0])

        # Euclidean distance
        dijs = cdist([d[i]], d)[0]
        # Dihedral distance
        rijs = np.sqrt(l * 0.5 * np.sum((1. - np.cos(r[i] - r)), axis=1))
        # SPE
        j = (rijs <= r_neighbor) | ((rijs > r_neighbor) & (dijs < rijs))
        d[j] += (learning_rate * ((rijs[j] - dijs[j]) / (dijs[j] + epsilon)))[:, None] * (d[j] - d[i])

        learning_rate -= alpha

    if verbose:
        print()

    return d


def _spe_intramolecular(r, d, r_neighbor, n_iter=10000, learning_rate=1, verbose=0):
    """
    The CPU implementation of pSPE with intermolecular_distance

    Args:
        r (ndarray): n-dimensional dataset (rows: frame; columns: angle/intramolecular distance)
        d (ndarray): projected embedding in low dim space
        r_neighbor (float): neighbor radius cutoff
        n_iter (int): number of optimization iteration (default: 10000)
        learning_rate (float): learning rate, aka computational temperature (default: 1)
        verbose (int): turn on:off verbose (default: False)

    """
    alpha = float(learning_rate) / float(n_iter)
    freq_progression = float(n_iter) / 100.
    epsilon = 1e-10

    for c in range(0, n_iter + 1):
        j = 0

        if c % freq_progression == 0 and verbose:
            percentage = float(c) / float(n_iter) * 100.
            sys.stdout.write("\rUnrolr Optimization         : %8.3f %%" % percentage)
            sys.stdout.flush()

        # Choose random embedding (pivot)
        i = np.random.randint(r.shape[0])

        # Euclidean distance
        dijs = cdist([d[i]], d)[0]
        # Intramolecular distance
        rijs = np.sqrt(np.mean((r[i] - r)**2, axis=1))
        # SPE
        j = (rijs <= r_neighbor) | ((rijs > r_neighbor) & (dijs < rijs))
        d[j] += (learning_rate * ((rijs[j] - dijs[j]) / (dijs[j] + epsilon)))[:, None] * (d[j] - d[i])

        learning_rate -= alpha

    if verbose:
        print()

    return d


def _spe_cpu(r, d, r_neighbor, metric="dihedral", n_iter=10000, learning_rate=1., verbose=0):
    """
    The CPU implementation of pSPE (dihedral_distance/intermolecular_distance)

    Args:
        r (ndarray): n-dimensional dataset (rows: frame; columns: angle/intramolecular distance)
        d (ndarray): projected embedding in low dim space
        r_neighbor (float): neighbor radius cutoff
        metric (str): distance metric (choices: dihedral or intramolecular) (default: dihedral)
        n_iter (int): number of optimization iteration (default: 10000)
        learning_rate (float): learning rate, aka computational temperature (default: 1)
        verbose (int): turn on:off verbose (default: False)

    """
    if metric == "dihedral":
        d = _spe_dihedral(r, d, r_neighbor, n_iter, learning_rate, verbose)
    elif metric == 'intramolecular':
        d = _spe_intramolecular(r, d, r_neighbor, n_iter, learning_rate, verbose)

    return d


def _evaluate_embedding_cpu(r, d, r_neighbor, metric="dihedral", epsilon=1e-4):
    """
    Evaluate the final embedding by calculating the stress and correlation
    
    Args:
        r (ndarray): n-dimensional dataset (rows: frame; columns: angle/intramolecular distance)
        d (ndarray): the final projected embedding in low dim space
        r_neighbor (float): neighbor radius cutoff
        metric (str): distance metric (choices: dihedral or intramolecular) (default: dihedral)
        epsilon (float): convergence criteria when computing final stress and correlation (default: 1e-4)

    """
    # Ignore divide per zeros
    np.seterr(divide='ignore', invalid='ignore')

    tmp_correl = []
    sij = []
    tmp_sij_sum = 0.0
    tmp_rij_sum = 0.0
    old_stress = 999.
    old_correl = 999.
    correlation = None
    stress = None
    l = 1. / r.shape[1]

    while True:
        # Choose random conformation as pivot
        i = np.random.randint(r.shape[0])

        # Euclidean distance
        dijs = cdist([d[i]], d)[0]

        if metric == 'dihedral':
            rijs = np.sqrt(l * 0.5 * np.sum((1. - np.cos(r[i] - r)), axis=1))
        elif metric == 'intramolecular':
            rijs = np.sqrt(np.mean((r[i] - r)**2, axis=1))

        # Compute current correlation
        tmp = (np.dot(rijs.T, dijs) / rijs.shape[0]) - (np.mean(rijs) * np.mean(dijs))
        tmp_correl.append(tmp / (np.std(rijs) * np.std(dijs)))
        correlation = np.mean(tmp_correl)

        # Compute current stress
        j = (rijs <= r_neighbor) | (dijs < rijs)
        sij = ((dijs[j] - rijs[j]) * (dijs[j] - rijs[j])) / (rijs[j])
        tmp_sij_sum += np.nansum(sij)
        tmp_rij_sum += np.sum(rijs)
        stress = tmp_sij_sum / tmp_rij_sum

        # Test for convergence
        if (np.abs(old_stress - stress) < epsilon) and (np.abs(old_correl - correlation) < epsilon):
            break

        old_stress = stress
        old_correl = correlation

    # Restore numpy warnings
    np.seterr(divide='warn', invalid='warn')

    return correlation, stress
