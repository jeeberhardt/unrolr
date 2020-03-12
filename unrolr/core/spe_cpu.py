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


import numpy as np
from scipy.spatial.distance import cdist

from .pca import PCA
from ..utils import transform_dihedral_to_circular_mean

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2020, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def _spe_dihedral(r, d, r_neighbor, n_iter=10000, learning_rate=1, verbose=0):
    alpha = float(learning_rate) / float(n_iter)
    freq_progression = float(n_iter) / 100.
    l = 1. / r.shape[0]

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
        # Dihedral distance
        rijs = np.sqrt(l * 0.5 * np.sum((1. - np.cos(r[i] - r)), axis=1))

        for rij, dij in zip(rijs, dijs):
            if ((rij <= r_neighbor) or (rij > r_neighbor and dij < rij)) and i != j:
                d[j] = d[j] + (learning_rate * ((rij - dij) / (dij + epsilon)) * (d[j] - d[i]))
            j += 1

        learning_rate -= alpha

    if verbose:
        print()

    return d


def _spe_intramolecular(r, d, r_neighbor, n_iter=10000, learning_rate=1, verbose=0):
    alpha = float(learning_rate) / float(n_iter)
    freq_progression = float(n_iter) / 100.

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

        for rij, dij in zip(rijs, dijs):
            if ((rij <= r_neighbor) or (rij > r_neighbor and dij < rij)) and i != j:
                d[j] = d[j] + (learning_rate * ((rij - dij) / (dij + epsilon)) * (d[j] - d[i]))
            j += 1

        learning_rate -= alpha

    if verbose:
        print()

    return d


def _spe_cpu(X, r_neighbor, metric="dihedral", init=None, n_components=2, 
             n_iter=10000, learning_rate=1., verbose=0):
    """
    The Unrolr (pSPE + dihedral_distance/intermolecular_distance) method itself!
    """
    # Initialization of the embedding
    if init == "pca":
        if metric == "dihedral":
            X = transform_dihedral_to_circular_mean(X)

        pca = PCA(n_components)
        d = pca.fit_transform(X).T
    else:
        # Generate initial (random)
        d = np.random.rand(n_components, X.shape[0])

    if metric == "dihedral":
        d = _spe_dihedral(X, d, r_neighbor, n_iter, learning_rate, verbose)
    else:
        d = _spe_intramolecular(X, d, r_neighbor, n_iter, learning_rate, verbose)

    return d


def _evaluate_embedding_cpu(X, embedding, r_neighbor, metric="dihedral", epsilon=1e-4):
    """
    Dirty function to evaluate the final embedding
    """
    old_stress = 999.
    old_correl = 999.
    correlation = None
    stress = None

    """
    while True:
        # Choose random conformation as pivot
        pivot = np.random.randint(X.shape[0])

        if metric == 'dihedral':
            # Dihedral distances
        elif metric == 'intramolecular':
            # Dihedral distances

        # Test for convergence
        if (np.abs(old_stress - stress) < epsilon) and (np.abs(old_correl - correl) < epsilon):
            correlation = correl
            stress = stress
        else:
            old_stress = stress
            old_correl = correl

        break
    """

    return correlation, stress
