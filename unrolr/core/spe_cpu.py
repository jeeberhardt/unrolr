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

from .pca import PCA
from ..utils import transform_dihedral_to_circular_mean

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2020, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def _spe_cpu(X, r_neighbor, metric="dihedral", init=None, n_components=2, 
                n_iter=10000, learning_rate=1., verbose=0):
    """
    The Unrolr (pSPE + dihedral_distance/intermolecular_distance) method itself!
    """
    alpha = learning_rate / float(n_iter)
    freq_progression = n_iter / 100.

    # Initialization of the embedding
    if init == "pca":
        if metric == "dihedral":
            X = transform_dihedral_to_circular_mean(X)

        pca = PCA(n_components)
        d = pca.fit_transform(X)
        d = np.ascontiguousarray(d.T, dtype=np.float32)
    else:
        # Generate initial (random)
        d = np.float32(np.random.rand(n_components, X.shape[0]))

    for i in range(0, n_iter + 1):
        if i % freq_progression == 0 and verbose:
            percentage = float(i) / float(n_iter) * 100.
            sys.stdout.write("\rUnrolr Optimization         : %8.3f %%" % percentage)
            sys.stdout.flush()

        # Choose random embedding (pivot)
        pivot = np.int32(np.random.randint(X.shape[0]))

        if metric == 'dihedral':

        elif metric == 'intramolecular':

        learning_rate -= alpha

    if verbose:
        print()

    # Return final embedding
    return d


def _evaluate_embedding_cpu(X, embedding, r_neighbor, metric="dihedral", epsilon=1e-4):
	"""
    Dirty function to evaluate the final embedding
    """
    old_stress = 999.
    old_correl = 999.
    correlation = None
    stress = None

    while True:
        # Choose random conformation as pivot
        pivot = np.int32(np.random.randint(X.shape[0]))

        if metric == 'dihedral':
            # Dihedral distances
        elif metric == 'intramolecular':
            # Dihedral distances

        # Test for convergence
        if (np.abs(old_stress - stress) < epsilon) and (np.abs(old_correl - correl) < epsilon):
            correlation = correl
            stress = stress

            break
        else:
            old_stress = stress
            old_correl = correl

    return correlation, stress

