#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2020
# Unrolr
#
# Functions to sample different value of neighbor radius and optimization cycle
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT

import numpy as np
import pandas as pd

from .. import Unrolr

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2020, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def neighborhood_radius_sampler(X, r_neighbors, metric="dihedral", n_components=2, 
                                n_iter=5000, n_runs=5, init="random", platform="OpenCL"):
    """Sample different neighborhood radius rc and compute the stress and correlation.
    
    Args:
        X (ndarray): n-dimensional ndarray (rows: frames; columns: features/angles)
        r_neighbors (array-like): list of the neighborhood raidus cutoff to try
        metric (str): metric to use to compute distance between conformations (dihedral or intramolecular) (default: dihedral)
        n_components (int): number of dimension of the embedding
        n_iter (int): number of optimization cycles
        n_runs (int): number of repetitions, in order to calculate standard deviation
        init (str): method to initialize the initial embedding (random or pca)(default: random)
        platform (str): platform to use for spe (OpenCL or CPU) (default: OpenCL)

    Returns:
        results (DataFrame): Pandas DataFrame containing columns ["run", "r_neighbor", "n_iter", "stress", "correlation"]

    """
    columns = ["run", "r_neighbor", "n_iter", "stress", "correlation"]
    data = []

    for r_neighbor in r_neighbors:
        U = Unrolr(r_neighbor, metric, n_components, n_iter, init=init, platform=platform)

        for i in range(n_runs):
            U.fit_transform(X)
            data.append([i, r_neighbor, n_iter, U.stress, U.correlation])

    df = pd.DataFrame(data=data, columns=columns)

    return df


def optimization_cycle_sampler(X, n_iters, r_neighbor, metric="dihedral", n_components=2, 
                               n_runs=5, init="random", platform="OpenCL"):
    """Sample different number of optimization cycle with a certain
    neighborhood radius rc and compute the stress and correlation.
    
    Args:
        X (ndarray): n-dimensional ndarray (rows: frames; columns: features/angles)
        n_iters (array-like): list of the iteration numbers to try
        r_neighbor (float): neighborhood raidus cutoff
        metric (str): metric to use to compute distance between conformations (dihedral or intramolecular) (default: dihedral)
        n_components (int): number of dimension of the embedding
        n_runs (int): number of repetitions, in order to calculate standard deviation
        init (str): method to initialize the initial embedding (random or pca)(default: random)
        platform (str): platform to use for spe (OpenCL or CPU) (default: OpenCL)

    Returns:
        results (DataFrame): Pandas DataFrame containing columns ["run", "r_neighbor", "n_iter", "stress", "correlation"]

    """
    columns = ["run", "r_neighbor", "n_iter", "stress", "correlation"]
    data = []

    for n_iter in n_iters:
        U = Unrolr(r_neighbor, metric, n_components, n_iter, init=init, platform=platform)

        for i in range(n_runs):
            U.fit_transform(X)
            data.append([i, r_neighbor, n_iter, U.stress, U.correlation])

    df = pd.DataFrame(data=data, columns=columns)

    return df
