#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2018
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
__copyright__ = "Copyright 2018, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def neighborhood_radius_sampler(X, r_neighbors, metric="dihedral", n_components=2, n_iter=5000, n_runs=5):
    """Sample different neighborhood radius rc."""
    columns = ["run", "r_neighbor", "n_iter", "stress", "correlation"]
    data = []

    for r_neighbor in r_neighbors:
        U = Unrolr(r_neighbor, metric, n_components, n_iter)

        for i in range(n_runs):
            U.fit(X)
            data.append([i, r_neighbor, n_iter, U.stress, U.correlation])

    df = pd.DataFrame(data=data, columns=columns)

    return df


def optimization_cycle_sampler(X, n_iters, r_neighbor, metric="dihedral", n_components=2, n_runs=5):
    """Sample different number of optimization cycle with a certain
    neighborhood radius rc and compute the stress and correlation."""
    columns = ["run", "r_neighbor", "n_iter", "stress", "correlation"]
    data = []

    for n_iter in n_iters:
        U = Unrolr(r_neighbor, metric, n_components, n_iter)

        for i in range(n_runs):
            U.fit(X)
            data.append([i, r_neighbor, n_iter, U.stress, U.correlation])

    df = pd.DataFrame(data=data, columns=columns)

    return df
