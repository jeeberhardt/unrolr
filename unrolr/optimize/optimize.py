#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2017
# Unrolr
#
# Functions to find the best r_neighbors value (and n_iter) possible
# Author: Jérôme Eberhardt <qksonoe@gmail.com>
#
# License: MIT

from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import pandas as pd

from .. import Unrolr

def find_optimal_r_neighbor(X, r_parameters, n_components=2, n_iter=5000, n_runs=5):

    idx = 0
    columns = ["run", "r_neighbor", "n_iter", "stress", "correlation"]
    df = pd.DataFrame(np.nan, index=[0], columns=columns)

    r_neighbors = np.arange(r_parameters[0], r_parameters[1])

    for r_neighbor in r_neighbors:
        for i in range(n_runs):

            U = Unrolr(r_neighbor, n_components, n_iter)
            U.fit(X)

            df.loc[idx] = [i+1, r, n_iter, U.stress, U.correlation]
            idx += 1

    # There is no consensus yet on how to find the optimal
    # value of r_neighbor. So this function just returns the dataframe.

    return df

def find_optimal_n_iter(X, n_iters, r_neighbor, n_components=2, n_runs=5):

    idx = 0
    columns = ["run", "r_neighbor", "n_iter", "stress", "correlation"]
    df = pd.DataFrame(np.nan, index=[0], columns=columns)

    for n_iter in n_iters:
        for i in range(n_runs):

            U = Unrolr(r_neighbor, n_components, n_iter)
            U.fit(X)

            df.loc[idx] = [i+1, rc, n_iter, U.stress, U.correlation]
            idx += 1

    # For the moment, there is not optimization. So this function
    # just resturns the dataframe.

    return df