#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2018
# Unrolr
#
# Functions to plot results from Unrolr
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2018, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def plot_sampling(fname, df, of="r_neighbor", show=True):

    xlabel = r"Neighborhood $r_{c}$"
    logx = False

    if of == "n_iter":
        xlabel = "#Cycles"
        logx = True

    fig, ax = plt.subplots(figsize=(15, 5))

    gb = df.groupby([of])
    aggregation = {"stress": [np.mean, np.std], "correlation": [np.mean, np.std]}
    gb = gb.agg(aggregation)

    gb.stress["mean"].plot(yerr=gb.stress["std"], color="crimson", logx=logx)

    ax2 = ax.twinx()

    gb.correlation["mean"].plot(yerr=gb.correlation["std"],
                                color="dodgerblue", logx=logx)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Stress", fontsize=20)
    ax.set_ylim(0, 0.2)

    ax2.set_ylabel(r"Correlation $\gamma$", fontsize=20)
    ax2.set_ylim(0, 1)

    plt.savefig(fname, dpi=300, format="png", bbox_inches="tight")

    if show:
        plt.show()
