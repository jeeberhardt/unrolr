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
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import iqr

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2018, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def _assignbins_2d(coordinates, bin_size):
    """ Create bins for 2D histogram """
    x_min, x_max = np.min(coordinates[:,0]), np.max(coordinates[:,0])
    y_min, y_max = np.min(coordinates[:,1]), np.max(coordinates[:,1])

    x_length = (x_max - x_min)
    y_length = (y_max - y_min)

    x_center = x_min + (x_length/2)
    y_center = y_min + (y_length/2)

    if x_length > y_length:
        x_limit = np.array([x_center-(x_length/2)-0.5, x_center+(x_length/2)+0.5])
        y_limit = np.array([y_center-(x_length/2)-0.5, y_center+(x_length/2)+0.5])
    else:
        x_limit = np.array([x_center-(y_length/2)-0.5, x_center+(y_length/2)+0.5])
        y_limit = np.array([y_center-(y_length/2)-0.5, y_center+(y_length/2)+0.5])

    edges_x = np.arange(float(x_limit[0]), (float(x_limit[1]) + bin_size), bin_size)
    edges_y = np.arange(float(y_limit[0]), (float(y_limit[1]) + bin_size), bin_size)

    return edges_x, edges_y


def _get_limit_histogram(hist):
    """ Find the x and y limit of the histogram. Because we don't 
    want to plot the whole histogram, but only the part where 
    there are conformations. """
    xlim = [np.nan, np.nan]
    ylim = [np.nan, np.nan]

    for i in xrange(0, hist.shape[0]):
        ix = np.where(np.isnan(hist[i,:])==False)[0]
        iy = np.where(np.isnan(hist[:,i])==False)[0]

        if ix.size:
            xlim = [np.nanmin([xlim[0], np.min(ix)]), np.nanmax([xlim[1], np.max(ix)])]
        if iy.size:
            ylim = [np.nanmin([ylim[0], np.min(iy)]), np.nanmax([ylim[1], np.max(iy)])]

    limit = [np.int(np.min([xlim[0], ylim[0]])), np.int(np.max([xlim[-1], ylim[-1]]))]
    
    return limit


def plot_embedding(fname, embedding, label="Dihedral distance", clim=None, 
                   bin_size=None, cmap="viridis", show=True):
    """ Plot 2D histogram of the embedding. The color code refers to the number
    of conformations in each bin of the histogram. """
    if bin_size is None:
        bin_size = 2. * (np.mean(iqr(embedding, axis=0)) / np.power(embedding.shape[0], 1./3))

    # Create 2D histogram
    edges_x, edges_y = _assignbins_2d(embedding, bin_size)
    hist = np.histogram2d(x=embedding[:,0], y=embedding[:,1], bins=(edges_x, edges_y))[0]
    # "Remove" all the bins without any conformations inside
    hist[hist <= 0.] = np.nan

    if clim is None:
        clim = [np.min(hist), np.max(hist)]

    limit = _get_limit_histogram(hist)

    # Make figure
    fig, ax = plt.subplots(figsize=(15., 15.))

    extent = [edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]]
    plt.imshow(hist, interpolation=None, origin="low", extent=extent, 
               vmin=clim[0], vmax=clim[1], cmap=cmap)

    ax.set_xlim(edges_x[limit[0]] - 0.1, edges_x[limit[1]] + 0.1)
    ax.set_ylim(edges_y[limit[0]] - 0.1, edges_y[limit[1]] + 0.1)
    ax.set_xlabel(label, fontsize=40)
    ax.set_ylabel(label, fontsize=40)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar_ticks = np.asarray(np.linspace(clim[0], clim[1], 10), dtype=np.int)
    cb = plt.colorbar(ticks=cbar_ticks, cax=cax)
    cb.set_label('#Conformations', size=40)
    cb.ax.tick_params(labelsize=20)
    cb.set_clim(clim[0], clim[1])

    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
