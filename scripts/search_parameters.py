#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Help to find the optimal neighbourhood radius rc and minimal number of cycle """

from __future__ import print_function

import os
import sys
import argparse

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spe import SPE

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2016, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"

# pd.set_option("display.max_rows", None)


def plot_result(df, groupby, xlabel, fig_name, logx=False):

    fig, ax = plt.subplots(figsize=(15, 5))

    gb = df.groupby([groupby])
    aggregation = {"stress": {"mean": np.mean, "std": np.std},
                   "correlation": {"mean": np.mean, "std": np.std}}
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

    plt.savefig(fig_name, dpi=300, format="png", bbox_inches="tight")


def parse_options():
    parser = argparse.ArgumentParser(description="Optimize SPE parameters")
    parser.add_argument("-d", "--h5", dest="hdf5filename", required=True,
                        action="store", type=str,
                        help="HDF5 file with dihedral angles")
    parser.add_argument("-r", "--rc", dest="rc",
                        action="store", type=float, nargs="+",
                        default=None,
                        help="rc value or rc range [0.1 1 0.01]")
    parser.add_argument("--run", dest="runs",
                        action="store", type=int, default=5,
                        help="number of spe runs")
    parser.add_argument("-n", "--ndim", dest="ndim",
                        action="store", type=int, default=2,
                        help="number of dimension")
    parser.add_argument("-t", "--dihedral", dest="dihedral_type",
                        action="store", type=str, nargs="+",
                        choices=["ca", "phi", "psi"],
                        default="ca", help="dihedral type")
    parser.add_argument("--start", dest="start",
                        action="store", type=int, default=0,
                        help="used frames from this position")
    parser.add_argument("--stop", dest="stop",
                        action="store", type=int, default=-1,
                        help="used frames until this position")
    parser.add_argument("-i", "--interval", dest="interval",
                        action="store", type=int, default=1,
                        help="used frames at this interval")
    parser.add_argument("-o" "--output", dest="output",
                        action="store", type=str, default=".",
                        help="directory output")

    return parser.parse_args()


def main():
    options = parse_options()

    dihe_file = options.hdf5filename
    rc = options.rc
    ndim = options.ndim
    start = options.start
    stop = options.stop
    interval = options.interval
    output = options.output
    runs = options.runs
    dihe_type = options.dihedral_type

    S = SPE(dihe_file, dihe_type, start, stop, interval)

    # If there is only one value, it means we want to test multiple values of cycle
    if len(rc) == 1:
        # Get rc value
        rc = rc[0]

        columns = ["run", "rc", "cycle", "stress", "correlation"]
        df_cycle = pd.DataFrame(np.nan, index=[0], columns=columns)
        idx = 0

        cycles = (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000)

        for cycle in cycles:

            print("# Run with rc = %4f cycle = %d" % (rc, cycle))

            for i in xrange(runs):

                S.fit(rc, cycle, ndim)
                S.save("%s/spe_optimization" % output)

                df_cycle.loc[idx] = [i+1, rc, cycle, S.stress, S.correlation]

                idx += 1

        param_str = "rc_%s_i_%s" % (rc, interval)

        # Plot result
        f_name = "%s/spe_optimization/cycle_vs_stress-correlation_%s.png" % (output, param_str)
        plot_result(df_cycle, "cycle", "Cycles", f_name, True)

        # Write result to csv file
        f_name = "%s/spe_optimization/cycle_vs_stress-correlation_%s.csv" % (output, param_str)
        df_cycle.to_csv(f_name, index=False)

    # If there is 3 values, it means we have to test multiple values of rc
    elif len(rc) == 3:

        df_rc = pd.DataFrame(np.nan, index=[0], columns=["run", "rc", "stress", "correlation"])
        idx = 0

        for r in np.arange(rc[0], rc[1]+rc[2], rc[2]):
            print("# Run with rc = %4f" % r)

            for i in xrange(runs):

                S.fit(r, 5000, ndim)
                S.save("%s/spe_optimization" % output)

                df_rc.loc[idx] = [i+1, r, S.stress, S.correlation]

                idx += 1

        param_str = "rc_%s_%s_%s_i_%s" % (rc[0], rc[1], rc[2], interval)

        # Plot result
        fig_name = "%s/spe_optimization/rc_vs_stress-correlation_%s.png" % (output, param_str)
        plot_result(df_rc, "rc", r"Neighborhood $r_{c}$", fig_name)

        # Write result to csv file
        file_name = "%s/spe_optimization/rc_vs_stress-correlation_%s.csv" % (output, param_str)
        df_rc.to_csv(file_name, index=False)

    else:
        print("Error: You need to specify at least a RC value (0.1) or a range of RC values (0.1 1 0.1)!")
        sys.exit(1)

if __name__ == "__main__":
    main()
