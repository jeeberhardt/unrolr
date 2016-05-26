#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spe import SPE

#pd.set_option('display.max_rows', None)

def plot_result(df, groupby, xlabel, fig_name, logx=False):

    fig, ax = plt.subplots(figsize=(15, 5))

    gb = df.groupby([groupby])
    aggregation = {'stress' : {'mean' : np.mean, 'std': np.std}, 
                   'correlation' : {'mean' : np.mean, 'std' : np.std}}
    gb = gb.agg(aggregation)
    
    gb.stress['mean'].plot(yerr=gb.stress['std'], color='crimson', logx=logx)

    ax2 = ax.twinx()

    gb.correlation['mean'].plot(yerr=gb.correlation['std'], 
                                color='dodgerblue', logx=logx)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Stress", fontsize=20)
    ax.set_ylim(0, 0.2)

    ax2.set_ylabel(r"Correlation $\gamma$", fontsize=20)
    ax2.set_ylim(0, 1)

    plt.savefig(fig_name, dpi=300, format='png', bbox_inches='tight')

def parse_options():
    parser = argparse.ArgumentParser(description='Optimize SPE parameters')
    parser.add_argument('-d', '--h5', dest='hdf5filename', required = True,
                        action='store', type=str,
                        help='HDF5 file with dihedral angles')
    parser.add_argument('--rc-range', dest='rc_range',
                        action='store', type=float, nargs=3, 
                        default=[0.1, 0.5, 0.01],
                        help='range values to test for neighborhood rc')
    parser.add_argument('--rc', dest='rc',
                        action='store', type=float, default= 0.3,
                        help='value for neighborhood rc (opt_cycle)')
    parser.add_argument('--opt-cycle', dest='opt_cycle',
                        action='store_true', default=False,
                        help='test multiple value of cycles for a giving rc')
    parser.add_argument('--opt-rc', dest='opt_rc',
                        action='store_true', default=False,
                        help='test multiple value of neighborhood rc')
    parser.add_argument('--run', dest='runs',
                        action='store', type=int, default=5,
                        help='number of spe runs')
    parser.add_argument('-n', '--ndim', dest='ndim',
                        action='store', type=int, default=2,
                        help='number of dimension')
    parser.add_argument('-t', '--dihedral', dest='dihedral_type',
                        action='store', type=str, nargs='+', 
                        choices=['ca', 'phi', 'psi'],
                        default='ca', help='dihedral type')
    parser.add_argument('--start', dest='start',
                        action='store', type=int, default=0,
                        help='used frames from this position')
    parser.add_argument('--stop', dest='stop',
                        action='store', type=int, default=-1,
                        help='used frames until this position')
    parser.add_argument('-i', '--interval', dest='interval',
                        action='store', type=int, default=1,
                        help='used frames at this interval')
    parser.add_argument('-o' '--output', dest='output',
                        action='store', type=str, default='.',
                        help='directory output')

    return parser.parse_args()

def main():
    options = parse_options()

    hdf5filename = options.hdf5filename
    rc_range = options.rc_range
    rc = options.rc
    opt_cycle = options.opt_cycle
    opt_rc = options.opt_rc
    ndim = options.ndim
    start = options.start
    stop = options.stop
    interval = options.interval
    output = options.output
    runs = options.runs
    dihedral_type = options.dihedral_type

    # We test the influence of the neighborhood rc on the stress and the correlation
    if opt_rc:

        df_rc = pd.DataFrame(np.nan, index =[0], 
                             columns=['run', 'rc', 'stress', 'correlation'])
        idx = 0

        current_rc = rc_range[0]

        while current_rc < (rc_range[1] + rc_range[2]):

            print('# Run with rc = %4f' % current_rc)

            for i in xrange(runs):

                spe = SPE(5000, current_rc, ndim)
                spe.fit(hdf5filename, dihedral_type, start, stop, interval, 
                        '%s/spe_optimization' % output, 0)

                df_rc.loc[idx] = [i+1, current_rc, spe.stress, spe.correlation]

                idx += 1

            current_rc += rc_range[2]

        # Plot result
        plot_result(df_rc, 'rc', r"Neighborhood $r_{c}$", 
                    "%s/spe_optimization/rc_vs_stress-correlation.png" % output)
        # Write result to csv file
        df_rc.to_csv('%s/spe_optimization/rc_vs_stress-correlation.csv' % output, 
                     index=False)

    # Now we test the influence of the number of cycle on the stress and the correlation
    if opt_cycle:

        columns = ['run', 'rc', 'cycle', 'stress', 'correlation']
        df_cycle = pd.DataFrame(np.nan, index=[0], columns=columns)
        idx = 0

        cycles = (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000)

        for cycle in cycles:

            print('# Run with rc = %4f cycle = %d' % (rc, cycle))

            for i in xrange(runs):

                spe = SPE(cycle, rc, ndim)
                spe.fit(hdf5filename, dihedral_type, start, stop, interval, 
                        '%s/spe_optimization' % output, 0)

                df_cycle.loc[idx] = [i+1, rc, cycle, spe.stress, spe.correlation]

                idx += 1

        # Plot result
        f_name = "%s/spe_optimization/cycle_vs_stress-correlation.png" % output
        plot_result(df_cycle, 'cycle', 'Cycles', f_name, True)

        # Write result to csv file
        f_name = '%s/spe_optimization/cycle_vs_stress-correlation.csv' % output
        df_cycle.to_csv(f_name, index=False)

if __name__ == '__main__':
    main()