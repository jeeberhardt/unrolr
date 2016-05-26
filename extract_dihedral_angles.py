#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import h5py
import argparse
import numpy as np

from itertools import groupby
from operator import itemgetter
from MDAnalysis import Universe, collection, Timeseries

def identify_groups_of_continuous_numbers(data):
    groups = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        groups.append(map(itemgetter(1), g))
    return groups

def extract_dihedral_angles_from_trajectory(top_file, dcd_files, dihedral_type, selection, output):

    for dcd in dcd_files:

        # Open trajectory file
        u = Universe(top_file, dcd)

        # Get list of selected residues
        selected_residues = np.unique(u.select_atoms(selection).resnums)

        # Clear collection
        collection.clear()

        # Count different type of dihedral angles
        number_ca = 0
        number_phi = 0
        number_psi = 0

        if 'ca' in dihedral_type:

            # Identify groups of continuous number and group them in sublist (for CA dihedral)
            fragments = identify_groups_of_continuous_numbers(selected_residues)

            for fragment in fragments:
                if len(fragment) >= 4:
                    for res in fragment[0:-3]:

                        dihedral = u.select_atoms('resid %d and name CA' % res, 'resid %d and name CA' % (res + 1),
                                                  'resid %d and name CA' % (res + 2), 'resid %d and name CA' % (res + 3))

                        # Add dihedral angle to the timeseries
                        collection.addTimeseries(Timeseries.Dihedral(dihedral))

                        number_ca += 1
                else:
                    print('Warning: This fragment (%s) will be ignored because it\'s too short !' % fragment)

        if 'phi' in dihedral_type:
            for res in selected_residues:

                try:
                    dihedral = u.residues[res].phi_selection()
                    collection.addTimeseries(Timeseries.Dihedral(dihedral))
                    number_phi += 1
                except:
                    pass

        if 'psi' in dihedral_type:
            for res in selected_residues:

                try:
                    dihedral = u.residues[res].psi_selection()
                    collection.addTimeseries(Timeseries.Dihedral(dihedral))
                    number_psi += 1
                except:
                    pass

        # Iterate through trajectory and compute (see docs for start/stop/skip options)
        collection.compute(trj = u.trajectory)

        # Write all dihedral angles to HDF5
        if 'ca' in dihedral_type:
            stop = number_ca
            add_dihedral_angles_to_hdf5(output, collection.data[0:stop,:].T, 'ca')
        if 'phi' in dihedral_type:
            start = number_ca
            stop = number_ca + number_phi
            add_dihedral_angles_to_hdf5(output, collection.data[start:stop,:].T, 'phi')
        if 'psi' in dihedral_type:
            start = number_ca + number_phi
            add_dihedral_angles_to_hdf5(output, collection.data[start:,:].T, 'psi')

    # Print total number of dihedral extracted and frame
    with h5py.File(output) as f:
        for key in f.keys():
            print('Dihedral angle %4s extracted   : %10d' % (key, f[key].shape[1]))
        print('Frames used                     : %10d' % (f[key].shape[0]))

def add_dihedral_angles_to_hdf5(h5filename, data, dataname):

    with h5py.File(h5filename, 'a') as a:

        try:
            dset = a.create_dataset(dataname, (data.shape[0], data.shape[1]), maxshape = (None, data.shape[1]))
            dset[:] = data
        except:
            old_size = a[dataname].shape
            a[dataname].resize((old_size[0] + data.shape[0], data.shape[1]))
            a[dataname][old_size[0]:,] = data

        a.flush()

def parse_options():
    parser = argparse.ArgumentParser(description='Extract CA dihedral angles')
    parser.add_argument('-p', '--top', dest='top_file', required = True, \
                        action='store', type=str, \
                        help = 'topology file used for simulation (pdb, psf)')
    parser.add_argument('-d', '--dcd', dest='dcd_files', required = True, \
                        action='store', type=str, nargs='+', \
                        help = 'list of dcd files')
    parser.add_argument('-s', '--selection', dest='selection', \
                        action='store', type=str, \
                        default='backbone', help = 'residu selection')
    parser.add_argument('-t', '--dihedral', dest='dihedral_type', \
                        action='store', type=str, nargs='+', choices = ['ca', 'phi', 'psi'], \
                        default='ca', help = 'dihedral type')
    parser.add_argument('-o' '--output', dest='output', \
                        action='store', type=str, default='dihedral_angles.h5', \
                        help='directory output')

    return parser.parse_args()

def main():

    options = parse_options()

    top_file = options.top_file
    dcd_files = options.dcd_files
    selection = options.selection
    dihedral_type = options.dihedral_type
    output = options.output

    extract_dihedral_angles_from_trajectory(top_file, dcd_files, dihedral_type, selection, output)

if __name__ == '__main__':
    main()