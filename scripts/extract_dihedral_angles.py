#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract all the dihedral angle from MD trajectory """

from __future__ import print_function

import os
import sys
import argparse
from itertools import groupby
from operator import itemgetter

import h5py
import numpy as np
from MDAnalysis import Universe, collection, Timeseries

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2016, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"

def identify_groups_of_continuous_numbers(data):
    groups = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        groups.append(map(itemgetter(1), g))
    return groups

def extract_dihedral_angles_from_trajectory(top_file, dcd_files, dihedral_type, selection, output):

    if os.path.isfile(output):
        print('Error: HDF5 dihedral angles file already exists!')
        sys.exit(1)

    for dcd in dcd_files:

        # Open trajectory file
        u = Universe(top_file, dcd)

        # Get only the selected part
        s_all = u.select_atoms(selection)
        # Get list of selected segids
        segids = np.unique(s_all.segids)

        # Clear collection
        collection.clear()

        # Count different type of dihedral angles
        n_ca = 0
        n_phi = 0
        n_psi = 0

        for segid in segids:

        	# Get only the segid from the selected part
        	s_seg = s.select_atoms('segid %s' % segid)
        	# Get list of selected residus from segid
        	residues = np.unique(s_seg.resnums)

	        if 'ca' in dihedral_type:

	            # Identify groups of continuous number and group them in sublist (for CA dihedral)
	            fragments = identify_groups_of_continuous_numbers(residues)

	            for fragment in fragments:
	                if len(fragment) >= 4:
	                    for residu in fragment[0:-3]:

	                        dihedral = s_seg.select_atoms('resid %d and name CA' % residu, 
	                        						      'resid %d and name CA' % (residu + 1),
	                                                      'resid %d and name CA' % (residu + 2), 
	                                                      'resid %d and name CA' % (residu + 3))

	                        # Add dihedral angle to the timeseries
	                        collection.addTimeseries(Timeseries.Dihedral(dihedral))

	                        n_ca += 1
	                else:
	                    print('Warning: This fragment (%s) will be ignored because it\'s too short !' % fragment)

	        if 'phi' in dihedral_type:
	            for residu in residues:

	                try:
	                    dihedral = s_seg.residues[residu].phi_selection()
	                    collection.addTimeseries(Timeseries.Dihedral(dihedral))
	                    n_phi += 1
	                except:
	                    pass

	        if 'psi' in dihedral_type:
	            for residu in residues:

	                try:
	                    dihedral = s_seg.residues[residu].psi_selection()
	                    collection.addTimeseries(Timeseries.Dihedral(dihedral))
	                    n_psi += 1
	                except:
	                    pass

        # Iterate through trajectory and compute (see docs for start/stop/skip options)
        collection.compute(trj=u.trajectory)

        # Write all dihedral angles to HDF5
        if 'ca' in dihedral_type:
            stop = n_ca
            add_dihedral_angles_to_hdf5(output, collection.data[0:stop,:].T, 'ca')
        if 'phi' in dihedral_type:
            start = n_ca
            stop = n_ca + n_phi
            add_dihedral_angles_to_hdf5(output, collection.data[start:stop,:].T, 'phi')
        if 'psi' in dihedral_type:
            start = n_ca + n_phi
            add_dihedral_angles_to_hdf5(output, collection.data[start:,:].T, 'psi')

    # Print total number of dihedral extracted and frame
    with h5py.File(output) as f:
        for key in f.keys():
            print('Dihedral angle %4s extracted   : %10d' % (key, f[key].shape[1]))
        print('Frames used                     : %10d' % (f[key].shape[0]))

def add_dihedral_angles_to_hdf5(h5filename, data, dataname):

    with h5py.File(h5filename, 'a') as a:

        try:
            dset = a.create_dataset(dataname, (data.shape[0], data.shape[1]), maxshape=(None, data.shape[1]))
            dset[:] = data
        except:
            old_size = a[dataname].shape
            a[dataname].resize((old_size[0] + data.shape[0], data.shape[1]))
            a[dataname][old_size[0]:,] = data

        a.flush()

def parse_options():
    parser = argparse.ArgumentParser(description='Extract CA dihedral angles')
    parser.add_argument('-p', '--top', dest='top_file', required=True,
                        action='store', type=str,
                        help = 'topology file used for simulation (pdb, psf)')
    parser.add_argument('-d', '--dcd', dest='dcd_files', required=True,
                        action='store', type=str, nargs='+',
                        help = 'list of dcd files')
    parser.add_argument('-s', '--selection', dest='selection',
                        action='store', type=str,
                        default='backbone', help='residu selection')
    parser.add_argument('-t', '--dihedral', dest='dihedral_type',
                        action='store', type=str, nargs='+', choices=['ca', 'phi', 'psi'],
                        default='ca', help='dihedral type')
    parser.add_argument('-o' '--output', dest='output',
                        action='store', type=str, default='dihedral_angles.h5',
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
