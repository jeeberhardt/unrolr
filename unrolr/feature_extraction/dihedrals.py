#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Jérôme Eberhardt 2016-2017
# Unrolr
#
# Extract phi/psi dihedral angles from trajectory
# Author: Jérôme Eberhardt <qksonoe@gmail.com>
#
# License: MIT

import os
import sys
import argparse
from itertools import groupby
from operator import itemgetter

import h5py
import numpy as np
from MDAnalysis import Universe, collection, Timeseries

from ..utils import save_dataset

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2017, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


def identify_groups_of_continuous_numbers(data):
    groups = []
    for k, g in groupby(enumerate(data), lambda (i, x): i - x):
        groups.append(map(itemgetter(1), g))
    return groups

def calpha_dihedrals(top_file, trj_files, selection='protein', start=0, stop=-1, skip=1):

    data = None

    for dcd in dcd_files:

        # Open trajectory file
        u = Universe(top_file, dcd)

        # Get only the selected part
        s_all = u.select_atoms(selection)
        # Get list of selected segids
        segids = np.unique(s_all.segids)

        # Clear collection
        collection.clear()

        for segid in segids:

            # Get only the segid from the selected part
            s_seg = s_all.select_atoms("segid %s" % segid)
            # Get list of selected residus from segid
            residues = np.unique(s_seg.resnums)

            # Identify groups of continuous number and group them in sublist (for CA dihedral)
            fragments = identify_groups_of_continuous_numbers(residues)

            for fragment in fragments:
                if len(fragment) >= 4:
                    for residu in fragment[0:-3]:

                        dihedral = s_seg.select_atoms("resid %d and name CA" % residu,
                                                      "resid %d and name CA" % (residu + 1),
                                                      "resid %d and name CA" % (residu + 2),
                                                      "resid %d and name CA" % (residu + 3))

                        # Add dihedral angle to the timeseries
                        collection.addTimeseries(Timeseries.Dihedral(dihedral))

            # Iterate through trajectory and compute (see docs for start/stop/skip options)
            collection.compute(trj=u.trajectory, start, stop, skip)

            if data:
                data = np.concatenate((data, collection.data))
            else:
                data = collection.data

    return data

def backbone_dihedrals(top_file, trj_files, selection='protein', start=0, stop=-1, skip=1):

    data = None

    for dcd in dcd_files:

        # Open trajectory file
        u = Universe(top_file, dcd)

        # Get only the selected part
        s_all = u.select_atoms(selection)
        # Get list of selected segids
        segids = np.unique(s_all.segids)

        # Clear collection
        collection.clear()

        for segid in segids:

            # Get only the segid from the selected part
            s_seg = s_all.select_atoms("segid %s" % segid)
            # Get list of selected residus from segid
            residues = np.unique(s_seg.resnums)

            for residu in residues[2:-1]:
                phi = s_seg.residues[residu].phi_selection()
                psi = s_seg.residues[residu].psi_selection()

                collection.addTimeseries(Timeseries.Dihedral(phi))
                collection.addTimeseries(Timeseries.Dihedral(psi))

            # Iterate through trajectory and compute (see docs for start/stop/skip options)
            collection.compute(trj=u.trajectory, start, stop, skip)

            if data:
                data = np.concatenate((data, collection.data))
            else:
                data = collection.data

    return data


def main():

    parser = argparse.ArgumentParser(description="Extract CA dihedral angles")
    parser.add_argument("-p", "--top", dest="top_file", required=True,
                        action="store", type=str,
                        help="topology file used for simulation (pdb, psf)")
    parser.add_argument("-t", "--trj", dest="trj_files", required=True,
                        action="store", type=str, nargs="+",
                        help="list of trj files")
    parser.add_argument("-s", "--selection", dest="selection",
                        default='protein', action="store", type=str,
                        default="backbone", help="residu selection")
    parser.add_argument("-d", "--dihedral", dest="dihedral_type",
                        action="store", type=str, nargs="+", choices=["calpha", "backbone"],
                        default="calpha", help="dihedral type")
    parser.add_argument("-o" "--output", dest="output",
                        action="store", type=str, default="dihedral_angles.h5",
                        help="directory output")
    options = parser.parse_args()

    top_file = options.top_file
    trj_files = options.trj_files
    selection = options.selection
    dihedral_type = options.dihedral_type
    output = options.output

    if dihedral_type == 'calpha':
        data = calpha_dihedrals(top_file, trj_files, selection)
    else:
        data = backbone_dihedrals(top_file, trj_files, selection)

    save_dataset(output, data)

if __name__ == '__main__':
    main()