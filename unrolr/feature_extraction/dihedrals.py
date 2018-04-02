#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2017
# Unrolr
#
# Extract calpha or phi/psi dihedral angles from trajectories
# Author: Jérôme Eberhardt <qksonoe@gmail.com>
#
# License: MIT


import argparse
from itertools import groupby
from operator import itemgetter

import h5py
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib import mdamath

from ..utils import save_dataset

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2018, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


class Dihedrals(AnalysisBase):
    def __init__(self, top_file, trj_files, selection='backbone', dihedral_type='calpha', **kwargs):
        u = Universe(top_file, trj_files)
        atomgroup = u.select_atoms(selection)
        super(Dihedrals, self).__init__(atomgroup.universe.trajectory, **kwargs)

        self._ag = atomgroup
        self._dihedral_type = dihedral_type

    def _prepare(self):
        self.result = []
        # Where we will store all the atomgroups for each ca dihedral
        self._ag_dihedrals = []

        # Get list of selected segids
        segids = np.unique(self._ag.segids)

        for segid in segids:
            # Get only the segid from the selected part
            s_seg = self._ag.select_atoms("segid %s" % segid)
            # Get list of selected residus from segid
            residues = np.unique(s_seg.resnums)

            if self._dihedral_type == 'calpha':
                # Identify groups of continuous residues and group them in sublist
                fragments = []
                for k, g in groupby(enumerate(residues), lambda (i, x): i - x):
                    fragments.append(map(itemgetter(1), g))

                for fragment in fragments:
                    if len(fragment) >= 4:
                        for residu in fragment[0:-3]:
                            select_str = "(resid %s or resid %s or resid %s or resid %s) and name CA"
                            select_str = select_str % (residu, residu+1, residu+2, residu+3)
                            dihedral = s_seg.select_atoms(select_str)
                            self._ag_dihedrals.append(dihedral)

            elif self._dihedral_type == 'backbone':
                for residu in residues[2:-2]:
                    phi = s_seg.residues[residu].phi_selection()
                    psi = s_seg.residues[residu].psi_selection()
                    self._ag_dihedrals.extend([phi, psi])

    def _single_frame(self):
        dihedral_angles = []

        for ag_dihedral in self._ag_dihedrals:
            a, b, c, d = ag_dihedral.atoms.positions.astype(np.float64)
            dihedral_angles.append(mdamath.dihedral(a-b, b-c, c-d))

        self.result.append(dihedral_angles)

    def _conclude(self):
        self.result = np.asarray(self.result)


def main():

    parser = argparse.ArgumentParser(description="Extract CA dihedral angles")
    parser.add_argument("-p", "--top", dest="top_file", required=True,
                        action="store", type=str,
                        help="topology file used for simulation (pdb, psf)")
    parser.add_argument("-t", "--trj", dest="trj_files", required=True,
                        action="store", type=str, nargs="+",
                        help="list of trj files")
    parser.add_argument("-s", "--selection", dest="selection",
                        action="store", type=str,
                        default="backbone", help="residu selection")
    parser.add_argument("-d", "--dihedral", dest="dihedral_type",
                        action="store", type=str, choices=['calpha', 'backbone'],
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

    d = Dihedrals(top_file, trj_files, selection, dihedral_type).run()
    data = d.result

    save_dataset(output, "dihedral_angles", data)

if __name__ == '__main__':
    main()