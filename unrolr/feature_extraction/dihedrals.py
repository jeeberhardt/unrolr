#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2018
# Unrolr
#
# Extract calpha or phi/psi dihedral angles from trajectories
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


import argparse
from itertools import groupby
from operator import itemgetter

import h5py
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase

from ..utils import save_dataset

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2018, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


class Dihedral(AnalysisBase):
    def __init__(self, top_file, trj_files, selection="backbone", dihedral_type="calpha", **kwargs):
        # Used to store the result
        self.result = []
        # Where we will store all the atomgroups for each ca dihedral
        self._atom_ix = []

        self._u = Universe(top_file, trj_files)
        self._ag = self._u.select_atoms(selection)
        self._dihedral_type = dihedral_type
        super(Dihedral, self).__init__(self._ag.universe.trajectory, **kwargs)

    def _dihedral(self, positions):
        """ Vectorized version of the dihedral angle function
        Source: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python"""
        
        b0 = -(positions[1::4] - positions[::4])
        b1 = positions[2::4] - positions[1::4]
        b2 = positions[3::4] - positions[2::4]

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1, axis=1)[:,None]

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - np.einsum("ij,ij->i", b0, b1)[:, None] * b1
        w = b2 - np.einsum("ij,ij->i", b2, b1)[:, None] * b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.einsum("ij,ij->i", v, w)
        y = np.einsum("ij,ij->i", np.cross(b1, v), w)
        return np.arctan2(y, x)

    def _prepare(self):
        # Get list of selected segids
        segids = np.unique(self._ag.segids)

        for segid in segids:
            # Get only the segid from the selected part
            s_seg = self._ag.select_atoms("segid %s" % segid)
            # Get list of selected residus from segid
            residues = np.unique(s_seg.resnums)

            if self._dihedral_type == "calpha":
                # Identify groups of continuous residues and group them in sublist
                fragments = []

                # https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
                for k, g in groupby(enumerate(residues), lambda x: x[0] - x[1]):
                    group = (map(itemgetter(1), g))
                    group = list(map(int, group))
                    fragments.append(group)

                for fragment in fragments:
                    if len(fragment) >= 4:
                        for residu in fragment[0:-3]:
                            select_str = "(resid %s or resid %s or resid %s or resid %s) and name CA"
                            select_str = select_str % (residu, residu + 1, residu + 2, residu + 3)
                            dihedral = s_seg.select_atoms(select_str)
                            self._atom_ix.extend(list(dihedral.ix))

            elif self._dihedral_type == "backbone":
                for residu in residues[2:-2]:
                    phi = s_seg.residues[residu].phi_selection()
                    psi = s_seg.residues[residu].psi_selection()
                    self._atom_ix.extend(list(phi.ix) + list(psi.ix))

    def _single_frame(self):
        d = self._dihedral(self._u.atoms[self._atom_ix].positions)
        self.result.append(np.asarray(d, dtype=np.float32))

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

    d = Dihedral(top_file, trj_files, selection, dihedral_type).run()
    data = d.result

    save_dataset(output, "dihedral_angles", data)

if __name__ == '__main__':
    main()
