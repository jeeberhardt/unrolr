#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2018
# Unrolr
#
# Extract intramolecular distances from trajectories
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


import argparse

import h5py
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.distances import self_distance_array

from ..utils import save_dataset

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2018, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


class IntramolecularDistance(AnalysisBase):
    def __init__(self, top_file, trj_files, selection='backbone', **kwargs):
        """Create IntramolecularDistance analysis object.
        
        Args:
            top_file (str): filename of the topology file
            trj_files (str or array-like): one or a list of trajectory files
            selection (str): protein selection (default: backbone)

        """
        # Used to store the result
        self.result = []

        self._u = Universe(top_file, trj_files)
        self._ag = self._u.select_atoms(selection)
        super(IntramolecularDistance, self).__init__(self._ag.universe.trajectory, **kwargs)

    def _single_frame(self):
        d = self_distance_array(self._ag.positions)
        self.result.append(np.asarray(d, dtype=np.float32))

    def _conclude(self):
        self.result = np.asarray(self.result)


def main():
    """Main function, intramolecular_distances.py can be executed as a standalone script
    
    Args:
        -p/--top (filename): topology file used for simulation (pdb, psf)
        -t/--trj (filename): one or list of trajectory files
        -s/--selection (str): protein selection
        -o/--output (filename): hdf5 output file name (default: intramolecular_distances.h5)

    Returns:
        output (file): hdf5 file containing the intramolecular distances (default: intramolecular_distances.h5)

    """

    parser = argparse.ArgumentParser(description="Extract intramolecular distances")
    parser.add_argument("-p", "--top", dest="top_file", required=True,
                        action="store", type=str,
                        help="topology file used for simulation (pdb, psf)")
    parser.add_argument("-t", "--trj", dest="trj_files", required=True,
                        action="store", type=str, nargs="+",
                        help="list of trj files")
    parser.add_argument("-s", "--selection", dest="selection",
                        action="store", type=str,
                        default="backbone", help="residu selection")
    parser.add_argument("-o" "--output", dest="output",
                        action="store", type=str, default="intramolecular_distances.h5",
                        help="directory output")
    options = parser.parse_args()

    top_file = options.top_file
    trj_files = options.trj_files
    selection = options.selection
    output = options.output

    i = IntramolecularDistance(top_file, trj_files, selection).run()
    data = i.result

    save_dataset(output, "intramolecular_distances", data)

if __name__ == '__main__':
    main()
