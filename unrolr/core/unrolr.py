#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2020
# Unrolr
#
# The core of Unrolr (pSPE + dihedral distance as metric)
# Author: Jérôme Eberhardt <qksoneo@gmail.com>
#
# License: MIT


from __future__ import print_function

import sys
import argparse

import numpy as np

from .pca import PCA
from .spe_opencl import _spe_opencl, _evaluate_embedding_opencl
from .spe_cpu import _spe_cpu, _evaluate_embedding_cpu
from ..utils import is_opencl_env_defined
from ..utils import read_dataset
from ..utils import transform_dihedral_to_circular_mean

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2020, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


class Unrolr():

    def __init__(self, r_neighbor, metric="dihedral", n_components=2, n_iter=10000,
                 random_seed=None, init="random", learning_rate=1., epsilon=1e-4, 
                 verbose=0, platform="OpenCL"):
        """Initialize Unrolr object.
        
        Args:
            r_neighbor (float): neighbor radius cutoff
            metric (str): distance metric (choices: dihedral or intramolecular) (default: dihedral)
            n_component (int): number of component of the final embedding (default: 2)
            n_iter (int): number of optimization iteration (default: 10000)
            random_seed (int): random seed (default: None)
            init (str): method to initialize the initial embedding (random or pca)(default: random)
            learning_rate (float): learning rate, aka computational temperature (default: 1)
            epsilon (float): convergence criteria when computing final stress and correlation (default: 1e-4)
            verbose (int): turn on:off verbose (default: False)

        """
        # pSPE parameters
        self._n_components = n_components
        self._r_neighbor = r_neighbor
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._metric = str(metric).lower()
        # Set numpy random state and verbose
        self._init = str(init).lower()
        self._random_seed = self._set_random_state(random_seed)
        self._verbose = verbose
        self._platform = str(platform).lower()
        # Output variable
        self.embedding = None
        self.stress = None
        self.correlation = None

        # Check PYOPENCL_CTX environnment variable
        if self._platform == "opencl":
            if not is_opencl_env_defined():
                print("Error: The environnment variable PYOPENCL_CTX is not defined !")
                print("Tip: python -c \"import pyopencl as cl; cl.create_some_context()\"")
                sys.exit(1)

    def _set_random_state(self, seed=None):
        """
        Set Random state (seed)
        """
        if not seed:
            seed = np.random.randint(low=1, high=1E6, size=1)[0]

        np.random.seed(seed=seed)

        return seed

    def fit_transform(self, r):
        """Run the Unrolr (pSPE + didhedral distance) method.
        
        Args:
            r (ndarray): n-dimensional dataset (rows: frame; columns: angle)

        """
        # Initialization of the embedding
        if self._init == "pca":
            if self._metric == "dihedral":
                r = transform_dihedral_to_circular_mean(r)

            pca = PCA(self._n_components)
            d = pca.fit_transform(r)
        else:
            # Generate initial (random)
            d = np.random.rand(r.shape[0], self._n_components)

        if self._platform == "opencl":
            _spe = _spe_opencl
            _evaluate_embedding = _evaluate_embedding_opencl
        else:
            _spe = _spe_cpu
            _evaluate_embedding = _evaluate_embedding_cpu

        # Fire off SPE calculation !!
        self.embedding = _spe(r, d, self._r_neighbor, self._metric, self._init, 
                              self._n_components, self._n_iter, self._learning_rate,
                              self._verbose)

        # Evaluation embedding
        correlation, stress = _evaluate_embedding(r, self.embedding, self._r_neighbor, 
                                                  self._metric, self._epsilon)

        self.stress = np.around(stress, decimals=4)
        self.correlation = np.around(correlation, decimals=4) 

    def save(self, fname="embedding.csv", frames=None):
        """Save all the data
        
        Args:
            fname (str): pathname of the csv file containing the final embedding (default: embedding.csv)
            frames (array-like): 1d-array containing frame numbers (Default: None)

        """
        fmt = ""

        if frames is not None:
            # Add frame idx to embedding
            self.embedding = np.column_stack((frames, self.embedding))
            fmt = "%012d,"

        # Create header and format
        header = "r_neighbor %s n_iter %s" %(self._r_neighbor, self._n_iter)
        header += " stress %s correlation %s" % (self.stress, self.correlation)
        header += " seed %s" % self._random_seed
        fmt += "%.5f" + (self._n_components - 1) * ",%.5f"

        # Save final embedding to txt file
        np.savetxt(fname, self.embedding, fmt=fmt, header=header)


def main():
    """Main function, unrolr.py can be executed as a standalone script
    
    Args:
        -f/--dihedral (filename): hdf5 file containing dihedral angles
        -r/--rc (float): neighborhood radius cutoff (default: 1)
        -n/-ndim (int): number of dimension of the final embedding (default: 2)
        -c/--cycles (int): number of optimization iteration (default: 1000)
        --start (int): index of the first frame to analyze (default: 1)
        --stop (int): index of the last frame to analyze (default: -1)
        --skip (int): number of frame to skip (default: 1)
        -o/--output (filename): csv output file name (default: embedding.csv)
        -s/--seed: random seed (default: None)

    Returns:
        output (file): csv file containing the final embedding (default: embedding.csv)

    """
    parser = argparse.ArgumentParser(description="Unrolr")
    parser.add_argument("-f", "--dihedral", dest="fname", required=True,
                        action="store", type=str,
                        help="HDF5 file with dihedral angles")
    parser.add_argument("-r", "--rc", dest="r_neighbor",
                        action="store", type=float, default=1.,
                        help="neighborhood cutoff")
    parser.add_argument("-n", "--ndim", dest="n_components",
                        action="store", type=int, default=2,
                        help="number of dimension")
    parser.add_argument("-c", "--cycles", dest="n_iter",
                        action="store", type=int, default=1000,
                        help="number of cycle")
    parser.add_argument("--start", dest="start",
                        action="store", type=int, default=0,
                        help="used frames from this position")
    parser.add_argument("--stop", dest="stop",
                        action="store", type=int, default=-1,
                        help="used frames until this position")
    parser.add_argument("-skip", "--skip", dest="skip",
                        action="store", type=int, default=1,
                        help="used frames at this interval")
    parser.add_argument("-o", "--output", dest="output",
                        action="store", type=str, default="embedding.csv",
                        help="output csv file")
    parser.add_argument("-s", "--seed", dest="random_seed",
                        action="store", type=int, default=None,
                        help="If you want to reproduce spe trajectory")

    options = parser.parse_args()

    fname = options.fname
    n_iter = options.n_iter
    r_neighbor = options.r_neighbor
    n_components = options.n_components
    start = options.start
    stop = options.stop
    skip = options.skip
    output = options.output
    random_seed = options.random_seed

    X = read_dataset(fname, "dihedral_angles", start, stop, skip)

    U = Unrolr(r_neighbor, n_components, n_iter, random_seed, verbose=1)
    U.fit_transform(X)

    print("Random seed              : %8d" % U.random_seed)
    print("Stress                   : %8.3f" % U.stress)
    print("Correlation              : %8.3f" % U.correlation)

    frames = np.arange(start, X.shape[0], skip)
    U.save(output, frames)

if __name__ == "__main__":
    main()
