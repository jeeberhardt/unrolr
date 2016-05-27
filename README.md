# Stochastic Proximity Embedding
Conformational clustering of MD trajectories using (pivot-based) Stochastic Proximity Embedding

## Prerequisites

You need, at a minimum:

* Python 2.7 or later
* NumPy
* H5py
* Pandas
* Matplotlib
* PyOpenCL
* MDAnalysis

## Installation

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed (NumPy, H5py, Pandas, Matplotlib).

For the rest, you just have to do this,
```bash
pip install pyopencl mdanalysis
```

## How-To

1 . First you need to extract all the C-alpha dihedral angles from your trajectory
```bash
python extract_dihedral_angles.py -t topology.psf -d traj.dcd
```
**Command line options**
* -t/--top: topology file (pdb, psf)
* -d/--dcd: single trajectory or list of trajectories (dcd, xtc)
* -s/--selection: selection commands (ex: resid 1:10)(default: all)(documentation: https://goo.gl/4t1mGb)
* -t/--dihedral: dihedral type you want extracted (choices: ca, phi, psi)(default: ca)
* -o/--ouput: output name (default: dihedral_angles.h5)

2 . Find the optimal rc parameter (and Find the optimal number of cycles) using only a small subset of conformations
```bash
python optimize.py -d dihedral_angles.h5 --rc-range 0.1 1.0 0.01 --opt-rc -i 100
python optimize.py -d dihedral_angles.h5 --rc 0.27 --opt-cycle -i 100
```

**Command line options**
* -d/--h5: HDF5 file with all the dihedral angles
* --rc-range: neighborhood Rc range values to test
* --rc: optimal neighborhood Rc value (if you want to find the optimal number of cycles)
* --run: number of SPE runs (default: 5)
* -n/--ndim: number of dimension (default: 2)
* -t/--dihedral: dihedral type you want to used (default: ca)
* --start: starting frame (default: 0)
* --stop: last frame (default: -1)
* -i/--interval: interval (default: 1)
* -o/--output: output directory (default: .)

3 . Run SPE algorithm with all the conformations
```bash
python spe.py -d dihedral_angles.h5 -c 10000 -r 0.27
```

**Command line options**
* -d/--h5: HDF5 file with all the dihedral angles
* -c/--cycles: number of optimization cycles
* -r/--rc: neighborhood rc value
* -t/--dihedral: dihedral type (choices: ca, phi, psi)(default: ca)
* -n/--ndim: number of dimension (default: 2)
* --run: number of SPE runs (default: 1)
* --start: starting frame (default: 0)
* --stop: last frame (default: -1)
* -i/--interval: interval (default: 1)
* -o/--output: output directory (default: .)
* -f/--frequency: SPE trajectory saving interval (0 if you don't want)(default: 0)
* -s/--seed: random seed (if you want to reproduce SPE results) (default: None)

## Citation
Soon ...

## License
MIT