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

## Tutorial

1 . First you need to extract all the C-alpha dihedral angles from your trajectory
```bash
python extract_dihedral_angles.py -t topology.psf -d traj_1.dcd
```
**Command line options**
* -t/--top: topology file (pdb, psf)
* -d/--dcd: single trajectory or list of trajectories (dcd, xtc)
* -s/--selection: selection commands (ex: resid 1:10)(default: all)(documentation:https://goo.gl/4t1mGb)
* -t/--dihedral: dihedral type you want extracted (ca, phi or psi)(default: ca)
* -o/--ouput: output name (default: dihedral_angles.h5)

2 . Find the optimal rc parameter (and Find the optimal number of cycles) using only a small subset
```bash
python optimize.py -d dihedral_angles.h5 --rc-range 0.1 1.0 0.01 --opt-rc -i 100
python optimize.py -d dihedral_angles.h5 --rc 0.27 --opt-cycle -i 100
```

**Command line options**
* -d/--h5: HDF5 file with dihedral angles
* --rc-range: Neighborhood Rc range values to test
* --rc: Optimal neighborhood Rc value if you want to find the optimal number of cycles
* --opt-cycle: Add it if you want to find the optimal number of cycles
* --opt-rc: Add it if you want to find the optimal neighborhood value
* --run: Number of SPE runs (default: 5)
* -n/--ndim: Number of dimension (default: 2)
* -t/--dihedral: Dihedral type you want to used (default: ca)
* --start: Use frames from this position (default: 0)
* --stop: Use frames until this position (default: -1)
* -i/--interval: Interval (default: 1)
* -o/--output: directory output (default: .)

3 . Run SPE algorithm with all the conformations
```bash
python spe.py -d dihedral_angles.h5 -c 10000 -r 0.27
```

## Citation
Soon ...

## License
MIT