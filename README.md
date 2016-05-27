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
**Arguments**
* -t/--top: topology file (pdb, psf)
* -d/--dcd: single trajectory or list of trajectory (dcd, xtc)
* -s/--selection: selection commands (ex: resid 1:10)(default: all)(documentation:https://goo.gl/4t1mGb)
* -t/--dihedral: dihedral type you want extracted (ca, phi or psi)(default: ca)
* -o/--ouput: output name (default: dihedral_angles.h5)

2 . Find the optimal rc parameter using only a small subset
```bash
python optimize.py -d dihedral_angles.h5 --rc-range 0.1 1.0 0.01 --opt-rc -i 100
```

3 . Find the optimal number of cycles (not necessary)
```bash
python optimize.py -d dihedral_angles.h5 --rc 0.27 --opt-cycle -i 100
```

4 . Run SPE algorithm with all the conformations
```bash
python spe.py -d dihedral_angles.h5 -c 10000 -r 0.27
```

## Citation
Soon ...

## License
MIT