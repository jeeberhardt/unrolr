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

But I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed (NumPy, H5py, Pandas, Matplotlib).

For the rest, you just have to do this,
```bash
pip install pyopencl mdanalysis
```

## Tutorial

1 . First you need to extract all the C-alpha dihedral angles from your trajectory
```bash
python extract_dihedral_angles.py -t topology.psf -d traj_1.dcd
```

2 . Find the optimal rc parameter (and/or the optimal number of cycles) with a small subset
```bash
python optimize.py -d dihedral_angles.h5 --rc-range 0.1 1.0 0.01 --opt-rc -i 100
python optimize.py -d dihedral_angles.h5 --rc 0.27 --opt-cycle -i 100
```

3 . Run SPE algorithm with all the conformations
```bash
python spe.py -d dihedral_angles.h5 -c 10000 -r 0.27
```

## Citation
Soon ...

## License
MIT