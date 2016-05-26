# Stochastic Proximity Embedding
Conformational clustering of MD trajectories using (pivot-based) Stochastic Proximity Embedding

## Prerequisites

You need, at a minimum:

* Python 2.7
* NumPy
* H5py
* Pandas
* Matplotlib
* PyOpenCL
* MDAnalysis

But I highly you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed (NumPy, H5py, Pandas, Matplotlib).

For the rest, you just have to do this,
```bash
pip install pyopencl mdanalysis
```

## Tutorial

1 . First you need to extract all the dihedral angles from your trajectory
```bash
python extract_dihedral_angles.py -t topology.psf -d traj_1.dcd -t ca
```

2 . Find the optimal rc parameter
```bash
python optimize.py -d dihedral_angles.h5 --rc-range 0.1 1.0 0.01 --opt-rc -t ca
```

3 . Run SPE algorithm
```bash
python spe.py -d dihedral_angles.h5 -c 10000 -t ca -r 0.27
```

## License
MIT