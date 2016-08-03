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

## Installation on UNIX

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed (NumPy, H5py, Pandas, Matplotlib).

1 . First, you have to install OpenCL. Good news for MacOS users, you don't have to install OpenCL, it works out-of-the-box, so you can skip this part and just install pyopencl and MDAanalysis. For others, from all the tutorials you can find on the internet, this one it is still the more succinct one that I found: [OpenCL installation](https://ethereum.gitbooks.io/frontier-guide/content/gpu.html).

2 . For the rest, you just have to do this,
```bash
pip install pyopencl mdanalysis
```

## OpenCL context

Before running the SPE algorithm, you need to define the OpenCL context. And it is a good way to see if everything is working correctly.

```bash
python -c 'import pyopencl as cl; cl.create_some_context()'
```

Here in my example, I have the choice between 3 differents computing device (2 graphic cards and one CPU). 

```bash
Choose platform:
[0] <pyopencl.Platform 'AMD Accelerated Parallel Processing' at 0x7f97e96a8430>
Choice [0]:0
Choose device(s):
[0] <pyopencl.Device 'Tahiti' on 'AMD Accelerated Parallel Processing' at 0x1e18a30>
[1] <pyopencl.Device 'Tahiti' on 'AMD Accelerated Parallel Processing' at 0x254a110>
[2] <pyopencl.Device 'Intel(R) Core(TM) i7-3820 CPU @ 3.60GHz' on 'AMD Accelerated Parallel Processing' at 0x21d0300>
Choice, comma-separated [0]:1
Set the environment variable PYOPENCL_CTX='0:1' to avoid being asked again.
```

Now you can set the environment variable.

```bash
export PYOPENCL_CTX='0:1'
```

## How-To

1 . First you need to extract all the C-alpha (or the Phi/Psi) dihedral angles from your trajectory
```bash
python extract_dihedral_angles.py -p topology.psf -d traj.dcd
```
**Command line options**
* -p/--top: topology file (pdb, psf)
* -d/--dcd: single trajectory or list of trajectories (dcd, xtc)
* -s/--selection: selection command (ex: resid 1:10)(default: all)(documentation: https://goo.gl/4t1mGb)
* -t/--dihedral: dihedral types you want to extract (choices: ca, phi, psi)(default: ca)
* -o/--ouput: output name (default: dihedral_angles.h5)

**Outputs**
* HDF5 file with the selected dihedral angles

2 . Find the optimal neighborhood RC value (or find the optimal number of cycles) using only a small subset of conformations (only 5000 or 10000), that cover the whole trajectory.
```bash
python optimize.py -d dihedral_angles.h5 --rc 0.1 1.0 0.01 -i 100 # if you want to find the optimal RC value
python optimize.py -d dihedral_angles.h5 --rc 0.27 -i 100 # if you want to find the optimal cycle value
```

**Command line options**
* -d/--h5: HDF5 file with all the dihedral angles
* --rc: neighborhood RC value (opt cycle) or RC range (0.1 1 0.1)(opt rc)
* --run: number of SPE runs (default: 5)
* -n/--ndim: number of dimension (default: 2)
* -t/--dihedral: dihedral types you want to use (choices: ca, phi, psi)(default: ca)
* --start: starting frame (default: 0)
* --stop: last frame (default: -1)
* -i/--interval: interval (default: 1)
* -o/--output: output directory (default: .)

**Outputs**
* configuration file of each spe run (optimized coordinates)
* stress/correlation in function of rc values (plot and raw data)
* stress/correlation in function of cycle values (plot and raw data)

3 . And finally, after finding the optimal rc and cycle values you can run the SPE algorithm at its full potential with all the conformations.
```bash
python spe.py -d dihedral_angles.h5 -c 10000 -r 0.27
```

**Command line options**
* -d/--h5: HDF5 file with all the dihedral angles
* -c/--cycles: number of optimization cycles
* -r/--rc: neighborhood rc value
* -t/--dihedral: dihedral types you want to use (choices: ca, phi, psi)(default: ca)
* -n/--ndim: number of dimension (default: 2)
* --run: number of SPE runs (default: 1)
* --start: starting frame (default: 0)
* --stop: last frame (default: -1)
* -i/--interval: interval (default: 1)
* -o/--output: output directory (default: .)
* -f/--frequency: SPE trajectory saving interval (0 if you don't want)(default: 0)
* -s/--seed: random seed (if you want to reproduce SPE results) (default: None)

**Outputs**
* configuration file (optimized coordinates)
* HDF5 file with spe trajectory (if selected)

## Citation
Soon ...

## License
MIT
