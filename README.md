[![Documentation Status](https://readthedocs.org/projects/unrolr/badge/?version=latest)](https://unrolr.readthedocs.io/en/latest/?badge=latest)

# Unrolr
Conformational analysis of MD trajectories based on (pivot-based) Stochastic Proximity Embedding using dihedral distance as a metric (https://github.com/jeeberhardt/unrolr).

## Prerequisites

You need, at a minimum (requirements.txt):

* Python
* NumPy
* H5py
* Pandas
* Matplotlib
* PyOpenCL
* MDAnalysis

## Installation on UNIX (Debian/Ubuntu)

1 . First, you have to install OpenCL:
* MacOS: Good news, you don't have to install OpenCL, it works out-of-the-box. (Update: bad news, OpenCL is now depreciated in macOS 10.14. Thanks Apple.)
* AMD:  You have to install the [AMDGPU graphics stack](https://amdgpu-install.readthedocs.io/en/amd-18.30/index.html).
* Nvidia: You have to install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
* Intel: And of course it's working also on CPU just by installing this [runtime software package](https://software.intel.com/en-us/articles/opencl-drivers). Alternatively, the CPU-based OpenCL driver can be also installed through the package ```pocl``` (http://portablecl.org/) using Anaconda.

For any other informations, the official installation guide of PyOpenCL is available [here](https://documen.tician.de/pyopencl/misc.html).

2 . I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:

```bash
$ conda create -n unrolr python=3
$ conda activate unrolr
$ conda install -c conda-forge mkl numpy scipy pandas matplotlib h5py MDAnalysis pyopencl ocl-icd-system
```

3 . Install unrolr
```bash
$ pip install unrolr
```
... or from the source directly

```bash
$ git clone https://github.com/jeeberhardt/unrolr
$ cd unrolr
$ python setup.py build install
```

## OpenCL context

Before running Unrolr, you need to define the OpenCL context. And it is a good way to see if everything is working correctly.

```bash
$ python -c 'import pyopencl as cl; cl.create_some_context()'
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
$ export PYOPENCL_CTX='0:1'
```

## Example

```python
from unrolr import Unrolr
from unrolr.feature_extraction import Dihedral
from unrolr.utils import save_dataset


top_file = 'examples/inputs/villin.psf'
trj_file = 'examples/inputs/villin.dcd'

# Extract all calpha dihedral angles from trajectory and store them into a HDF5 file
d = Dihedral(top_file, trj_file, selection='all', dihedral_type='calpha').run()
X = d.result
save_dataset('dihedral_angles.h5', "dihedral_angles", X)

# Fit X using Unrolr (pSPE + dihedral distance) and save the embedding into a csv file
# The initial embedding is obtained using PCA (init = 'pca') with the OpenCL implementation
# to run SPE, a CPU implementation can be used as an alternative (platform='CPU')
U = Unrolr(r_neighbor=0.27, n_iter=50000, init='pca', platform='OpenCL', verbose=1)
U.fit_transform(X)
U.save(fname='embedding.csv')

print('%4.2f %4.2f' % (U.stress, U.correlation))
```

## Todo list
- [ ] Compare SPE performance with UMAP
- [x] Compatibility with python 3
- [x] Compatibility with the latest version of MDAnalysis (==0.17)
- [ ] Unit tests
- [x] Accessible directly from pip
- [ ] Improve OpenCL performance (global/local memory)

## Citation
Eberhardt, J., Stote, R. H., & Dejaegere, A. (2018). Unrolr: Structural analysis of protein conformations using stochastic proximity embedding. Journal of Computational Chemistry, 39(30), 2551-2557. https://doi.org/10.1002/jcc.25599

## License
MIT
