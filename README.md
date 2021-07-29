[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/unrolr/badge/?version=latest)](https://unrolr.readthedocs.io/en/latest/?badge=latest)

# Unrolr
Conformational analysis of MD trajectories based on (pivot-based) Stochastic Proximity Embedding using dihedral distance as a metric (https://github.com/jeeberhardt/unrolr).

## Documentation

The installation instructions, documentation and tutorials can be found on [readthedocs.org](https://unrolr.readthedocs.io/en/latest/).

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
