Unrolr: Analysis of MD simulations using Stochastic Proximity Embedding 
=========================================================================================

Abstract
--------

Molecular dynamics (MD) simulations are widely used to explore the conformational space of biological macromolecules. Advances in hardware, as well as in methods, make the generation of large and complex MD datasets much more common. Although different clustering and dimensionality reduction methods have been applied to MD simulations, there remains a need for improved strategies that handle nonlinear data and/or can be applied to very large datasets. We present an original implementation of the pivot-based version of the stochastic proximity embedding method aimed at large MD datasets using the dihedral distance as a metric. The advantages of the algorithm in terms of data storage and computational efficiency are presented, as well as the implementation realized.

**Example**

.. code-block:: python

	from __future__ import print_function

	from unrolr import Unrolr
	from unrolr.feature_extraction import Dihedral
	from unrolr.utils import save_dataset


	top_file = 'examples/inputs/villin.psf'
	trj_file = 'examples/inputs/villin.dcd'

	# Extract all calpha dihedral angles from trajectory and store them into a HDF5 file (start/stop/step are optionals)
	d = Dihedral(top_file, trj_file, selection='all', dihedral_type='calpha', start=0, stop=None, step=1).run()
	X = d.result
	save_dataset('dihedral_angles.h5', "dihedral_angles", X)

	# Fit X using Unrolr (pSPE + dihedral distance) and save the embedding into a csv file
	U = Unrolr(r_neighbor=0.27, n_iter=50000, verbose=1)
	U.fit_transform(X)
	U.save(fname='embedding.csv')

	print('%4.2f %4.2f' % (U.stress, U.correlation))

**Todo list**

- [ ] Compare SPE performance with UMAP
- [x] Compatibility with python 3
- [x] Compatibility with the latest version of MDAnalysis (==0.17)
- [ ] Unit tests
- [x] Accessible directly from pip
- [ ] Improve OpenCL performance (global/local memory)

**Citation**

Eberhardt, J., Stote, R. H., & Dejaegere, A. (2018). Unrolr: Structural analysis of protein conformations using stochastic proximity embedding. Journal of Computational Chemistry, 39(30), 2551-2557. https://doi.org/10.1002/jcc.25599

**License**

MIT

.. _dev-docs:

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User guide / tutorial

   installation
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   unrolr
