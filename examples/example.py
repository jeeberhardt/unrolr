#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from unrolr import Unrolr
from unrolr.feature_extraction import Dihedral
from unrolr.plotting import plot_sampling
from unrolr.sampling import neighborhood_radius_sampler, optimization_cycle_sampler
from unrolr.utils import save_dataset


init = 'random' # or 'pca'
platform = 'CPU' # or 'CPU'
top_file = 'inputs/villin.psf'
trj_file = 'inputs/villin.dcd'


print('Extract all dihedral angles...')
d = Dihedral(top_file, trj_file, selection='backbone', dihedral_type='calpha').run()
X = d.result
save_dataset('outputs/dihedral_angles.h5', 'dihedral_angles', X)

print('Find optimal r_neighbor value...')
r_neighbors = np.linspace(0.1, 1.0, int((1 / 0.05) - 1))
df = neighborhood_radius_sampler(X[::2,], r_neighbors, init=init, platform=platform)
plot_sampling('outputs/r_neighbor_vs_stress-correlation.png', df, of='r_neighbor', show=False)
df.to_csv('outputs/r_neighbor_vs_stress-correlation.csv', index=False)

print('Find optimal n_iter value...')
n_iters = [500, 1000, 5000, 10000, 50000, 100000]
df = optimization_cycle_sampler(X[::2,], n_iters=n_iters, r_neighbor=0.27, init=init, platform=platform)
plot_sampling('outputs/n_iter_vs_stress-correlation.png', df, of='n_iter', show=False)
df.to_csv('outputs/n_iter_vs_stress-correlation.csv', index=False)

print('Unrolr fitting...')
U = Unrolr(r_neighbor=0.27, n_iter=50000, verbose=1)
U.fit_transform(X)
U.save(fname='outputs/embedding.csv')

print(U.stress, U.correlation)
