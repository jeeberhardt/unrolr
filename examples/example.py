#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unrolr import Unrolr
from unrolr.feature_extraction import Dihedral
from unrolr.sampling import neighborhood_radius_sampler, optimization_cycle_sampler
from unrolr.plotting import plot_sampling
from unrolr.utils import save_dataset


top_file = 'inputs/villin.psf'
trj_file = 'inputs/villin.dcd'

print("Extract all dihedral angles...")
d = Dihedral(top_file, trj_file, selection='all', dihedral_type='calpha').run(start=0, stop=None, step=1)
X = d.result
save_dataset('outputs/dihedral_angles.h5', "dihedral_angles", X)

print("Find optimal r_neighbor value...")
df = neighborhood_radius_sampler(X[::2,], r_neighbors=[0.1, 0.5, 0.01])
plot_sampling('outputs/r_neighbor_vs_stress-correlation.png', df, of='r_neighbor', show=False)
df.to_csv('outputs/r_neighbor_vs_stress-correlation.csv', index=False)

print("Find optimal n_iter value...")
df = optimization_cycle_sampler(X[::2,], n_iters=(10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000), r_neighbor=0.27)
plot_sampling('outputs/n_iter_vs_stress-correlation.png', df, of='n_iter', show=False)
df.to_csv('outputs/n_iter_vs_stress-correlation.csv', index=False)

print("Unrolr fitting...")
U = Unrolr(r_neighbor=0.27, n_iter=50000, verbose=1)
U.fit(X)
U.save(fname='outputs/embedding.csv')

print U.stress, U.correlation
