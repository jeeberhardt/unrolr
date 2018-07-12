#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unrolr import Unrolr
from unrolr.feature_extraction import Dihedral
from unrolr.optimize import find_optimal_r_neighbor, find_optimal_n_iter
from unrolr.plotting import plot_optimization
from unrolr.utils import save_dataset


top_file = 'inputs/villin.psf'
trj_file = 'inputs/villin.dcd'

print "Extract all dihedral angles ..."
d = Dihedral(top_file, trj_file, selection='all', dihedral_type='calpha', start=0, stop=None, step=1).run()
X = d.result
save_dataset('dihedral_angles.h5', "dihedral_angles", X)

print "Find optimal r_neighbor value ..."
df = find_optimal_r_neighbor(X[::2,], r_parameters=[0.1, 0.5, 0.01])
plot_optimization('outputs/r_neighbor_vs_stress-correlation.png', df, of='r_neighbor')
df.to_csv('outputs/r_neighbor_vs_stress-correlation.csv', index=False)

print "Find optimal n_iter value ..."
df = find_optimal_n_iter(X[::2,], n_iters=(10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000), r_neighbor=0.27)
plot_optimization('outputs/n_iter_vs_stress-correlation.png', df, of='n_iter')
df.to_csv('outputs/n_iter_vs_stress-correlation.csv', index=False)

print "Unrolr fitting ..."
U = Unrolr(r_neighbor=0.27, n_iter=50000, verbose=1)
U.fit(X)
U.save(fname='outputs/embedding.csv')

print U.stress, U.correlation
