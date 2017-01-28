# Unrolr
Conformational clustering of MD trajectories using (pivot-based) Stochastic Proximity Embedding with dihedral angles

## Tutorial

### Extraction of dihedral angles

The first step will be the extraction of all the pseudo C-alpha dihedral angles (by default) using the topology file (in psf format) and 200 ns aMD trajectory (with only 10000 frames) of the villin headpiece (in dcd format).

```bash
python extract_dihedral_angles.py -p villin.psf -d villin.dcd
```

As output, you will have a HDF5 file, named ```dihedral_angles.h5```, containing all the pseudo C-alpha dihedral angle (32 in total) from 10.000 frames of the villin headpiece. This HDF5 file can be opened and visualised easily using [HDFView](https://support.hdfgroup.org/products/java/hdfview/).

### Search optimal pSPE parameters

Now the next big step will be the determination of the optimal neighborhood radius rc and optionally the optimal number of cycles needed to achieve a good convergence, generally between 10.000 and 50.000 cycles. However, concerning the optimal neighborhood radius rc there is no general rule, because it depends exclusively of the studied system. The choice of its value will influence significantly the representation in the low dimensional space (n = 2): **if rc is too small, only local distances will be faithfully represented in low dimension, and the final representation will appear as cloud of disconnected clusters. On the contrary, if rc is too large, we loose the magical power of rc and the method will revert to a linear dimensionality reduction method like multidimensional scaling.**

#### How to choose the optimal neighbourhood radius rc cutoff?

As the optimal value of rc depends of the studied system, we can quickly test multiple value, and choose one that will minimizes the stress and maximizes the correlation between the distances in high dimension space and 2D dimension space. For this, we will systematically test different values of rc from 0.01 to 1.0 by increments of 0.01 (argument ```-r 0.1 1 0.01```). However, for the sake of effeciency, we won't use all the conformations (10.000 in our case), but just a reduced set (5.000 only using argument ```-i 2```). For this reduced set, 5 successive independent runs of pSPE are performed (argument ```--run 5```), with 5.000 steps of optimization cycles (fixed).

```bash
python search_parameters.py -d dihedral_angles.h5 -r 0.1 1 0.01 -i 2 --run 5
```

As output, you will find a file, named ```rc_vs_stress-correlation_rc_0.1_1.0_0.01_i_2.csv```, containing all the results, the stress and the correlation in function of the neighbourhood radius rc for each pSPE run, and the plot corresponding, named ```rc_vs_stress-correlation_rc_0.1_1.0_0.01_i_2.png```.

<div>
<img src="outputs/rc_vs_stress-correlation_rc_0.1_1.0_0.01_i_2.png" alt="rc_vs_stress_correlation">
</div>

The correlation between the actual and the projected distances increases as the neighbourhood radius rc is increased and converges to a plateau value of 0.80 (80%) for values of rc larger than 0.23. Further increase in rc does not improve the correlation but adversely affects the stress.

#### What is the minimal number of optimization cycles? (optional)

Generally, from my personal experience, the minimal number of optimization cycles needed is similar and independent of the nature and the size of the studied system, between 10.000 and 50.000 cycles. But still, we can test the influence of the number of pSPE optimization cycles on the correlation and the stress, while keeping the value of rc fixed at 0.23.

```bash
python search_parameters.py -d dihedral_angles.h5 -r 0.23 -i 2 --run 5
```

As output, you will find this time a file (named ```cycle_vs_stress-correlation_rc_0.23_i_2.csv```) containing all the results, the stress and the correlation in function of the number of optimization cycles for each pSPE run, and the plot corresponding (named ```cycle_vs_stress-correlation_rc_0.23_i_2.png```).

<div>
<img src="outputs/cycle_vs_stress-correlation_rc_0.23_i_2.png" alt="rc_vs_stress_correlation">
</div>

It can be seen that a minimum number of 10.000 cycles of optimization, at least, is needed to obtain converged values of the correlation and stress. Additional data (not shown here) shows that the size of the data set does not affect the convergence rate.

### Fire off Unrolr!

As the final step, after determining the optimal neighbourhood radius rc cutoff, equal to 0.23 in this case, and the minimal number of optimization cycles, at least 10.000 cycles, the pSPE method can now be applied to the complete data set.

```bash
python unrolr.py -d dihedral_angles.h5 -r 0.23 -c 50000
```

The final pSPE optimization process takes approximately 13 seconds for 10.000 conformations with 32 pseudo C-alpha dihedral angles and 50.000 cycles on a single (and now old) AMD Radeon HD 7950 GPU. As output, you will find the final optimized configuration, named ```configuration.txt```. Using the tool [visualize](https://github.com/jeeberhardt/visualize), you can now explore easily the conformational space sampled during the MD simulation.
