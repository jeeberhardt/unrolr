# Stochastic Proximity Embedding
Conformational clustering of MD trajectories using (pivot-based) Stochastic Proximity Embedding

## Tutorial

### Extraction of dihedral angles

The first step will be the extraction of all the pseudo C-alpha dihedral angles (by default) using the topology file (in psf format) and 200 ns aMD trajectory (with only 10000 frames) of the villin headpiece (in dcd format).

```bash
python extract_dihedral_angles.py -p villin.psf -d villin.dcd
```

As output, you will have a HDF5 file, named ```dihedral_angles.h5``` containing all the pseudo C-alpha dihedral angle (32 in total) from 10.000 frames of the villin headpiece. This HDF5 file can be opened and visualised easily using [HDFView](https://support.hdfgroup.org/products/java/hdfview/).

### Search optimal pSPE parameters

Now the next big step will be the determination of the optimal neighborhood radius rc and optionally the optimal number of cycles needed to achieve a good convergence, generally equal to 10.000 or 50.000 cycles. However, concerning the optimal neighborhood radius rc there is no general rule, because it depends exclusively of the studied system. The choice of its value will influence significantly the representation in the low dimensional space (n = 2): **if rc is too small, only local distances will be faithfully represented in low dimension, and the final representation will appear as cloud of disconnected clusters. On the contrary, if rc is too large, we loose the magical power of rc and the method will revert to a linear dimensionality reduction method.**

#### Choose the optimal neighborhood radius rc cutoff

```bash
python search_parameters.py -d dihedral_angles.h5 -r 0.1 1.0 0.01 -i 2
```

<div>
<img src="outputs/rc_vs_stress-correlation_rc_0.1_1.0_0.01_i_2.png" alt="rc_vs_stress_correlation">
</div>


#### Choose the optimal number of cycles

```bash
python search_parameters.py -d dihedral_angles.h5 -r 0.23 -i 2
```

<div>
<img src="outputs/cycle_vs_stress-correlation_rc_0.23_i_2.png" alt="rc_vs_stress_correlation">
</div>

### Fire off pSPE!

```bash
python spe.py -d dihedral_angles.h5 -r 0.23 -c 50000
```
