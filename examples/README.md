# Stochastic Proximity Embedding
Conformational clustering of MD trajectories using (pivot-based) Stochastic Proximity Embedding

## Tutorial

1. The first step will be the extraction of all the dihedral angles using the topology file (in psf format) and 200 ns aMD trajectory (with only 10000 frames) of the villin headpiece (in dcd format).

```bash
python extract_dihedral_angles.py -p villin.psf -d villin.dcd
```

As output, you will have an HDF5 file containing all the pseudo C-alpha dihedral angle (32 in total) from 10.000 frames.

2. Now the next big step will be the determination of the optimal neighborhood radius rc. However, there is no general rule

### Choose the optimal neighborhood radius rc cutoff
<div>
<img src="outputs/rc_vs_stress-correlation_rc_0.1_1.0_0.01_i_2.png" alt="rc_vs_stress_correlation">
</div>


### Choose the optimal number of cycles
<div>
<img src="outputs/cycle_vs_stress-correlation_rc_0.23_i_2.png" alt="rc_vs_stress_correlation">
</div>

3. And the final step, 
