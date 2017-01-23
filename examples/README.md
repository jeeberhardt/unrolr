# Stochastic Proximity Embedding
Conformational clustering of MD trajectories using (pivot-based) Stochastic Proximity Embedding

## Tutorial

1. The first step will be the extraction of all the dihedral angles using the topology file (in psf format) and 200 ns aMD trajectory (with only 10000 frames) of the villin headpiece (in dcd format).


### Choose the optimal neighborhood rc cutoff
<div>
<img src="outputs/rc_vs_stress-correlation_rc_0.1_1.0_0.01_i_2.png" alt="rc_vs_stress_correlation">
</div>


### Choose the optimal number of cycles
<div>
<img src="outputs/cycle_vs_stress-correlation_rc_0.23_i_2.png" alt="rc_vs_stress_correlation">
</div>
