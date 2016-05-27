#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import h5py
import argparse
import numpy as np
import pyopencl as cl

"""
export PYOPENCL_NO_CACHE=1
export PYOPENCL_CTX='0:1'
"""

class SPE():

    def __init__(self, cycles, rc, ndim, random_seed=0):

        # Check PYOPENCL_CTX environnment variable
        if not self.check_environnment_variable('PYOPENCL_CTX'):
            print('Error: The environnment variable PYOPENCL_CTX is not defined !')
            print('Tip: python -c \'import pyopencl as cl; cl.create_some_context()\'')
            sys.exit(1)

        # Set numpy random state
        self.random_seed = self.set_random_state(random_seed)

        # Set all the SPE variables
        self.cycles = cycles
        self.rc = rc
        self.ndim = ndim
        self.configuration = None
        self.stress = None
        self.correlation = None

    def check_environnment_variable(self, variable):
        """
        Check if an environnment variable exist or not
        """
        if os.environ.get(variable):
            return True
        else:
            return False

    def set_random_state(self, seed=0):
        # Set Random state (seed)
        if seed == 0:
            seed = np.random.randint(low=1, high=999999, size=1)[0]

        np.random.seed(seed = seed)

        return seed

    def read_data_from_hdf5(self, h5filename, dataname):
        """
        Read data from HDF5 file with the name dataname
        """
        with h5py.File(h5filename, 'r') as r:
            return r[dataname][:]

    def read_dihedral_angles_from_hdf5(self, h5filename, datanames, start, stop, interval):
        """
        Read dihedral angles from HDF5 with a certain interval, start and stop
        """
        data = None

        for dataname in datanames:
            if data is not None:
                data = np.concatenate((data, self.read_data_from_hdf5(h5filename, dataname)), axis=1)
            else:
                data = self.read_data_from_hdf5(h5filename, dataname)

        try:
            if stop == -1:
                return np.ascontiguousarray(data[start::interval,:], dtype=np.float32)
            else:
                return np.ascontiguousarray(data[start:stop:interval,:], dtype=np.float32)
        except:
            print("Error with the data selection")
            sys.exit(1)

    def store_data_to_hdf5(self, h5filename, data, dataname):
        """
        Store data in a HDF5 file with the name dataname
        """
        with h5py.File(h5filename, 'a') as w:
            w[dataname] = data
            w.flush()

    def get_total_number_of_frames(self, h5filename, datanames):
        with h5py.File(h5filename) as f:
            return f[datanames[0]].shape[0]

    def backup_directory(self, dir_name):
        """
        Backup the old directory
        """
        if os.path.isdir(dir_name):
            count = 1
            exist = True
            
            while exist:
                if os.path.isdir(dir_name + '#%d' % (count)):
                    count += 1
                else:
                    os.rename(dir_name, dir_name + '#%d' % (count))
                    exist = False
        else:
            pass

    def create_new_directory(self, dir_name):
        """
        Create a new directory and backup the old one if necessary
        """
        if os.path.exists(dir_name):
            self.backup_directory(dir_name)

        os.makedirs(dir_name)

    def fit(self, h5filename, dihedral_type, start=0, stop=-1, interval=-1, dir_output='.', save_frequency=10):
        """
        Create a directory and run the SPE method
        """
        if not isinstance(dihedral_type, (list, tuple)):
            dihedral_type = dihedral_type.split()

        # Get all dihedral angles
        dihedrals = self.read_dihedral_angles_from_hdf5(h5filename, dihedral_type, start, stop, interval)

        # Create directory and backup old directory
        dir_name = 'spe_%s_%s_c_%s_rc_%s_d_%s' % (dihedrals.shape[0], '_'.join(dihedral_type), 
                                                  self.cycles, self.rc, self.ndim)
        self.create_new_directory('%s/%s' % (dir_output, dir_name))

        # Fire off SPE calculation !!
        traj_name = '%s/%s/trajectory.h5' % (dir_output, dir_name)
        self.spe(dihedrals, self.cycles, self.rc, self.ndim, traj_name, save_frequency)

        # Evaluation embedding
        self.evaluate_embedding(dihedrals, self.configuration, self.rc)

        # We add the frame idx to the configuration
        if stop == -1:
            stop = self.get_total_number_of_frames(h5filename, dihedral_type)

        frames = np.arange(start, stop, interval)
        data = np.column_stack((frames, self.configuration.T))

        # Save final configuration to txt file
        txt_name = '%s/%s/configuration.txt' % (dir_output, dir_name)
        header = 'seed %s cycle %s rc %s stress %s corr %s'
        fmt = '%010d' + (self.ndim * '%10.5f')
        np.savetxt(txt_name, data, fmt=fmt, header=header % (self.random_seed, 
                   self.cycles, self.rc, self.stress, self.correlation))

    def spe(self, dihedrals, cycles, rc, ndim, output, frequency):
        """
        The SPE method itself !
        """
        learning_rate = 1.0
        #dc = 1.0
        alpha = float(learning_rate - 0.01) / float(cycles)

        # Create context and queue
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        # Compile kernel
        program = cl.Program(ctx, """
        __kernel void dihedral_distances(__global const float* a, __global float* r, int x, int size)
            {
                int i = get_global_id(0);

                r[i] = 0.0;
                
                for(int g=0; g<size; g++)
                {
                    r[i] += cos(a[x*size+g] - a[i*size+g]);
                }
         
                r[i] = sqrt((1.0/size) * 0.5 * (size - r[i]));
                //r[i] = sqrt(0.5 * (size - r[i]));
            }

        __kernel void euclidean_distances(__global const float* a, __global float* r, int x, int size, int ndim)
            {
                int i = get_global_id(0);

                r[i] = 0.0;

                for(int g=0; g<ndim; g++)
                {
                    r[i] += (a[g*size+i] - a[g*size+x]) * (a[g*size+i] - a[g*size+x]);
                }

                r[i] = sqrt(r[i]);
            }

        __kernel void spe(__global float* rij, __global float* dij, __global float* d, 
                          int x, int size, float rc, float learning_rate)
            {   
                const float eps = 1e-10;
                int i = get_global_id(0);
                int j = get_global_id(1);

                int index = i * size + j;
                int pindex = i * size + x;
                
                if ((rij[j] <= rc) || (rij[j] > rc && dij[j] < rij[j]) && (index != pindex))
                {   
                    d[index] = d[index] + (learning_rate * ((rij[j]-dij[j])/(dij[j]+eps)) * (d[index]-d[pindex]));
                } 
            }

        __kernel void tspe(__global float* rij, __global float* dij, __global float* d, 
                          int x, int size, float dc, float learning_rate)
            {   
                const float eps = 1e-10;
                int i = get_global_id(0);
                int j = get_global_id(1);

                int index = i * size + j;
                int pindex = i * size + x;
                
                if ((dij[j] <= dc) || (dij[j] > dc && dij[j] < rij[j]) && (index != pindex))
                {   
                    d[index] = d[index] + (learning_rate * ((rij[j]-dij[j])/(dij[j]+eps)) * (d[index]-d[pindex]));
                } 
            }
        """).build()

        # Send dihedral angles to CPU/GPU
        dihe_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dihedrals)

        # Allocate space on CPU/GPU to store rij and dij
        tmp = np.zeros((dihedrals.shape[0],), dtype=np.float32)
        rij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, tmp.nbytes)
        dij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, tmp.nbytes)

        # Generate initial (random)
        d = np.float32(np.random.rand(ndim, dihedrals.shape[0]))
        # Send initial (random) configuration to the CPU/GPU
        d_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=d)

        # If frequency superior at 0, we open a HDF5
        if frequency > 0:
            # Open HDF5 file
            f = h5py.File(output)
            # Store initial (random) configuration to HDF5
            store_data_to_hdf5(output, d.T, 'trajectory/frame_%010d' % 0)

        freq_progression = cycles / 100.

        for i in xrange(0, cycles + 1):

            if i % freq_progression == 0:
                percentage = float(i) / float(cycles) * 100.
                sys.stdout.write('\rSPE Optimization         : %8.3f %%' % percentage)
                sys.stdout.flush()

            # Choose random configuration (pivot)
            x = np.int32(np.random.randint(dihedrals.shape[0]))

            # Compute dihedral distances
            program.dihedral_distances(queue, (dihedrals.shape[0],), None, dihe_buf, rij_buf, 
                                       x, np.int32(dihedrals.shape[1])).wait()
            # Compute euclidean distances
            program.euclidean_distances(queue, (d.shape[1],), None, d_buf, dij_buf, x, 
                                        np.int32(d.shape[1]), np.int32(d.shape[0])).wait()
            # Stochastic Proximity Embbeding
            program.spe(queue, d.shape, None, rij_buf, dij_buf, d_buf, x, np.int32(d.shape[1]), 
                        np.float32(rc), np.float32(learning_rate)).wait()

            # Trustworthy Stochastic Proximity Embbeding
            #program.tspe(queue, d.shape, None, rij_buf, dij_buf, d_buf, x, np.int32(d.shape[1]), 
            #            np.float32(dc), np.float32(learning_rate)).wait()

            learning_rate -= alpha
            #dc -= alpha

            if (frequency > 0) and (i % frequency == 0) and (i > 0):
                # Get the current configuration
                cl.enqueue_copy(queue, d, d_buf)
                # Store current configuration to HDF5 file
                store_data_to_hdf5(output, d.T, 'trajectory/frame_%010d' % i)

        # At the end, we close the HDF5 file
        if frequency > 0:
            f.close()

        # Get the last configuration d
        cl.enqueue_copy(queue, d, d_buf)

        print()

        self.configuration = d

    def evaluate_embedding(self, dihedrals, configuration, rc, epsilon = 1e-5):
        """
        Evaluate the final configuration
        """
        # Creation du contexte et de la queue
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        # On compile le kernel
        program = cl.Program(ctx, """
        __kernel void dihedral_distances(__global const float* a, __global float* r, int x, int size)
            {
                int i = get_global_id(0);

                r[i] = 0.0;
                
                for(int g=0; g<size; g++)
                {
                    r[i] += cos(a[x*size+g] - a[i*size+g]);
                }
         
                r[i] = sqrt((1.0/size) * 0.5 * (size - r[i]));
                //r[i] = sqrt(0.5 * (size - r[i]));
            }

        __kernel void euclidean_distances(__global const float* a, __global float* r, int x, int size, int ndim)
            {
                int i = get_global_id(0);

                r[i] = 0.0;

                for(int g=0; g<ndim; g++)
                {
                    r[i] += (a[g*size+i] - a[g*size+x]) * (a[g*size+i] - a[g*size+x]);
                }

                r[i] = sqrt(r[i]);
            }

        __kernel void stress(__global float* rij, __global float* dij, __global float* sij, float rc)
            {   
                int i = get_global_id(0);

                sij[i] = 0.0;
                
                if ((rij[i] <= rc) || (dij[i] < rij[i]))
                {   
                    sij[i] = ((dij[i]-rij[i])*(dij[i]-rij[i]))/(rij[i]);
                }
            }
        """).build()

        # Send dihedral angles and configuration on CPU/GPU
        config_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=configuration)
        dihe_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dihedrals)

        # Allocate memory
        rij = np.zeros((dihedrals.shape[0],), dtype=np.float32)
        dij = np.zeros((dihedrals.shape[0],), dtype=np.float32)
        rij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, rij.nbytes)
        dij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, dij.nbytes)

        sij = np.zeros((dihedrals.shape[0],), dtype=np.float32)
        sij_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, sij.nbytes)

        tmp_correl = []
        tmp_sij_sum = 0.0
        tmp_dij_sum = 0.0
        
        old_stress = 999.
        old_correl = 999.

        while True:

            # Choose random conformation as pivot
            x = np.int32(np.random.randint(dihedrals.shape[0]))

            # Dihedral distances
            program.dihedral_distances(queue, (dihedrals.shape[0],), None, dihe_buf, rij_buf, 
                                       x, np.int32(dihedrals.shape[1])).wait()
            # Euclidean distances
            program.euclidean_distances(queue, (configuration.shape[1],), None, config_buf, 
                                        dij_buf, x, np.int32(configuration.shape[1]), 
                                        np.int32(configuration.shape[0])).wait()
            # Compute part of stress
            program.stress(queue, (dihedrals.shape[0],), None, rij_buf, dij_buf, sij_buf, 
                           np.float32(rc)).wait()

            # Get rij, dij and sij
            cl.enqueue_copy(queue, rij, rij_buf)
            cl.enqueue_copy(queue, dij, dij_buf)
            cl.enqueue_copy(queue, sij, sij_buf)

            # Compute current correlation
            tmp = (np.dot(rij.T, dij) / rij.shape[0]) - (np.mean(rij) * np.mean(dij))
            tmp_correl.append(tmp / (np.std(rij) * np.std(dij)))
            correl = np.mean(tmp_correl)

            # Compute current stress
            tmp_sij_sum += np.sum(sij[~np.isnan(sij)])
            tmp_dij_sum += np.sum(dij)
            stress = tmp_sij_sum / tmp_dij_sum
            
            # Test for convergence
            if (np.abs(old_stress - stress) < epsilon) and (np.abs(old_correl - correl) < epsilon):
                self.correlation = correl 
                self.stress = stress

                break
            else:
                old_stress = stress
                old_correl = correl

def parse_options():
    parser = argparse.ArgumentParser(description='SPE python script')
    parser.add_argument('-d', '--h5', dest='hdf5filename', required=True,
                        action='store', type=str,
                        help='HDF5 file with dihedral angles')
    parser.add_argument('-c', '--cycles', dest='cycles',
                        action='store', type=int, default=1000,
                        help='number of cycle')
    parser.add_argument('-t', '--dihedral', dest='dihedral_type',
                        action='store', type=str, nargs='+',
                        choices=['ca', 'phi', 'psi'],
                        default='ca', help='dihedral type')
    parser.add_argument('-r', '--rc', dest='rc',
                        action='store', type=float, default=1.,
                        help='neighborhood cutoff')
    parser.add_argument('-n', '--ndim', dest='ndim',
                        action='store', type=int, default=2,
                        help='number of dimension')
    parser.add_argument('--run', dest='runs',
                        action='store', type=int, default=1,
                        help='number of spe runs')
    parser.add_argument('--start', dest='start',
                        action='store', type=int, default=0,
                        help='used frames from this position')
    parser.add_argument('--stop', dest='stop',
                        action='store', type=int, default=-1,
                        help='used frames until this position')
    parser.add_argument('-i', '--interval', dest='interval',
                        action='store', type=int, default=1,
                        help='used frames at this interval')
    parser.add_argument('-o' '--output', dest='output',
                        action='store', type=str, default='.',
                        help='directory output')
    parser.add_argument('-f' '--frequency', dest='frequency',
                        action='store', type=int, default=0,
                        help='trajectory saving interval (0 if you don\'t want)')
    parser.add_argument('-s' '--seed', dest='random_seed',
                        action='store', type=int, default=0,
                        help='If you want to reproduce spe trajectory')

    return parser.parse_args()

def main():

    options = parse_options()

    hdf5filename = options.hdf5filename
    start = options.start
    stop = options.stop
    interval = options.interval
    ndim = options.ndim
    runs = options.runs
    cycles = options.cycles
    rc = options.rc
    output = options.output
    frequency = options.frequency
    random_seed = options.random_seed
    dihedral_type = options.dihedral_type

    for i in xrange(runs):

        spe = SPE(cycles, rc, ndim, random_seed)
        print("Random seed              : %8d" % spe.random_seed)

        spe.fit(hdf5filename, dihedral_type, start, stop, interval, output, frequency)

        print(spe.stress, spe.correlation)

if __name__ == '__main__':
    main()