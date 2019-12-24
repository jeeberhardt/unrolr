__kernel void dihedral_distance(__global const float* a, __global float* r, int x, int size)
{
    int i = get_global_id(0);
    float tmp;

    r[i] = 0.0;

    for(int g=0; g<size; g++)
    {
        r[i] += cos(a[x*size+g] - a[i*size+g]);
    }

    tmp = (1.0 / size) * 0.5 * (size - r[i]);
    r[i] = sqrt(tmp);
}

__kernel void intramolecular_distance(__global const float* a, __global float* r, int x, int size)
{
    int i = get_global_id(0);

    r[i] = 0.0;

    for(int g=0; g<size; g++)
    {
        r[i] += pow(a[x*size+g] - a[i*size+g], 2);
    }

    r[i] = sqrt(r[i] / size);
}

__kernel void euclidean_distance(__global const float* a, __global float* r, int x, int size, int ndim)
{
    int i = get_global_id(0);

    r[i] = 0.0;

    for(int g=0; g<ndim; g++)
    {
        r[i] += (a[g*size+i] - a[g*size+x]) * (a[g*size+i] - a[g*size+x]);
    }

    r[i] = sqrt(r[i]);
}

__kernel void spe(__global float* rij, __global float* dij, __global float* d, int x, int size, float rc, float learning_rate)
{
    const float eps = 1e-10;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i * size + j;
    int pindex = i * size + x;

    if (((rij[j] <= rc) || (rij[j] > rc && dij[j] < rij[j])) && (index != pindex))
    {
        d[index] = d[index] + (learning_rate * ((rij[j]-dij[j])/(dij[j]+eps)) * (d[index]-d[pindex]));
    }
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
