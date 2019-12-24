Installation
============
Conformational analysis of MD trajectories based on (pivot-based) Stochastic Proximity Embedding using dihedral distance as a metric. 

Prerequisites
-------------

You need, at a minimum (requirements.txt):

* Python 2.7 or python 3
* NumPy
* H5py
* Pandas
* Matplotlib
* PyOpenCL
* MDAnalysis (>=0.17)

Installation on UNIX (Debian/Ubuntu)
------------------------------------

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed (NumPy, H5py, Pandas, Matplotlib).

1 . First, you have to install OpenCL:

* MacOS: Good news, you don't have to install OpenCL, it works out-of-the-box. (Update: bad news, OpenCL is now depreciated in macOS 10.14. Thanks Apple.)
* AMD:  You have to install the `AMDGPU graphics stack <https://amdgpu-install.readthedocs.io/en/amd-18.30/index.html>`_.
* Nvidia: You have to install the `CUDA toolkit <https://developer.nvidia.com/cuda-downloads>`_.
* Intel: And of course it's working also on CPU just by installing this `runtime software package <https://software.intel.com/en-us/articles/opencl-drivers>`_. Alternatively, the CPU-based OpenCL driver can be also installed through the package ```pocl``` (http://portablecl.org/) with the conda package manager.

For any other informations, the official installation guide of PyOpenCL is available `here <https://documen.tician.de/pyopencl/misc.html>`_.

2 . As a final step, installation from PyPi server

.. code-block:: bash

	pip install unrolr

Or from the source

.. code-block:: bash

	# Get the package
	wget https://github.com/jeeberhardt/unrolr/archive/master.zip
	unzip unrolr-master.zip
	rm unrolr-master.zip
	cd unrolr-master

	# Install the package
	python setup.py install

And if somehow pip is having problem to install all the dependencies,

.. code-block:: bash

	conda config --append channels conda-forge
	conda install pyopencl mdanalysis

	# Try again
	python setup.py install

OpenCL context
--------------

Before running Unrolr, you need to define the OpenCL context. And it is a good way to see if everything is working correctly.

.. code-block:: bash

	python -c 'import pyopencl as cl; cl.create_some_context()'

Here in my example, I have the choice between 3 differents computing device (2 graphic cards and one CPU). 

.. code-block:: bash

	Choose platform:
	[0] <pyopencl.Platform 'AMD Accelerated Parallel Processing' at 0x7f97e96a8430>
	Choice [0]:0
	Choose device(s):
	[0] <pyopencl.Device 'Tahiti' on 'AMD Accelerated Parallel Processing' at 0x1e18a30>
	[1] <pyopencl.Device 'Tahiti' on 'AMD Accelerated Parallel Processing' at 0x254a110>
	[2] <pyopencl.Device 'Intel(R) Core(TM) i7-3820 CPU @ 3.60GHz' on 'AMD Accelerated Parallel Processing' at 0x21d0300>
	Choice, comma-separated [0]:1
	Set the environment variable PYOPENCL_CTX='0:1' to avoid being asked again.

Now you can set the environment variable.

.. code-block:: bash

	export PYOPENCL_CTX='0:1'

