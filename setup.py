#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Jérôme Eberhardt 2016-2018
# Author: Jérôme Eberhardt <qksonoe@gmail.com>
#
# License: MIT

from setuptools import setup, find_packages

setup(name='unrolr',
      version=0.3,
      description='Dimensionality reduction method for MD trajectories',
      author='Jérôme Eberhardt',
      author_email='qksoneo@gmail.com',
      url='https//github.com/jeeberhardt/unrolr',
      packages=find_packages(),
      package_data={'': ['LICENSE',
                         'README.md',
                         'requirement.txt']
                   },
      install_requires=['setuptools', 'h5py', 'matplotlib',
                        'numpy', 'pandas', 'pyopencl',
                        'MDAnalysis>=0.15'],
      include_package_data=True,
      license='MIT',
      keywords=['bioinformatics', 'molecular structures',
               'molecular dynamics', 'OpenCL',
               'dimensionality reduction', 'stochastic proximity embedding',
               'dihedral angles'],
      classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Topic :: Scientific/Engineering'
      ]
      )
