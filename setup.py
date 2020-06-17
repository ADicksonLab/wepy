#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import setup, find_packages

import itertools as it

import versioneer

# setuptools only specifies abstract requirements. For the concrete
# requirements i.e. index or repo URL see requirements.txt
base_requirements = [
    'numpy',
    'h5py',
    'networkx==2.3',
    'pandas',
    'dill',
    'click',
    'scipy',
    'geomm',
    'matplotlib',
    'tabulate',
    'jinja2',
    'pint',
    'multiprocessing_logging',
    'eliot',
]

# extras requirements list
md_requirements = [
    'mdtraj',
    'openmm_systems==0.0.0',
]
distributed_requirements = ['dask[bag]']

prometheus_requirements = [
    'prometheus_client',
    'pympler',
]

# # combination of all the extras requirements
all_requirements = list(it.chain.from_iterable([
    base_requirements,
    md_requirements,
    distributed_requirements,
    prometheus_requirements,
]))

setup(
    name='wepy',
    version=versioneer.get_version(),
    author="Samuel D. Lotz",
    author_email="samuel.lotz@salotz.info",
    description="Weighted Ensemble Framework",
    #long_description=open('README.org').read(),
    license="MIT",
    url="https://github.com/ADicksonLab/wepy",
    classifiers=[
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3'
    ],
    # building/dev
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'tox'],

    cmdclass=versioneer.get_cmdclass(),

    # package
    packages=find_packages(where='src'),

    package_dir={'' : 'src'},

    # if this is true then the package_data won't be included in the
    # dist. Use MANIFEST.in for this
    include_package_data=True,

    # pymodules is for single file standalone modules not part of the
    # package
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    entry_points={
        'console_scripts' : [
            'wepy=wepy.cli:cli',
        ],
        'pytest11' : [
            'pytest-wepy=pytest_wepy',
        ],

    },

    install_requires=base_requirements,

    extras_require={
        'md' : md_requirements,
        'distributed' : distributed_requirements,
        'prometheus' : prometheus_requirements,
        'all' : all_requirements,
    }
)
