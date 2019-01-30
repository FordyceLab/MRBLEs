#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setuptools setup for MRBLEs package."""

from os import path
# To use a consistent encoding
from codecs import open as c_open
# Always prefer setuptools over distutils
from setuptools import setup, find_packages

ABS_PATH = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with c_open(path.join(ABS_PATH, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='mrbles',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.13.3',

    description='MRBLEs decoding and analysis package',
    long_description=LONG_DESCRIPTION,

    # The project's main homepage.
    url='https://github.com/FordyceLab/MRBLEs',

    # Author details
    author='Björn Harink',
    author_email='bjorn@harink.info',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # Python versions supported
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, <4',

    # What does your project relate to?
    keywords='mrbles optical encoding suspension arrays',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(include=['mrbles'],
                           exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['future',
                      'six',
                      'numpy>=1.14.0',
                      'scipy',
                      'pandas>=0.23.1',
                      'xarray',
                      'opencv-python>=3.2.0',
                      'scikit-learn',
                      'scikit-image',
                      'photutils>=0.4.1',
                      'packaging',
                      'matplotlib',
                      'plotly',
                      'lmfit',
                      'xlrd',
                      'packaging',
                      'notebook>=5.7.1',
                      'jupyter',
                      'seaborn>=0.9.0',
                      'openpyxl',
                      'xlrd'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },

    # Data files are specified in MANIFEST.in file.
    include_package_data=False,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('test_images',
    #              ['data/peptide_biotin_streptavidin_01_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_02_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_03_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_04_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_05_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_06_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_07_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_08_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_09_MMStack_Pos0.ome.tif',
    #               'data/peptide_biotin_streptavidin_10_MMStack_Pos0.ome.tif'])
    #            ],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
