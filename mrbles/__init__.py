# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRBLEs Analysis Package
=======================
This package provides the tools to: (1) find the MRBLEs in a monochrome
brightfield microscopy image; (2) decode the MRBLEs beads by spectral unmxing,
using reference spectra, and then spectrally decode the found beads using
Iterative Closest Point Matching and Gaussian Mixture Modeling; (3) extract,
statistical values of interest in additional fluorescence channels using the
morphology of the MRBLEs, their locations, and respective code; (4) analyze
affinity information based on titrations of MRBLEs assays.
"""

# [Future imports]
from __future__ import print_function, division, absolute_import

# [File header]     | Copy and edit for each file in this project!
# title             : MRBLEs Analysis Package
# description       : Bead Analysis
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20160308

# [Main header with project metadata] | Only in the main file!
# Name of package
__name__ = "mrbles"
__package__ = "mrbles"
# Copyright and credits
__copyright__ = ("Copyright 2015-2018 - "
                 "The Encoded Beads Project - "
                 "ThornLab@UCSF and "
                 "FordyceLab@Stanford")
# Original author(s) of this Python project, like: ("...",
__author__ = ("Bjorn Harink")  #                    "name")
# People who contributed to this Python project, like: ["...",
__credits__ = ["Kurt Thorn",  #                         "name"]
               "Huy Nguyen"]
# Maintainer contact information
__maintainer__ = "Bjorn Harink"
__email__ = "bjorn@harink.info"
# Software information
__license__ = "MIT"
__version__ = '0.8.5'
__status__ = "Development"
# Package settings
__all__ = ['core', 'data']

# __bibtex__ = r"""@Article{Harink:2017,
#   Author    = {Harink, Bjorn},
#   Title     = {MRBLEs Analysis},
#   Journal   = {SoftwareX},
#   Volume    = {x},
#   Number    = {x},
#   Pages     = {xx--xx},
#   abstract  = {MRBLEs analysis sofwtare is...},
#   publisher = {Elsevier},
#   year      = 2017
# }"""

# [Module Imports]
import sys
from mrbles import core
from mrbles import data
from mrbles import inspect
from mrbles import pipeline
from mrbles.pipeline import *
from mrbles import kinetics

# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    from future.standard_library import install_aliases
    from __builtin__ import *  # NOQA
    install_aliases()

print(__copyright__)
