# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRBLEs Analysis Package
=======================
This package provides the tools to: (1) find the MRBLEs in a monochrome
bright-field microscopy image; (2) decode the MRBLEs beads by spectral unmxing,
using reference spectra, and then spectrally decode the found beads using
Iterative Closest Point Matching and Gaussian Mixture Modeling; (3) extract,
statistical values of interest in additional fluorescence channels using the
morphology of the MRBLEs, their locations, and respective code; (4) analyze
affinity information based on titrations of MRBLEs assays.
"""

# [Future imports]
from __future__ import (absolute_import, division, print_function)
from future import standard_library

# [File header]     | Copy and edit for each file in this project!
# title             : MRBLEs Analysis Package
# description       : Bead Analysis
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen, Polly Fordyce
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
# Original author(s) of this Python project:
__author__ = "Bjorn Harink"
# People who contributed to this Python project:
__credits__ = ["Kurt Thorn",
               "Huy Nguyen",
               "Polly Fordyce"]
# Maintainer contact information
__maintainer__ = "Bjorn Harink"
__email__ = "bjorn@harink.info"
# Software information
__license__ = "MIT"
__version__ = '0.12.3'
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
#   abstract  = {MRBLEs analysis software is...},
#   publisher = {Elsevier},
#   year      = 2017
# }"""

# [Module Imports]
# import sys
from mrbles import core
from mrbles import data
from mrbles import pipeline
from mrbles.pipeline import *
from mrbles import report
from mrbles import kinetics
from mrbles import path

standard_library.install_aliases()
print(__copyright__, " - Version: ", __version__)
