# !/usr/bin/env python

# [Future imports]
# Function compatibility between Python 2.x and 3.x
from __future__ import print_function, division
from future.standard_library import install_aliases
install_aliases()
import sys
if sys.version_info < (3,0): from __builtin__ import *

# [Module Imports]
from bead_analysis.core import *
from bead_analysis.data import *
import bead_analysis.inspect as inspect
import bead_analysis.simp as simp
import bead_analysis.kinetics as kin

# [Main header with project metadata] | Only in the main file!
# Copyright and credits
__copyright__   = ("Copyright 2017 - "
                   "The Encoded Beads Project - "
                   "ThornLab@UCSF and "
                   "FordyceLab@Stanford")
# Original author(s) of this Python project, like: ("...", 
__author__      = ("Bjorn Harink")  #               "name")
# People who contributed to this Python project, like: ["...",
__credits__     = ["Kurt Thorn",  #                     "name"]
                   "Huy Nguyen"]
# Maintainer contact information
__maintainer__  = "Bjorn Harink" 
__email__       = "bjorn@harink.info" 
# Software information
__license__     = "MIT" 
__version__     = "v0.5"
__status__      = "Prototype"

print(__copyright__)

"""Bead Analysis Package

This package contains the necessary classes, methods and data structures to perform the Bead Analysis for MRBLs.
"""