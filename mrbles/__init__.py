# !/usr/bin/env python

"""
Bead Analysis Package
=====================

This package contains the necessary classes, methods and data structures to
perform the Bead Analysis for MRBLs.
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
__copyright__ = ("Copyright 2017 - "
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
__version__ = "0.7.0"
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
from . import core
from .core import *  # TODO Must change to specific before public
from . import data
from .data import *  # TODO Must change to specific before public
from . import inspect
from . import simp
from . import kinetics

# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    from future.standard_library import install_aliases
    from __builtin__ import *  # NOQA
    install_aliases()

print(__copyright__)
