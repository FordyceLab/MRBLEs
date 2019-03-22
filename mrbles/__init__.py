# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""MRBLEs Analysis Package.

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
__copyright__ = ("Copyright 2015-2019 - "
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
__version__ = '1.0.0'
__status__ = "Development"
# Package settings
__all__ = ['core', 'data']


# Citation of scientific publication
__bibtex__ = r"""@article{10.1371/journal.pone.0203725,
    author = {Harink, BjÃ¶rn AND Nguyen, Huy AND Thorn, Kurt AND Fordyce, Polly},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {An open-source software analysis package for Microspheres with Ratiometric Barcode Lanthanide Encoding (MRBLEs)},
    year = {2019},
    month = {03},
    volume = {14},
    url = {https://doi.org/10.1371/journal.pone.0203725},
    pages = {1-20},
    abstract = {Multiplexed bioassays, in which multiple analytes of interest are probed in parallel within a single small volume, have greatly accelerated the pace of biological discovery. Bead-based multiplexed bioassays have many technical advantages, including near solution-phase kinetics, small sample volume requirements, many within-assay replicates to reduce measurement error, and, for some bead materials, the ability to synthesize analytes directly on beads via solid-phase synthesis. To allow bead-based multiplexing, analytes can be synthesized on spectrally encoded beads with a 1:1 linkage between analyte identity and embedded codes. Bead-bound analyte libraries can then be pooled and incubated with a fluorescently-labeled macromolecule of interest, allowing downstream quantification of interactions between the macromolecule and all analytes simultaneously via imaging alone. Extracting quantitative binding data from these images poses several computational image processing challenges, requiring the ability to identify all beads in each image, quantify bound fluorescent material associated with each bead, and determine their embedded spectral code to reveal analyte identities. Here, we present a novel open-source Python software package (the mrbles analysis package) that provides the necessary tools to: (1) find encoded beads in a bright-field microscopy image; (2) quantify bound fluorescent material associated with bead perimeters; (3) identify embedded ratiometric spectral codes within beads; and (4) return data aggregated by embedded code and for each individual bead. We demonstrate the utility of this package by applying it towards analyzing data generated via multiplexed measurement of calcineurin protein binding to MRBLEs (Microspheres with Ratiometric Barcode Lanthanide Encoding) containing known and mutant binding peptide motifs. We anticipate that this flexible package should be applicable to a wide variety of assays, including simple bead or droplet finding analysis, quantification of binding to non-encoded beads, and analysis of multiplexed assays that use ratiometric, spectrally encoded beads.},
    number = {3},
    doi = {10.1371/journal.pone.0203725}
}"""


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
