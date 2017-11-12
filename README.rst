MRBLEs Analysis Package
=======================

This project aims to provide a package to **find**, **decode**, **extract**,
**analyze** and MRBLEs using a set of images.

This package provides the tools to: (1) **find** the MRBLEs in a monochrome
brightfield microscopy image; (2) **decode** the MRBLEs beads by spectral
unmxing, using reference spectra, and then spectrally decode the found beads
using Iterative Closest Point Matching and Gaussian Mixture Modeling; (3)
**extract**, statistical values of interest in additional fluorescence
channels using the morphology of the MRBLEs, their locations, and respective
code; (4) **analyze** affinity information based on titrations of MRBLEs
assays.

----

MRBLEs (Microspheres with Ratiometric Barcode Lanthanide Encoding) rely on
spectral multiplexing to track analytes throughout an experiment. In these
assays, we can create microspheres containing > 1,000 unique ratios of
lanthanide nanophosphors that can be uniquely identified via imaging alone.
We are currently developing new assays that use these microspheres to
understand how signaling proteins recognize their peptide substrates and to
improve our ability to extract information from single cells.
