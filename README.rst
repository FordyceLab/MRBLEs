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

MRBLEs project
--------------
MRBLEs (Microspheres with Ratiometric Barcode Lanthanide Encoding) rely on
spectral multiplexing to track analytes throughout an experiment. In these
assays, we can create microspheres containing > 1,000 unique ratios of
lanthanide nanophosphors that can be uniquely identified via imaging alone.
We are currently developing new assays that use these microspheres to
understand how signaling proteins recognize their peptide substrates and to
improve our ability to extract information from single cells.

Links
-----
Documentation: https://fordycelab.github.io/MRBLEs/

Source code: https://github.com/FordyceLab/MRBLEs/

Installation
------------
The MRBLEs software package is a Python Package available through the Python Package Index (PyPI): https://pypi.org/.
To makes us of `mrbles` follow the following instructions:
* Install (preferably) Python >3.5 `Windows <https://www.python.org/downloads/windows/>`_ or `MacOS <https://www.python.org/downloads/mac-osx/>`_.

 `mrbles` is built to be compatible with Pyhton >2.7, but its dependencies (other Python Packages) are getting less and less support for Python 2.7.
 - Detailed Windows installation: https://docs.python.org/3/using/windows.html
   - Must have C++ compiler installed to be able to install package from source code. See link above.
   - If using Python 2.7, also install `Microsoft Visual C++ Compiler for Python 2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_.
   - If using Python >3.6, also install `Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017>`_.
   - More info on `compilers here <https://wiki.python.org/moin/WindowsCompilers#Which_Microsoft_Visual_C.2B-.2B-_compiler_to_use_with_a_specific_Python_version_.3F>`_.
 - Detailed MacOS installation: https://docs.python-guide.org/starting/install3/osx/#install3-osx
   - Must have XCode (or other C++ compiler) installed to be able to install package from source code. See link above.
 - During installation make sure to tick the box "Add Pyhton 3.x to PATH" (Windows) or manually add Python root and Scripts to system path.


- Once Python is installed it can be accessed using your systems' terminal, using:
 - `> python` or `> python3` depending if only Python 3 installed or both Python 2 & 3.


- Now the `mrbles` package can be installed, using the following command in the terminal (exit Python environment first, or start a new terminal):
 - `> pip install mrbles` or `> pip3 install mrbles` depending if only Python 3 installed or both Python 2 & 3.