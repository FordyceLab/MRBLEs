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
To makes us of `mrbles` follow the following instructions.

Python installation
~~~~~~~~~~~~~~~~~~~
Install (preferably) Python >3.5 `Windows <https://www.python.org/downloads/windows/>`_ or `MacOS
<https://www.python.org/downloads/mac-osx/>`_. `mrbles` is built to be compatible with Pyhton >2.7,
but its dependencies (other Python Packages) are getting less and less support for Python 2.7.

Windows
_______
This is based on the detailed Windows installation instructions from the official Python page:
https://docs.python.org/3/using/windows.html

* Install Python by downloading the latest version of Python: https://www.python.org/downloads/.
* Execute the installation file and follow the instructions.
* On the first window make sure to tick the box "*Add Pyhton x.x to PATH*".
* Preferably, use the "*Customize installation*".
* On the second window use the standard selections, or if other people are using the same machine, also select "*Install for all users*".
* Preferably, use a custumized installation location, e.g. for Python 3.7 "C:\Pyhton37". This way it is easy to access when necessary.
* Press "*Install*".

To make use of Python packages that do not come with a pre-compiled file (which `mrbles` does), as in they need to be
compiled on your computer from the source code. It is required to install a C++ compiler.

* If using Python 2.7, install `Microsoft Visual C++ Compiler for Python 2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_.
* If using Python >3.6, install `Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017>`_.
* More info on `compilers here <https://wiki.python.org/moin/WindowsCompilers#Which_Microsoft_Visual_C.2B-.2B-_compiler_to_use_with_a_specific_Python_version_.3F>`_.

Mac OSx
_______
This is based on the detailed Mac OSx installation instructions from the Python-Guide page: https://docs.python-guide.org/starting/install3/osx/#install3-osx

The installation for Mac OSx is slightly more complicated, but will help you in the future with Python dependencies.

* First, download and install XCode: https://developer.apple.com/xcode/.
  - This is required for the installation of Python packages (C++ compiler) and for the installation of Brew, which is software that makes installing Python and dependencies much easier.
* Second, install brew using the following command in Mac OSx Terminal:

  `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)`


The `mrbles` package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once Python is installed it can be accessed using your systems' Command Prompt (Windows) or Terminal (Mac OSx), using: `python` (if both Python 2 & 3 are installed this is: `python3` or `python3`).

To install packages one can use the following commands, using the packag name:

* `pip install package_name` or
* `python -m pip install package_name`

Visit the Python Package Index (PyPI) for package names: https://pypi.org/.

Now that everything is installed the `mrbles` package can be installed, using the following commands in the terminal:

* `pip install mrbles` (if both Python 2 & 3 are installed this is: `pip3` or `pip`) or
* `python -m pip install mrbles` (if both Python 2 & 3 are installed this is: `pip3` or `pip`)

Using the `mrbles` package
--------------------------
The `mrbles` package automatically installs the Jupyter Notebook environment: https://http://jupyter.org/.

To test the `mrbles` package download the example Notebook and the data files:

* Notebook: https://github.com/FordyceLab/MRBLEs/blob/master/examples/example-notebook-shipped-data.ipynb.
* Data file: https://github.com/FordyceLab/MRBLEs/tree/master/data.
  - The quickest way is to download all the GitHub files in a zip file: https://github.com/FordyceLab/MRBLEs/archive/master.zip.
* Place the Notebook file in a location at your convenience, e.g. "C:\\docs\\mrbles_test" or "/users/your_name/docs/mrbles_test".
* Place the data files in the sub-folder "data" of the location of the Notebook file.
* Open a Command Prompt or Terminal and navigate to that folder location.
* Now start your Jupyter Notebook environment by using the following command:
`jupyter notebook`
* This should open your default browser and display the contents of the folder your started the Jupyter Notebook environment from.
* Click on the downloaded "example-notebook-shipped-data.ipynb", this will open that file.
* Following the instruction in the opened Notebook.
* Fore more information on using Jupyter Notebook: https://jupyter-notebook-beginner-guide.readthedocs.io.