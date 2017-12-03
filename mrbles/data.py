# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Classes and Functions
==========================

This file stores the data classes and functions for the MRBLEs Analysis module.
"""

# [Future imports]
from __future__ import print_function, division

# [File header]     | Copy and edit for each file in this project!
# title             : data.py
# description       : MRBLEs - Data Structures
# author            : Bjorn Harink
# credits           :
# date              : 20160623

# [Modules]
# General Python
import sys
import os
import warnings
import re
# Data Structures
import numpy as np
import pandas as pd
import xarray as xd
from xml.dom import minidom
# File import
from skimage.external import tifffile as tff
# Graphs
import matplotlib.pyplot as plt

# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    from future.standard_library import install_aliases
    from __builtin__ import *  # NOQA
    install_aliases()

# TODO
# Check error exceptions
# Create error checking functions for clustering


# Descriptor classes


class PropEdit(object):
    """Dynamically add attributes as properties.

    Used as a inheritance class.
    """

    def _addprop(inst, name, method):
        cls = type(inst)
        # if not hasattr(cls, '__perinstance'):
        #    cls = type(cls.__name__, (cls,), {})
        #    cls.__perinstance = True
        #    inst.__class__ = cls
        setattr(cls, name, method)

    def _removeprop(inst, name):
        cls = type(inst)
        delattr(cls, name)


class FrozenClass(object):
    """Freeze class."""

    __isfrozen = False

    def __setattr__(self, key, value):
        """Create attribute."""
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def _thaw(self):
        self.__isfrozen = False


class DataOutput(object):
    """Data output methods."""

    def __init_(self):
        super(DataOutput, self).__init__()
        self._dataframe
        self._crop_x = None
        self._crop_y = None

    @property
    def data(self):
        """Return cropped) data.

        As set by default output argument:
        Xarray DataArray ('xr') or NumPy ndarray ('np').
        """
        data_crop = self._check_crop(self._dataframe)
        data_out = self._data_out(data_crop)
        return data_out

    @property
    def xdata(self):
        """Return uncropped Xarray data."""
        return self._dataframe

    @property
    def pdata(self):
        """Return uncropped Pandas data."""
        return self._dataframe

    @property
    def ndata(self):
        """Return uncropped NumPy data."""
        return self._dataframe.values

    # Crop properties and methods
    @property
    def crop_x(self):
        """Crop x slice."""
        return self._crop_x

    @crop_x.setter
    def crop_x(self, value):
        self._crop_x = self._set_slice(value)

    @property
    def crop_y(self):
        """Crop Y slice."""
        return self._crop_y

    @crop_y.setter
    def crop_y(self, value):
        self._crop_y = self._set_slice(value)

    def _data_out(self, func):
        """Check data output method setting for Numpy or Pandas."""
        if (self.output == "pd") or (self.output == "xr"):
            return func
        elif self.output == "nd":
            return func.values
        else:
            raise ValueError("Unspecified output method: '%s'." % self.output)

    def _check_crop(self, data):
        if not hasattr(self, '_crop_x'):
            self._crop_x = None
        if not hasattr(self, '_crop_y'):
            self._crop_y = None
        if self._crop_x is not None and self._crop_y is not None:
            data_crop = data.loc[dict(x=self._crop_x, y=self._crop_y)]
        elif self._crop_x is not None:
            data_crop = data.loc[dict(x=self._crop_x)]
        elif self._crop_y is not None:
            data_crop = data.loc[dict(y=self._crop_y)]
        else:
            data_crop = data
        return data_crop

    @staticmethod
    def _set_slice(values):
        if type(values) is slice or values is None:
            return values
        elif type(values) is list:
            return slice(values[0], values[1])
        else:
            raise ValueError(
                "Use slice(value, value) or [value, value] for range! Input:"
                "%s" % values)


class ChannelDescriptor(object):
    """Channel descriptor."""

    def __init__(self, name):
        """Initilize name of channel."""
        self.name = name

    def __get__(self, obj, objtype):
        """Get value."""
        return obj.spec_get(self.name)

    def __set__(self, obj, val):
        """Set value."""
        obj._dataframe[self.name] = val


# Classes


class Spectra(PropEdit, FrozenClass, DataOutput):
    """Data structure for reference spectra.

    Class can be instantiated without data. See functions.

    Parameters
    ----------
    data : ndarray
        Spectra data points per channel. Array must have same shape as spectra
        and channels. E.g. 9 channels and 4 spectra, must have (9L, 4L) shape.
        Defaults to None.
    spectra : list of strings
        Spectra names. E.g. ['Dy', 'Tm'].
        Defaults to None.
    channels : list of strings
        Channel names. E.g. ['Ex292-Em435', 'Ex292-Em465', 'Ex292-Em580'].
        Defaults to None.
    output : str, optional
        Sets default output method. Options: 'nd' for NumPy ndarray or 'pd'
        for Pandas Dataframe/Panel4D.
        Defaults to 'ndarray'.

    Functions
    ---------
    spec_add : function
        Add spectrum to object.
    spec_get : function
        Get spectrum by name or number from object.
    spec_del : function
        Delete spectrum from object.

    """

    def __init__(self, data=None, spectra=None, channels=None, output='nd'):
        """Create Spectra data object."""
        super(Spectra, self).__init__()
        self._dataframe = pd.DataFrame(data, columns=spectra, index=channels)
        self.output = output
        self._freeze()

    def __repr__(self):
        """Return Pandas dataframe representation."""
        return repr([self._dataframe])

    def __setitem__(self, index, value):
        """Set method, see method 'spec_set'."""
        self.spec_add(index, value)

    def __getitem__(self, index):
        """Get method, see method 'spec_get'."""
        return self.spec_get(index)

    # Spectrum methods and properties
    @property
    def spec_names(self):
        """Return spectra names."""
        return self._dataframe.columns.tolist()

    @property
    def spec_size(self):
        """Return spectra count."""
        return len(self.spec_names)

    def spec_index(self, name):
        """Return spectrum by index."""
        return self._dataframe.columns.get_loc(name)

    def spec_get(self, index, output=None):
        """Return spectrum data by name or number.

        Parameters
        ----------
        index : str/int
           Number or string of spectrum.
        output : string, optional
            Sets output method. Options: 'nd', NumPy array, or 'pd', Pandas
            dataframe.
            Defaults to setting in instantiation, see class description.

        Returns
        -------
        data : NumPy array/Pandas dataframe
            Returns the data as NumPy array or Pandas dataframe, depending on
            output method set by parameter 'output' or default method, if
            output not set.

        Examples
        --------
        >>> spectra_object.spec_get('Eu')
        array([ 0.01129608,  0.00995838,  0.01085018,  0.02348395,  0.00653983,
        0.00460761,  0.55960166,  0.32401902,  0.04964328])
        >>> spectra_object.spec_get(0)
        array([ 0.01129608,  0.00995838,  0.01085018,  0.02348395,  0.00653983,
        0.00460761,  0.55960166,  0.32401902,  0.04964328])
        >>> spectra_object.spec_get('Eu', output='pd')
        Ex292-Em435    0.011296
        Ex292-Em474    0.009958
        Ex292-Em527    0.010850
        Ex292-Em536    0.023484
        Ex292-Em546    0.006540
        Ex292-Em572    0.004608
        Ex292-Em620    0.559602
        Ex292-Em630    0.324019
        Ex292-Em650    0.049643
        Name: Eu, dtype: float64

        """
        if type(index) is int:
            index = self.spec_names[index]
        data = self._dataframe[index]
        if output is None:
            output = self.output
        return self._data_out(data, output)

    def spec_add(self, index, data=None, channels=None):
        """Add spectrum."""
        # Check if spectrum name is set and add as attribute
        if index not in self.spec_names and type(index) is not int:
            self._addprop(index, ChannelDescriptor(index))
        # Set channels
        if self.c_size == 0 and channels is not None:
            self._dataframe[index] = None
            for ch in channels:
                self.c_add(ch)
        if not np.array_equal(np.array(self.c_names), np.array(channels)):
            warnings.warn("Channel names not the same or channels unchecked!")
        # Set data to index
        if type(index) is int:
            index = self.spec_names[index]
        self._dataframe[index] = data

    def spec_del(self, index):
        """Delete spectrum."""
        self._dataframe = self._dataframe.drop(index, axis=1)
        self._removeprop(index)

    # Channel methods and properties
    @property
    def c_names(self):
        """Return label names."""
        return self._dataframe.index.tolist()

    @property
    def c_size(self):
        """Return label count."""
        return len(self.c_names)

    def c_index(self, name):
        """Return label index."""
        return self._dataframe.index.get_loc(name)

    def c_get(self, index, output=None):
        """Return label data by name or number.

        Parameters
        ----------
        index : str/int
           Number or string of channel.
        output : string, optional
            Sets output method. Options: 'ndarray', NumPy array, or 'pandas',
            Pandas dataframe.
            Defaults to setting in instantiation, see class description.

        Returns
        -------
        data : NumPy array/Pandas dataframe
            Returns the data as NumPy array or Pandas dataframe, depending on
            output method set by parameter 'output' or default method, if
            output not set.

        Examples
        --------
        See method 'spec_get'.

        """
        if type(index) is int:
            index = self.c_names[index]
        data = self._dataframe.loc[index]
        if output is None:
            output = self.output
        return self._data_out(data, output)

    def c_add(self, name, data=None):
        """Add channel."""
        self._dataframe.ix[name] = data

    def c_del(self, name, data=None):
        """Add channel."""
        self._dataframe = self._dataframe.drop(name)

    # Data file output/input functions
    def to_csv(self, filepath):
        """Write CSV file with reference values."""
        if self._dataframe is not None:
            self._dataframe.to_csv(filepath)
        else:
            print("No spectra to export!")

    def to_excel(self, filepath):
        """Write excel file with reference values."""
        if self._dataframe is not None:
            self._dataframe.to_excel(filepath)
        else:
            print("No spectra to export!")

    def read_csv(self, filepath):
        """Read CSV file with reference values."""
        if self._dataframe is not None:
            self._dataframe = pd.read_csv(filepath, 'rb')
        else:
            print("No spectra to export!")

    def read_excel(self, filepath, sheetname=0):
        """Read Excel file with reference values."""
        if self._dataframe is not None:
            self._dataframe = pd.read_excel(
                open(filepath, 'rb'), sheetname=sheetname)
        else:
            print("No spectra to export!")

    # Plot functions
    def plot(self, show=False):
        """Plot refrence spectra."""
        self._dataframe.plot(title="Spectra", rot=90)
        if show is True:
            plt.show()


class ImageSetRead(DataOutput):
    """Image set data object that loads image set from file(s).

    Parameters
    ----------
    file_path : string/list [string, string, ...]
        File path as string, e.g. 'C:/folder/file.tif', or as list of file
        paths, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].
    series : int, optional
        Sets the series number if file has multiple series.
        To Loads all series set to series='all'.
        Defaults to 0.
    output : str, optional
        Sets default output method. Options: 'nd' for NumPy ndarray or 'xd'
        for xarray.
        Defaults to 'ndarray'.

    Attributes
    ----------
    See function descriptions.

    Returns
    -------
    _dataframe : xarray dataframe
        Returned when calling the instance.
    _dataframe[idx] : NumPy ndarray
        Returns the index value or slice values: [start:stop:stride]. Warning:
        when using column names stop values are included.

    Examples
    --------
    >>> image_data_object = ImageSetRead('C:/folder/file.tif')
    >>> image_files = ['C:/folder/file.tif', 'C:/folder/file.tif']
    >>> image_data_object = ImageSetRead(image_files, output='xd')
    >>> image_data_object['BF', 100:400, 100:400]
    (301L, 301L)

    """

    def __init__(self, file_path, series=0, output='nd'):
        """Initialize file load object."""
        super(ImageSetRead, self).__init__()
        self._dataframe, self._metadata, self._files = \
            self.load(file_path, series)
        self.output = output

    def __repr__(self):
        """Return xarray dataframe representation."""
        return repr([self._dataframe])

    def __getitem__(self, index):
        """Get method, see method 'c_get'."""
        return self.data.loc[index]

    # Main image load function
    @classmethod
    def load(cls, file_path, series=0):
        """Load image files into data structures.

        Class method. Can be used without instantiating.

        Parameters
        ----------
        file_path : string/list [string, string, ...]
            File path as string, e.g. 'C:/folder/file.tif', or as list of file
            paths, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].

        series : int, optional
            Sets the series number if file has multiple series (or positions).
            Use series='all' for loading all series.
            Defaults to 0.

        Examples
        --------
        >>> ImageSetRead.load('C:/folder/file.tif')
        >>> image_files = ['C:/folder/file.tif', 'C:/folder/file.tif']
        >>> ImageSetRead.load(image_files)

        """
        if type(file_path) is str:
            file_path = [file_path]
        with tff.TiffSequence(file_path, pattern='XYCZT') as ts, \
                tff.TiffFile(file_path[0]) as tf:
            files = ts.files
            image_metadata = cls._get_metadata(tf)
            if len(tf.series) > 1 and series == 'all':
                data = []
                for idx, serie in enumerate(tf.series):
                    data.append(ts.asarray(series=idx))
                panel_data = np.array(data)
            else:
                panel_data = ts.asarray(series=series)
        if len(file_path) == 1 and series != 'all':
            panel_data = np.vstack(panel_data)
        image_data = cls._convert_to_xd(
            panel_data, image_metadata, file_path, series)
        return image_data, image_metadata, files

    # Channel properties and methods
    @property
    def c_size(self):
        """Return channel count."""
        return self._dataframe.c.size

    @property
    def c_names(self):
        """Return channel names."""
        return self._dataframe.c.values

    def c_index(self, name):
        """Return channel number."""
        return self.c_names.get_loc(name)

    # File properties and methods
    @property
    def f_size(self):
        """Return file count."""
        return len(self._files)

    @property
    def f_names(self):
        """Return file names."""
        return self._files

    @property
    def is_multi_file(self):
        """Return if from multiple files."""
        return len(self._files) > 1

    # Series properties and methods
    @property
    def s_size(self):
        """Return series count."""
        return len(self._metadata['series'])

    # Position properties and methods
    @property
    def p_size(self):
        """Return position count."""
        return len(self._metadata['series'])

    # Z-slice properties and methods
    @property
    def z_size(self):
        """Return Z-slice count."""
        return self._metadata['summary']['Slices']

    # Time properties and methods
    @property
    def t_size(self):
        """Return time-point count."""
        return len(np.unique(self._metadata['index_map']['frame']))

    @property
    def t_interval(self):
        """Return set time interval.

        Default in milliseconds (ms), check object.t_unit for time unit.
        """
        return self._metadata['summary']['Interval_ms']

    @property
    def t_deltas(self):
        """Return time deltas between each image acquisition.

        Default in milliseconds (ms), check object.t_unit for time unit.
        """
        xml_string = self._metadata['series'][0].pages[0].tags['image_description'].value  # NOQA
        xml_tree = minidom.parseString(xml_string)
        t_deltas = [float(xm.attributes['DeltaT'].value)
                    for xm in xml_tree.getElementsByTagName('Plane')]
        return t_deltas

    @property
    def t_unit(self):
        """Return time unit."""
        xml_string = self._metadata['series'][0].pages[0].tags['image_description'].value  # NOQA
        xml_tree = minidom.parseString(xml_string)
        xm = xml_tree.getElementsByTagName('Plane')
        t_unit = str(xm[0].attributes['DeltaTUnit'].value)
        return t_unit

    # Axes properties and methods
    @property
    def axes(self):
        """Return data order.

        Returns
        -------
        data_order : string
            Returns order as string with: P for position; F for file; T for
            timepoint, C for channel; Y for Y-axis; X for X-axis.

        Examples
        --------
        >>> image_data_object = ImageSetRead('C:/folder/file.tif')
        >>> image_data_object.axes
        'TCYX'

        """
        return ''.join(self._dataframe.dims).upper()

    @staticmethod
    def scan_path(path, pattern="*.tif"):
        """Scan folder recursively for files matching the pattern.

        Parameters
        ----------
        path : string
            Folder path as string, e.g. r'C:/folder/file.tif'.

        pattern : string
            File extenstion of general file pattern as search string,
            e.g. '20160728_MOL_*'
            Defaults to '*.tif'.

        """
        image_files = []
        r = re.compile(pattern)
        for root, dirs, files in os.walk(path):
            file_list = [os.path.join(root, x) for x in files if r.match(x)]
            if file_list:
                image_files.append(file_list)
        return np.hstack(image_files).tolist()

    @classmethod
    def scan_paths(cls, paths, pattern=".tif"):
        """Scan folders recursively for files matching the pattern.

        Parameters
        ----------
        path : list
            Folder paths as list, e.g. ['C:/folder/file.tif', ...].

        pattern : string
            File extenstion of general file pattern, e.g. '20160728_MOL_*'
            Defaults to '*.tif'.

        """
        if type(paths) is str:
            image_files = cls.scan_path(paths, pattern=pattern)
        elif len(paths) > 1:
            image_files = [cls.scan_path(path, pattern=pattern)
                           for path in paths]
        else:
            print("Can't resolve base path(s).")
        return image_files

    # Private methods
    @staticmethod
    def _convert_to_xd(data, metadata, file_path, series):
        """Convert data and metadata to xarray DataArray."""
        if data.ndim == 2:
            panel_data = xd.DataArray(data, dims=['y', 'x'])
        else:
            # Added check if using newer version of Scikit-Image
            try:
                if type(metadata['series'][0]) is tff.tifffile.TiffPageSeries:
                    dims = [letter.lower()
                            for letter in metadata['series'][0].axes]
            except DeprecationWarning:
                # For backwards compatibility of Scikit-Image versions <0.12.3.
                dims = [letter.lower()
                        for letter in metadata['series'][0]['axes']]
                warnings.warn(
                    "Scikit-Image latest update has changed method to retrieve"
                    "metadata. Please upgrade to latest Scikit-Image package.")
            if len(metadata['series']) > 1 and (series is 'all'):
                dims.insert(0, 'p')
                data = np.squeeze(data)
            if 'i' in dims:
                dims[dims.index('i')] = 'c'
            if len(file_path) > 1:
                dims.insert(0, 'f')
            panel_data = xd.DataArray(data, dims=dims, coords={
                                      'c': metadata['summary']['ChNames']})
        return panel_data

    @staticmethod
    def _get_metadata(image_object):
        if image_object.is_micromanager is True:
            metadata = image_object.micromanager_metadata
            metadata['series'] = image_object.series
            return metadata
        else:
            warnings.warn("Not a Micro Manager TIFF file.")
            return None
