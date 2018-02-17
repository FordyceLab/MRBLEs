# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Classes and Functions
==========================

This file stores the data classes and functions for the MRBLEs Analysis module.
"""

# [Future imports]
from __future__ import (absolute_import, division, print_function)
from builtins import (str, super, object)

# [File header]     | Copy and edit for each file in this project!
# title             : data.py
# description       : MRBLEs - Data Structures
# author            : Bjorn Harink
# credits           :
# date              : 20160623

# [Modules]
# General Python
# import sys
import os
import warnings
import re
# Data Structures
from xml.dom import minidom
import numpy as np
import pandas as pd
import xarray as xr
# File import
from skimage.external import tifffile as tff

# TODO: Check error exceptions
# TODO: Create error checking functions for clustering
# TODO: Replace py 2 3 compatibility stuff


# Descriptor classes


class TableDataFrame(object):
    """Pandas based dataframe object for table data."""

    def __init__(self, data=None, flag_filt=True, **kwargs):
        """Instantiate table data strcuture."""
        super(TableDataFrame, self).__init__()
        self.flag_filt = flag_filt
        if 'flag_name' in kwargs:
            self.flag_name = kwargs['flag_name']
        else:
            self.flag_name = 'flag'
        self._dataframe = data

    def __repr__(self):
        """Return dataframe representation."""
        return repr(self.data)

    def __getitem__(self, index):
        """Get method."""
        return self.data.loc[index]

    @property
    def data(self):
        """Return Pandas dataframe object."""
        data = self._check_flag(self._dataframe)
        return data

    @property
    def pdata(self):
        """Return unflagged Pandas dataframe."""
        return self._dataframe

    def combine(self, data):
        """Combine data with dataframe.

        Parameters
        ----------
        data : Pandas DataFrame

        """
        if isinstance(self.data, pd.DataFrame) & \
                isinstance(data, pd.DataFrame):
            index = data.index
            self._dataframe = pd.concat([data.reset_index(drop=True),
                                         self.data.reset_index(drop=True)],
                                        axis=1)
            self._dataframe.index = index
        else:
            raise ValueError("Not Pandas DataFrame: %s." % type(data))

    @classmethod
    def flatten_dict(cls, dict_data, prefix='.'):
        """Flatten dictionary with given prefix."""
        def items():
            # A closure for recursively extracting dict like values
            for key, value in dict_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in cls.flatten_dict(value).items():
                        yield key + prefix + sub_key, sub_value
                else:
                    yield key, value
        return dict(items())

    @staticmethod
    def get_set_names(data_set, set_dim='set'):
        """Return list of sets."""
        if set_dim in data_set.index.names:
            sets_list = list(data_set.groupby(set_dim).groups.keys())
        else:
            sets_list = None
        return sets_list

    def _check_flag(self, data):
        if self.flag_name in data.columns and self.flag_filt is True:
            flag_out_data = data[data[self.flag_name] == False]  # NOQA: E712
        else:
            flag_out_data = data
        return flag_out_data


class ImageDataFrame(object):
    """Xarray based dataframe object for images."""

    def __init__(self, data=None):
        """Instantiate image data strcuture."""
        super(ImageDataFrame, self).__init__()
        self._dataframe = data
        self._crop_x = slice(None, None, None)
        self._crop_y = slice(None, None, None)

    def __repr__(self):
        """Return dataframe representation."""
        return repr(self.data)

    def __getitem__(self, index):
        """Get method."""
        if isinstance(index, str):
            if isinstance(self.data, dict):
                data = self.data[index]
            else:
                data = self.data.loc[index]
        elif isinstance(index, slice):
            data = self.data
        elif index[0] == slice(None, None, None):
            data = {key: data.loc[index[1:]]
                    for key, data in self.data.items()}
        else:
            data = self.data[index[0]].loc[index[1:]]
        return data

    @property
    def data(self):
        """Return cropped Xarray dataframe."""
        return self._check_crop(self._dataframe)

    @property
    def xdata(self):
        """Return uncropped Xarray dataframe."""
        return self._dataframe

    def combine(self, images):
        """Combine iamges with dataframe.

        Parameters
        ----------
        images : Xarray DataArray, dict of Xarray DataArrays

        """
        if isinstance(self.data, dict) & isinstance(images, dict):
            self._dataframe = {
                key: self.data[key].combine_first(
                    images[key].astype(self.data[key])
                )
                for key in self.data.keys()
            }
        elif isinstance(self.data, xr.DataArray) & \
                isinstance(images, xr.DataArray):
            self._dataframe = self.data.combine_first(
                images.astype(self.data)
            )
        else:
            raise ValueError("Not dict or Xarray DataArray: %s."
                             % type(images))

    # Crop properties
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

    # Crop methods
    def _check_crop(self, data):
        if isinstance(data, dict):
            data_crop = {key: value.loc[dict(x=self._crop_x, y=self._crop_y)]
                         for key, value in data.items()}
        elif isinstance(data, xr.DataArray):
            data_crop = data.loc[dict(x=self._crop_x, y=self._crop_y)]
        else:
            data_crop = None
        return data_crop

    @staticmethod
    def get_set_names(data_set, set_dim='c'):
        """Return list of sets."""
        sets_list = list(data_set.coords[set_dim].values)
        return sets_list

    @staticmethod
    def _set_slice(values):
        if isinstance(values, list):
            slice_values = slice(values[0], values[1])
        elif values is None:
            slice_values = slice(None)
        else:
            slice_values = values
        return slice_values


# Classes


class ImageSetRead(ImageDataFrame):
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

    def __init__(self, file_path, series=0, output='xr'):
        """Initialize file load object."""
        super(ImageSetRead, self).__init__()
        self._dataframe, self._metadata, self._files = \
            self.load(file_path, series)
        self.output = output

    def __repr__(self):
        """Return xarray dataframe representation."""
        return repr([self._dataframe])

    def __getitem__(self, index):
        """Get method."""
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
        if isinstance(file_path, str):
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
        paths : list
            Folder paths as list, e.g. ['C:/folder/file.tif', ...].

        pattern : string
            File extension of general file pattern, e.g. '20160728_MOL_*'
            Defaults to '*.tif'.

        """
        if isinstance(paths, str):
            image_files = cls.scan_path(paths, pattern=pattern)
        elif len(paths) > 1:
            image_files = [cls.scan_path(path, pattern=pattern)
                           for path in paths]
        else:
            print("Can't resolve base path(s).")
            image_files = None
        return image_files

    # Private methods
    @staticmethod
    def _convert_to_xd(data, metadata, file_path, series):
        """Convert data and metadata to xarray DataArray."""
        if data.ndim == 2:
            panel_data = xr.DataArray(data, dims=['y', 'x'])
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
            panel_data = xr.DataArray(
                data,
                dims=dims,
                coords={'c': metadata['summary']['ChNames']},
                encoding={'dtype': np.uint16}
            )
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
