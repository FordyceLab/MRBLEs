# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : data.py
# description       : Bead Analysis data structures
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20160623
# version update    : 20160623
# version           : v0.1
# usage             : As part of Bead Analysis module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General
import sys
import os

import fnmatch
# Math
import numpy as np
from skimage.external import tifffile as tff
import pandas as pd

# TO-DO
# Check error exceptions
# Create error checking functions for clustering

# Main functions and classes
# Software-package specific functions

IMAGE_CHANNELS = {"BF" : 1,
                  "435": 2,
                  "474": 3,
                  "527": 4,
                  "536": 5,
                  "546": 6,
                  "572": 7,
                  "620": 8,
                  "630": 9,
                  "650": 10,
                  "Cy3": 11,
                  "Cy5": 12}

LANTHANIDES = {"Eu": "Europium", 
                "Sm": "Samarium", 
                "Dy": "Dysprosium", 
                "Tm": "Thulium", 
                "CeTb": "Cerium-Terbium"}

NPL_CHANNELS = ["Eu", "Sm", "Dy", "Tm", "CeTb"]

class PropEdit(object):
    def _addprop(inst, name, method):
        cls = type(inst)
        if not hasattr(cls, '__perinstance'):
            cls = type(cls.__name__, (cls,), {})
            cls.__perinstance = True
            inst.__class__ = cls
        setattr(cls, name, method)
                
    def _removeprop(inst, name):
        cls = type(inst)
        delattr(cls, name)

class FrozenClass(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True
    def _thaw(self):
        self.__isfrozen = False

class Spectra(PropEdit, FrozenClass):
    """Spectra
    Data structure for reference spectra
    """
    def __init__(self, data=None, channels=None, length=None):
        self._init_data = data
        self._data = None
        if channels is None:
            channels = []
        if data is None:
            data = []
            self._data = None
        if len(channels) > 0 and length is None:
            raise AttributeError("Must set fixed length or set data!")
        elif len(data) == 0 and len(channels) > 0:
            data = np.zeros((len(channels), length))
            for idx, name in enumerate(channels):
                self.add_channel(name, data[idx])
        elif len(data) == len(channels) and len(channels) != 0:
            for idx, name in enumerate(channels):
                self.add_channel(name, data[idx])
        self._freeze()
        
    @property
    def channels(self):
        return self._data.dtype.names

    def channel_no(self, name):
        return self.channels.index(name)

    @property
    def data(self):
        return self._data

    @property
    def all(self):
        return self._data.view(np.float64).reshape(self._data.shape + (-1,))

    def add_channel(self, name, data=None):
        """Add Channel
        Add channel to spectra object.
        """
        self._thaw()
        if data is None and self._data is not None:
            data = np.zeros(self._data.shape)
        elif data is None:
            data = np.zeros(3)
        if self._data is None:
            self._data = np.array(np.array(data), dtype=[(name, 'float64')])
        elif int(self._data.shape[0]) == len(data):
            self._data = np.lib.recfunctions.rec_append_fields(self._data, name, np.array(data), dtypes='float64')
        else:
            raise IndexError("%s wrong length, must be %i" % (data, self._data.shape[0]))
        self._addprop(name, ChannelDescriptor(name))
        self._freeze()

    def remove_channel(self, name):
        self._removeprop(name)
        self._data = np.lib.recfunctions.rec_drop_fields(self._data, name)

class ChannelDescriptor(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return obj._data[self.name]

    def __set__(self, obj, val):
        if len(obj._data.dtype.names) == 1:
            obj._data = np.array(np.array(val), dtype=[(self.name, 'float64')])
        elif obj._data.shape[0] == len(val):
            obj._data[self.name] = np.array(val, dtype='float64')
        else:
            raise IndexError("%s wrong size, must be %i" % (val, obj._data.shape[0]))

class ImageSetRead(FrozenClass):
    """Image set data object that loads image set from file(s).

    Parameters
    ----------
    file_path : string/list [string, string, ...]
        File path as string, e.g. 'C:/folder/file.tif', or as list of file paths, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].

    series : int, optional
        Sets the series number if file(s) has/have multiple series.
        Defaults to 0.

    Attributes
    ----------
    See function descriptions.

    Examples
    --------
    >>> image_data_object = ImageSetRead('C:/folder/file.tif')
    >>> image_files = ['C:/folder/file.tif', 'C:/folder/file.tif']
    >>> image_data_object = ImageSetRead(image_files)
    """
    def __init__(self, file_path, series=0, array_data='pandas'):
        self._image_data, self._metadata, self._files = self.load(file_path, series)
        self._array_method = array_data   # TO-DO
        self._freeze()
    
    # File properties and functions
    @property
    def f_size(self):
        return len(self._files)
    @property
    def f_names(self):
        return self._files
    @property
    def is_multi_file(self):
        """"Return if from multiple files.
        """
        return len(self._files) > 1

    # Series properties and functions
    @property
    def s_size(self):
        """Return series count.
        """
        return len(self._metadata['series'])

    # Channel properties and functions
    @property
    def c_size(self):
        """Return channel count.
        """
        return self._image_data.items.size
    @property
    def c_names(self):
        """"Return channel names.
        """
        return self._image_data.items
    def c_index(self, name):
        """Return channel number.
        """
        return self.c_names.get_loc(name)
    def c_(self, index):
        """Return channel data by name or number.
        
        Parameters
        ----------
        index = str/int
           Number or string of channel.
        """
        if type(index) is int:
            index = self.c_names[index]
        return self._image_data.xs(index, axis='items')

    # Position properties and functions
    @property
    def p_size(self):
        """Return position count.
        """
        return len(self._metadata['summary']['Positions'])

    # Z-slice properties and functions
    @property
    def z_size(self):
        """Return Z-slice count.
        """
        return self._metadata['summary']['Slices']

    @property
    def axes(self):
        """Return data order.

        Returns
        -------
        data_order : string
            Returns order as string with: F for file; T for timepoint, C for channel; Y for Y-axis; X for X-axis.
        
        Examples
        --------
        >>> image_data_object = ImageSetRead('C:/folder/file.tif')
        >>> image_data_object.axes
        'TCYX'
        """
        if self.is_multi_file: return ('F' + self._metadata['series'][0]['axes'])
        else: return self._metadata['series'][0]['axes']

    @property
    def ndata(self):
        return self._image_data.as_matrix()

    @classmethod
    def load(cls, file_path, series=0):
        """Load files into data structures.

        Class method. Can be used without instantiating.

        Parameters
        ----------
        file_path : string/list [string, string, ...]
            File path as string, e.g. 'C:/folder/file.tif', or as list of file paths, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].

        series : int, optional
            Sets the series number if file(s) has/have multiple series.
            Defaults to 0.

        Examples
        --------
        >>> ImageSetRead.load('C:/folder/file.tif')
        >>> image_files = ['C:/folder/file.tif', 'C:/folder/file.tif']
        >>> ImageSetRead.load(image_files)
        """
        image_data = None
        image_metadata = None
        files = None
        if type(file_path) is list:
            try:
                with tff.TiffSequence(file_path) as ts, tff.TiffFile(file_path[0]) as tf:
                    files = ts.files
                    image_metadata = cls._get_metadata(tf)
                    image_data = pd.Panel4D(ts.asarray(series=series), items = image_metadata['summary']['ChNames'])
            except ValueError:
                print("Not all images are the same shape: '%s'" % sys.exc_info()[1])
        elif type(file_path) is str:
            with tff.TiffFile(file_path) as tf:
                image_metadata = cls._get_metadata(tf)
                image_data = pd.Panel(tf.asarray(series=series), items = image_metadata['summary']['ChNames'])
                files = [tf.filename]
        else:
            raise TypeError("'file_path' is not of type 'str' or 'list' %s" % type(file_path))
        return image_data, image_metadata, files

    @staticmethod
    def scan_path(path, pattern="*.tif"):
        """Scan directory recursively for files matching the pattern.
        path = stgring, path to scan
        pattern = string, file extension
        """
        image_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    image_files.append(os.path.join(root, file))
        return image_files

    @classmethod
    def scan_paths(cls, paths, pattern=".tif"):
        """Load multiple image sets from base path(s) recursively.
        paths = string, list of strings
        pattern = string, file extension
        """
        if isinstance(paths, basekeyword):
            image_files = cls.scanPath(paths, pattern=pattern)
        elif len(paths) > 1:
            image_files = map(cls.scanPath, paths, pattern=pattern)
        else:
            print("Can't resolve base path(s).")
        return image_files

    # Private functions
    @staticmethod
    def _get_metadata(image_object):
        if image_object.is_micromanager == True:
            metadata = image_object.micromanager_metadata
            metadata['series'] = image_object.series
            return metadata
        else:
            raise ValueError("Not a Micro Manager TIFF file.")

class Bead(object):
    """Bead
    Per-bead data object"""
    def __init__(self, data):
        self._data = data

    @property
    def label_no(self):
        return self._data[0]
    @property
    def code_no(self):
        return self._data[1]
    @property
    def dims(self):
        return self._data[2]
    @property
    def ratios(self):
        return self._data[3]


class BeadImage(object):
    """Bead Image
    Per-image data object"""
    def __init__(self, image, beads, label_mask, channels):
        self._image = image
        self._beads = beads
        self._label_mask = label_mask
        self._channels = channels

    @property
    def image(self):
        return self._image
    @property
    def beads(self):
        return self._beads
    @property
    def label_mask(self):
        return self._label_mask

    @property
    def unique(self):
        for bead in self._beads:
            bead.code_no
        return self._label_mask

class BeadSet(object):
    """Bead Set
    Per-set data object"""
    def __init__(self, bead_images, channels):
        self.raw_data = data
        self._channels = channels
        self._bead_no = 0
        self._code_no = 0
        self._bead_dim = None
        
    @property
    def bead_no(self):
        return self._bead_count
    @property
    def code_no(self):
        return self._codes
    @property
    def channel_no(self):
        return len(self._channels)
    @property
    def channels(self):
        return self._channels
    @property
    def bead_dim(self):
        return self._channel_no

    def filter():
        pass