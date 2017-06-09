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
# version update    : 20170601
# version           : v0.5
# usage             : As part of Bead Analysis module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General Python
import sys
import os
import warnings
import fnmatch
# Data Structures
import numpy as np
import pandas as pd
import xarray as xr
# File import
from skimage.external import tifffile as tff
# Graphs
import matplotlib.pyplot as plt

# TO-DO
# Check error exceptions
# Create error checking functions for clustering

### Main functions and classes

## Functions


## Classes
class PropEdit(object):
    """Dynamically add attributes as properties.

    Used as a inheritance class.
    """
    def _addprop(inst, name, method):
        cls = type(inst)
        #if not hasattr(cls, '__perinstance'):
        #    cls = type(cls.__name__, (cls,), {})
        #    cls.__perinstance = True
        #    inst.__class__ = cls
        setattr(cls, name, method)

    def _removeprop(inst, name):
        cls = type(inst)
        delattr(cls, name)


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def _thaw(self):
        self.__isfrozen = False


class OutputMethod():
    """Data output methods.
    """
    def _data_out(self, func, output=None):
        """Checks data output method setting for Numpy or Pandas.
        """
        if (output == "pd") or (output == "xr"):
            return func
        elif output == "nd":
            return func.values
        else:
            raise ValueError("Unspecified output method: '%s'." % output)

    # TO-DO!
    def _crop_data(self):
        pass

class ChannelDescriptor(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return obj.spec_get(self.name)

    def __set__(self, obj, val):
        obj._dataframe[self.name] = val


class Spectra(PropEdit, FrozenClass, OutputMethod):
    """Data structure for reference spectra

    Class can be instantiated without data. See functions.

    Parameters
    ----------
    data : ndarray
        Spectra data points per channel. Array must have same shape as spectra and channels.
        E.g. 9 channels and 4 spectra, must have (9L, 4L) shape.
        Defaults to None.

    spectra : list of strings
        Spectra names. E.g. ['Dy', 'Tm'].
        Defaults to None.

    channels : list of strings
        Channel names. E.g. ['Ex292-Em435', 'Ex292-Em465', 'Ex292-Em580'].
        Defaults to None.

    output : str, optional
        Sets default output method. Options: 'nd' for NumPy ndarray or 'pd' for Pandas Dataframe/Panel4D.
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
        super(Spectra, self).__init__()
        self._dataframe = pd.DataFrame(data, columns=spectra, index=channels)
        self.output = output
        self._freeze()

    def __repr__(self):
        """Returns Pandas dataframe representation.
        """
        return repr([self._dataframe])

    def __setitem__(self, index, value):
        """Set method, see method 'spec_set'
        """
        self.spec_add(index, value)

    def __getitem__(self, index):
        """Get method, see method 'spec_get'.
        """
        return self.spec_get(index)

    # Spectrum methods and properties
    @property
    def spec_names(self):
        """Return spectra names.
        """
        return self._dataframe.columns.tolist()
 
    @property
    def spec_size(self):
        """Return spectra count.
        """
        return len(self.spec_names)
    
    def spec_index(self, name):
        """Return spectrum by index.
        """
        return self._dataframe.columns.get_loc(name)

    def spec_get(self, index, output=None):
        """Return spectrum data by name or number.

        Parameters
        ----------
        index : str/int
           Number or string of spectrum.

        output : string, optional
            Sets output method. Options: 'nd', NumPy array, or 'pd', Pandas dataframe.
            Defaults to setting in instantiation, see class description.

        Returns
        -------
        data : NumPy array/Pandas dataframe
            Returns the data as NumPy array or Pandas dataframe, depending on output method set by parameter 'output' or default method, if output not set.

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
        """Add spectrum.
        """
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
        self._dataframe = self._dataframe.drop(index, axis=1)
        self._removeprop(index)

    # Channel methods and properties    
    @property
    def c_names(self):
        """Return label names.
        """
        return self._dataframe.index.tolist()
    
    @property
    def c_size(self):
        """Return label count.
        """
        return len(self.c_names)
    
    def c_index(self, name):
        """Return label index.
        """
        return self._dataframe.index.get_loc(name)

    def c_get(self, index, output=None):
        """Return label data by name or number.

        Parameters
        ----------
        index : str/int
           Number or string of channel.

        output : string, optional
            Sets output method. Options: 'ndarray', NumPy array, or 'pandas', Pandas dataframe.
            Defaults to setting in instantiation, see class description.

        Returns
        -------
        data : NumPy array/Pandas dataframe
            Returns the data as NumPy array or Pandas dataframe, depending on output method set by parameter 'output' or default method, if output not set.

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
        """Add channel.
        """
        self._dataframe.ix[name] = data

    def c_del(self, name, data=None):
        """Add channel.
        """
        self._dataframe = self._dataframe.drop(name)

    # Data properties and methods
    @property
    def data(self):
        return self._data_out(self._dataframe, self.output)
    @property
    def ndata(self):
        return self._dataframe.values
    @property
    def pdata(self):
        return self._dataframe
    
    # Data file output/input functions
    def to_csv(self, filepath):
        if self._dataframe is not None:
            self._dataframe.to_csv(filepath)
        else:
            print("No spectra to export!")

    def to_excel(self, filepath):
        if self._dataframe is not None:
            self._dataframe.to_excel(filepath)
        else:
            print("No spectra to export!")

    def read_csv(self, filepath):
        if self._dataframe is not None:
            self._dataframe = pd.read_csv(filepath,'rb')
        else:
            print("No spectra to export!")

    def read_excel(self, filepath, sheetname=0):
        if self._dataframe is not None:
            self._dataframe = pd.read_excel(open(filepath,'rb'), sheetname=sheetname)
        else:
            print("No spectra to export!")

    # Plot functions
    def plot(self, show=False):
        self._dataframe.plot(title="Spectra", rot=90)
        if show is True:
            plt.show()


class ImageSetRead(FrozenClass, OutputMethod):
    """Image set data object that loads image set from file(s).

    Parameters
    ----------
    file_path : string/list [string, string, ...]
        File path as string, e.g. 'C:/folder/file.tif', or as list of file paths, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].

    series : int, optional
        Sets the series number if file has multiple series. 
        To Loads all series set to series='all'.
        Defaults to 0.

    output : str, optional
        Sets default output method. Options: 'nd' for NumPy ndarray or 'xr' for xarray.
        Defaults to 'ndarray'.

    Attributes
    ----------
    See function descriptions.

    Returns
    -------
    _dataframe : xarray dataframe
        Returned when calling the instance.
    _dataframe[idx] : NumPy ndarray
        Returns the index value or slice values: [start:stop:stride]. Warning: when using column names stop values are included.

    Examples
    --------
    >>> image_data_object = ImageSetRead('C:/folder/file.tif')
    >>> image_files = ['C:/folder/file.tif', 'C:/folder/file.tif']
    >>> image_data_object = ImageSetRead(image_files, output='pd')
    >>> image_data_object['BF', 100:400, 100:400]
    (301L, 301L)
    >>> image_data_object['BF', 100:400, 100:400]
    (301L, 301L)
    """
    def __init__(self, file_path, series=0, output='nd'):
        super(ImageSetRead, self).__init__()
        self._dataframe, self._metadata, self._files = \
            self.load(file_path, series)
        self.output = output
        self.crop_x = None
        self.crop_y = None
        self._freeze()

    def __repr__(self):
        """Returns xarray dataframe representation.
        """
        return repr([self._dataframe])

    def __getitem__(self, index):
        """Get method, see method 'c_get'.
        """
        return self.c_get(index)

    # Main image load function
    @classmethod
    def load(cls, file_path, series=0):
        """Load image files into data structures.

        Class method. Can be used without instantiating.

        Parameters
        ----------
        file_path : string/list [string, string, ...]
            File path as string, e.g. 'C:/folder/file.tif', or as list of file paths, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].

        series : int, optional
            Sets the series number if file has multiple series. Use series='all' for loading all series.
            Defaults to 0.

        Examples
        --------
        >>> ImageSetRead.load('C:/folder/file.tif')
        >>> image_files = ['C:/folder/file.tif', 'C:/folder/file.tif']
        >>> ImageSetRead.load(image_files)
        """
        if type(file_path) is str: 
            file_path = [file_path]
        with tff.TiffSequence(file_path, pattern='XYCZT') as ts, tff.TiffFile(file_path[0]) as tf:
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
        image_data = cls.convert_to_xd(panel_data, image_metadata, file_path, series)
        return image_data, image_metadata, files

    # Channel properties and methods
    @property
    def c_size(self):
        """Return channel count.
        """
        return self._dataframe.c.size

    @property
    def c_names(self):
        """"Return channel names.
        """
        return self._dataframe.c.values

    def c_index(self, name):
        """Return channel number.
        """
        return self.c_names.get_loc(name)

    def c_get(self, index, output=None):
        """Return data by chanel name and/or number.

        Parameters
        ----------
        index : str/int
           Number or string of channel name.

        output : string, optional
            Sets output method. Options: 'ndarray', NumPy array, or 'pandas', Pandas dataframe.
            Defaults to setting in instantiation, see class description.

        Returns
        -------
        data : NumPy array/xarray DataArray
            Returns the data as NumPy array or xarray DataArray, depending on output method set by parameter 'output' or default method, if output not set.
        """
        if output is None:
            output = self.output
        data_crop = self.check_crop()
        data = self._data_out(data_crop.loc[index], output)
        return data

    # File properties and methods
    @property
    def f_size(self):
        """Return file count.
        """
        return len(self._files)

    @property
    def f_names(self):
        """Return file names.
        """
        return self._files

    @property
    def is_multi_file(self):
        """"Return if from multiple files.
        """
        return len(self._files) > 1

    # Series properties and methods
    @property
    def s_size(self):
        """Return series count.
        """
        return len(self._metadata['series'])

    # Position properties and methods
    @property
    def p_size(self):
        """Return position count.
        """
        return len(self._metadata['series'])

    # Z-slice properties and methods
    @property
    def z_size(self):
        """Return Z-slice count.
        """
        return self._metadata['summary']['Slices']

    # Time properties and methods
    @property
    def t_size(self):
        """Return time count.
        """
        return self._dataframe.shape[0]

    @property
    def t_interval(self):
        """Return time interval.
        """
        return self._metadata['summary']['Interval_ms']

    # Axes properties and methods
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
        return ''.join(self._dataframe.dims).upper()
    
    # Data properties and methods
    @property
    def data(self):
        """Return data as xarray DataArray or NumPy ndarray, as set by default output argument.
        """
        return self._data_out(self._dataframe, self.output)

    @property
    def ndata(self):
        """Return as NumPy ndarray.
        """
        return self._dataframe.values

    @property
    def xdata(self):
        """Return xarray DataArray.
        """
        return self._dataframe

    # Crop properties and methods
    @property
    def crop_x(self):
        return self._crop_x
    @crop_x.setter
    def crop_x(self, value):
        self._crop_x = self.set_slice(value)

    @property
    def crop_y(self):
        return self._crop_y
    @crop_x.setter
    def crop_y(self, value):
        self._crop_y = self.set_slice(value)

    def set_crop(self, crop_x, crop_y):
        self.crop_x = crop_x
        self.crop_y = crop_y

    def check_crop(self):
        if self.crop_x is not None and self.crop_y is not None:
            data_crop = self._dataframe.loc[dict(x=self.crop_x, y=self.crop_y)]
        elif self.crop_x is not None:
            data_crop = self._dataframe.loc[dict(x=self.crop_x)]
        elif self.crop_y is not None:
            data_crop = self._dataframe.loc[dict(y=self.crop_y)]
        else:
            data_crop = self._dataframe
        return data_crop

    @staticmethod
    def set_slice(values):
        if type(values) is slice or values is None:
            return values
        elif type(values) is list:
            return slice(values[0], values[1])
        else:
            raise ValueError("Use slice(value, value) or [value, value] for range! Input: %s" % values)

    @staticmethod
    def convert_to_xd(data, metadata, file_path, series):
        """Convert data and metadata to xarray DataArray.
        """
        if data.ndim == 2:
            panel_data = xr.DataArray(data, dims=['y','x'])
        else:
            # Added check if using newer version of Scikit-Image
            if type(metadata['series'][0]) is tff.tifffile.TiffPageSeries:
                dims = [letter.lower() for letter in metadata['series'][0].axes]
            # For Scikit-Image versions 0.12.1 and below
            else:
                dims = [letter.lower() for letter in metadata['series'][0]['axes']]
            if len(metadata['series']) > 1 and (series is 'all'):
                dims.insert(0, 'p')
                data = np.squeeze(data)
            if len(file_path) > 1:
                dims.insert(0, 'f')
            if 'i' in dims:
                dims[dims.index('i')] = 'c'
            panel_data = xr.DataArray(data, dims=dims, coords={'c':metadata['summary']['ChNames']})
        return panel_data


    @staticmethod
    def scan_path(path, pattern="*.tif"):
        """Scan folder recursively for files matching the pattern.

        Parameters
        ----------
        path : string
            Folder path as string, e.g. 'C:/folder/file.tif' or r'C:\folder\file.tif'.

        pattern : string
            File extenstion of general file pattern as search string, e.g. '20160728_MOL_*'
            Defaults to '*.tif'.
        """
        image_files = []
        for root, dirs, files in os.walk(path):
            for file in fnmatch.filter(files, pattern):
                image_files.append(os.path.join(root, file))
        return image_files

    @classmethod
    def scan_paths(cls, paths, pattern=".tif"):
        """Scan folders recursively for files matching the pattern.

        Parameters
        ----------
        path : list
            Folder paths as list, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].

        pattern : string
            File extenstion of general file pattern as search string, e.g. '20160728_MOL_*'
            Defaults to '*.tif'.
        """
        if type(paths) is str:
            image_files = cls.scanPath(paths, pattern=pattern)
        elif len(paths) > 1:
            image_files = map(cls.scanPath, paths, pattern=pattern)
        else:
            print("Can't resolve base path(s).")
        return image_files

    # Private methods
    @staticmethod
    def _get_metadata(image_object):
        if image_object.is_micromanager == True:
            metadata = image_object.micromanager_metadata
            metadata['series'] = image_object.series
            return metadata
        else:
            warnings.warn("Not a Micro Manager TIFF file.")
            return None


## Experimental Data Types
class BeadImage(object):
    def __init__(self, data):
        dedault_columns = ('code',
                           'dim')
        self._dataframe = pd.DataFrame(columns=dedault_columns)

    def __repr__(self):
        return repr([self._dataframe])

    def __setitem__(self, index, value):
        if type(index) is int:
            index = self.columns[index]
        self._dataframe[index] = value

    def __getitem__(self, index):
        return self._dataframe.ix[index]


class BeadSet(object):
    """Bead Set
    Per-set data object"""

    def __init__(self):
        self._dataframe = None

    def __repr__(self):
        return repr([self._dataframe])

    def __setitem__(self, index, value):
        if type(index) is int:
            index = self.columns[index]
        self._dataframe[index] = value

    def __getitem__(self, index):
        return self._dataframe.ix[index]

    @property
    def columns(self):
        return self._dataframe.columns.values

    @property
    def count(self):
        return self._dataframe.index.size

    @property
    def code(self, value=None):
        if key is not None:
            return self._dataframe[self._dataframe['code']==value].values
        else:
            return self._dataframe['code'].values
    @code.setter
    def code(self, value):
        self._dataframe['code'] = value

    @property
    def lbl(self):
        return self._dataframe['lbl'].values
    @lbl.setter
    def ratio(self, value):
        self._dataframe['lbl'] = value

    @property
    def img(self):
        return self._dataframe['img'].values
    @img.setter
    def ratio(self, value):
        self._dataframe['img'] = value

    @property
    def dim(self):
        return [self._dataframe['R'].values,self._dataframe['Y'].values,self._dataframe['X'].values]
    @dim.setter
    def dim(self, value):
        """Insert list for [R,Y,X]
        """
        self._dataframe['R'] = value[:,0]
        self._dataframe['R'] = value[:,1]
        self._dataframe['R'] = value[:,2]

    def get_median(mask, images, label=None):
        """Get Median Intensities of each object location from the given image.
        mask = labeled mask of objects
        images = image set of spectral images
        """
        if label is None:
            idx = np.arange(1, len(np.unique(mask)))
        else:
            idx = label
        data_size = len(np.unique(mask)) - 1
        channel_no = images[:, 0, 0].size
        channels = xrange(channel_no)
        medians_data = np.empty((data_size, channel_no))
        for ch in channels:
            # Get median value of each object
            medians_data[:, ch] = ndi.labeled_comprehension(
                images[ch, :, :], mask, idx, np.median, float, -1)
        return medians_data

    def get_data(mask, images, function, label=None):
        """Get data of each object location from the given image with provided function.
        mask = labeled mask of objects
        images = image set of spectral images
        function = e.g. np.median
        """
        if label is None:
            idx = np.arange(1, len(np.unique(mask)))
        else: 
            idx = label
        channel_no = images[:, 0, 0].size
        channels = xrange(channel_no)
        data = [ndi.labeled_comprehension(images[ch, :, :], mask, idx, function, float, -1) for ch in channels]
        return np.array(data)

    def get_per(labels, data):
        idx = np.arange(1, len(np.unique(labels)))
        size = len(idx)

    def median_per(labels, data):
        pass        

    def median_all():
        pass

    def mean_per():
        pass

    def mean_all():
        pass

