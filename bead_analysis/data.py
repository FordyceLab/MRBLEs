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
# version update    : 20160808
# version           : v0.2
# usage             : As part of Bead Analysis module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General
import sys
import os
import warnings
import fnmatch
# Math
import numpy as np
from skimage.external import tifffile as tff
import pandas as pd
# Graphs
import matplotlib.pyplot as plt

# TO-DO
# Check error exceptions
# Create error checking functions for clustering

# Main functions and classes
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
    def _data_out(self, func, output=None):
            """Checks data output method.
            """
            if output is None:
                output = self.output
            if output is "pd":
                return func
            elif output is "nd":
                return func.values
            else:
                raise ValueError("Unspecified output method: '%s'." % output)


class Spectra(PropEdit, FrozenClass, OutputMethod):
    """Spectra
    Data structure for reference spectra
    """

    def __init__(self, data=None, spectra=None, channels=None, output='nd'):
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
        """Return spectra index.
        """
        return self._dataframe.columns.get_loc(name)

    def spec_get(self, index, output=None):
        """Return channel data by name or number.

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
        if self.c_names != channels:
            warnings.warn("Channel names not the same!")
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
        data = self._dataframe.ix[index]
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

    def plot(self, show=False):
        self._dataframe.plot(title="Spectra", rot=90)
        if show is True:
            plt.show()


class ChannelDescriptor(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return obj.spec_get(self.name)

    def __set__(self, obj, val):
        obj._dataframe[self.name] = val

# TO-DO
# - Add mask for CROP
class ImageSetRead(FrozenClass, OutputMethod):
    """Image set data object that loads image set from file(s).

    Parameters
    ----------
    file_path : string/list [string, string, ...]
        File path as string, e.g. 'C:/folder/file.tif', or as list of file paths, e.g. ['C:/folder/file.tif', 'C:/folder/file.tif'].

    series : int, optional
        Sets the series number if file(s) has/have multiple series.
        Defaults to 0.

    output : str, optional
        Sets default output method. Options: 'nd', NumPy ndarray, or 'pd', Pandas Panel4D.
        Defaults to 'ndarray'

    Attributes
    ----------
    See function descriptions.

    Returns
    -------
    _dataframe : Pandas dataframe
        Returned when calling the instance.
    _dataframe[idx] : NumPy ndarray
        Returns the index value or slice values: [start:stop:stride]. Warning: when using column names stop values are included, e.g. inst[

    Examples
    --------
    >>> image_data_object = ImageSetRead('C:/folder/file.tif')
    >>> image_files = ['C:/folder/file.tif', 'C:/folder/file.tif']
    >>> image_data_object = ImageSetRead(image_files, output='pd')
    >>> image_data['BF', 100:400, 100:400]
    (301L, 301L)
    >>> image_data['BF', 100:400, 100:400]
    (301L, 301L)
    """

    def __init__(self, file_path, series=0, output='nd'):
        self._dataframe, self._metadata, self._files = \
            self.load(file_path, series)
        self.output = output
        self._freeze()

    def __repr__(self):
        """Returns Pandas dataframe representation.
        """
        return repr([self._dataframe])

    def __getitem__(self, index):
        """Get method, see method 'spec_get'.
        """
        return self.c_get(index) 

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

    # Channel properties and methods
    @property
    def c_size(self):
        """Return channel count.
        """
        return self._dataframe.items.size

    @property
    def c_names(self):
        """"Return channel names.
        """
        return self._dataframe.items.tolist()

    def c_index(self, name):
        """Return channel number.
        """
        return self.c_names.get_loc(name)

    def c_get(self, index, output=None):
        """Return channel data by name or number.

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
        >>> image_data_object.c_get('BF')
        [[ 522,  583,  584, ...,  614,  613,  385],
        [ 551,  599,  615, ...,  633,  637,  397],
        [ 573,  613,  611, ...,  619,  625,  398],
        ..., 
        [1006, 1170, 1152, ..., 1201, 1145,  611],
        [ 986, 1178, 1173, ..., 1185, 1148,  609],
        [1003, 1192, 1201, ..., 1156, 1172,  605]]], dtype=uint16)
        >>> image_data_object.c_get('BF', output='pd')
        <class 'pandas.core.panel.Panel'>
        Dimensions: 161 (items) x 501 (major_axis) x 502 (minor_axis)
        Items axis: 0 to 160
        Major_axis axis: 0 to 500
        Minor_axis axis: 0 to 501
        """
        #if type(index) is int:
        #    index = self.c_names[index]
        data = self._dataframe.ix[index]
        return self._data_out(data, output)
        #return self._dataframe.ix[index]

    # Position properties and methods
    @property
    def p_size(self):
        """Return position count.
        """
        return len(self._metadata['summary']['Positions'])

    # Z-slice properties and methods
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
        if self.is_multi_file:
            return ('F' + self._metadata['series'][0]['axes'])
        else:
            return self._metadata['series'][0]['axes']
    
    # Data properties and methods
    @property
    def ndata(self):
        """Return Pandas dataframe as NumPy ndarray.
        """
        return self._dataframe.as_matrix()

    @property
    def pdata(self):
        """Return Pandas dataframe.
        """
        return self._dataframe

    @classmethod
    def load(cls, file_path, series=0):
        """Load image files into data structures.

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
                    image_data = pd.Panel4D(ts.asarray(series=series), 
                                            items=image_metadata['summary']['ChNames'])
            except ValueError:
                print("Not all images are the same shape: '%s'" %
                      sys.exc_info()[1])
        elif type(file_path) is str:
            with tff.TiffFile(file_path) as tf:
                files = [tf.filename]
                image_metadata = cls._get_metadata(tf)
                image_data = pd.Panel(tf.asarray(series=series), 
                                      items=image_metadata['summary']['ChNames'])
        else:
            raise TypeError(
                "'file_path' is not of type 'str' or 'list' %s" % type(file_path))
        return image_data, image_metadata, files

    @staticmethod
    def scan_path(path, pattern="*.tif"):
        """Scan folder recursively for files matching the pattern.

        Parameters
        ----------
        path : string
            Folder path as string, e.g. 'C:/folder/file.tif'.

        pattern : string
            File extenstion of general file pattern as search string, e.g. '20160728_MOL_*'
            Defaults to '*.tif'.
        """
        image_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
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
            raise ValueError("Not a Micro Manager TIFF file.")


class BeadSet(object):
    """Bead Set
    Per-set data object"""

    def __init__(self):
        self._dataframe = pd.DataFrame(columns=('img','lbl','code','R','Y','X'))

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
    def code(self):
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

    def get_median(labels, images):
        """Get Median Intensities of each object location from the given image.
        labels = Labeled mask of objects
        images = image set of spectral images
        """
        idx = np.arange(1, len(np.unique(labels)))
        data_size = len(np.unique(labels)) - 1
        channel_no = images[:, 0, 0].size
        channels = xrange(channel_no)
        medians_data = np.empty((data_size, channel_no))
        for ch in channels:
            # Get median value of each object
            medians_data[:, ch] = ndi.labeled_comprehension(
                images[ch, :, :], labels, idx, np.median, float, -1)
        return medians_data

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

