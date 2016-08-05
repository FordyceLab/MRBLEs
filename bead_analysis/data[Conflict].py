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

# Math
import numpy as np

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
    """Frozen Class
    Freeze or thaw right to add attributes to instance.
    """
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True
    def _thaw(self):
        self.__isfrozen = False

class ChannelDescriptor(object):
    """Channel Descriptor
    """
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return obj._data[self.name]

    def __set__(self, obj, val):
        if len(obj._data.dtype.names) == 1:
            obj._data = np.array(np.array(val), dtype=[(self.name, 'float64')])
        elif int(obj._data.shape[0]) == len(val):
            obj._data[self.name] = np.array(val, dtype='float64')
        else:
            raise IndexError("%s wrong size, must be %i" % (val, obj._data.shape[0]))

class Spectra(PropEdit, FrozenClass):
    """Spectra
    Data structure for reference spectra
    """
    def __init__(self, data=None, channels=None):
        self._init_data = data
        self._data = None
        if channels is None:
            channels = []
        if data is None:
            data = []
        if len(data) == 0 and len(channels) > 0:
            data = np.zeros((len(channels),3))
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
        if data is None:
            data = np.zeros(self._data.shape)
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