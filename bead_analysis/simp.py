# !/usr/bin/env python

# [Future imports]
# Function compatibility between Python 2.x and 3.x
from __future__ import print_function, division
from future.standard_library import install_aliases
install_aliases()
import sys
if sys.version_info < (3,0): from __builtin__ import *

# [File header]     | Copy and edit for each file in this project!
# title             : simp.py
# description       : Simplified and condensed function.
# author            : Bjorn Harink
# credits           : 
# date              : 20170219
# version update    : 20170219
# version           : v0.1
# usage             : As module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [Modules]
# General Python
import os
import types
import warnings
from math import sqrt
# Data Structure
import numpy as np
import pandas as pd
# Image Processing
from scipy import ndimage as ndi
# Graphs
from matplotlib import pyplot as plt
# Project
from bead_analysis.core import *
from bead_analysis.data import *


class ReferenceSpectra(object):
    def __init__(self, files, object_channel, channels, find_param, dark_noise=0):
        super(ReferenceSpectra, self).__init__()
        self.files = files
        self.object_channel = object_channel
        if (type(channels) is list) and (len(channels) == 2):
            self._channels = self.set_slice(channels)
        else:
            self._channels = channels
        self.find_param = find_param
        self.darknoise = dark_noise
        self.crop_x = None
        self.crop_y = None

        self.ref_data = Spectra()
        self.ref_objects = FindBeadsCircle(min_r=find_param[0], max_r=find_param[1], param_1=find_param[2], param_2=find_param[3])

    @property
    def output(self):
        return self.ref_data

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

    @staticmethod
    def set_slice(values):
        if type(values) is slice or values is None:
            return values
        elif type(values) is list:
            return slice(values[0], values[1])
        else:
            raise ValueError("Use slice(value, value) or [value, value] for range! Input: %s" % values)

    def get_spectra(self):
        #for name, file in self.files.iteritems():
        for name, file in self.files.items():
            print("Spectrum: %s" % name)
            img_obj = ImageSetRead(file)
            self.ref_objects.find(img_obj[self.object_channel, self._crop_y, self._crop_x])
            channels = img_obj[self._channels, self._crop_y, self._crop_x]
            if type(self._channels) is slice:
                channel_names = img_obj.c_names[np.where(img_obj.c_names == self._channels.start)[0][0] :
                                                np.where(img_obj.c_names == self._channels.stop)[0][0]+1]
            elif type(self._channels) is list and len(self._channels) > 2:
                channel_names = self._channels
            data = self.get_spectrum(self.darknoise, channels, self.ref_objects.labeled_mask)
            self.ref_data.spec_add(name, data = data, channels = channel_names)

    def set_back(self, file, channels, roi_x, roi_y):
        if (type(channels) is list) and (len(channels) == 2):
            channels = self.set_slice(channels)
        BACK_CROPx = self.set_slice(roi_x)
        BACK_CROPy = self.set_slice(roi_y)
        print("Spectrum Bkg: %s, %s" % (BACK_CROPx, BACK_CROPy))
        bkg_img_obj = ImageSetRead(file)
        ref_data_tmp = np.array([np.median(ch) for ch in bkg_img_obj[channels,BACK_CROPy,BACK_CROPx]])
        ref_data_tmp /= ref_data_tmp.sum()  # Normalize, no dark noise subtraction
        self.ref_data.spec_add('Bkg', ref_data_tmp)
    
    @staticmethod
    def get_spectrum(dark_noise, channels, mask):
        """Get spectrum from image set using mask

        The median of the masked area is extracted, the camera dark noise subtracted, and normalized.

        Parameters
        ----------
        dark_noise : int
            Intrinsic dark noise of camera. Image taken when shutter closed.

        channels : slice, list
            Slice of cha
        """
        # Dark noise is subtracted from the lanthanide spectra and then normalized with the sum of the data.
        data = np.array([ndi.median(ch, mask) for ch in channels])
        data -= dark_noise  # Dark noise subtract
        data /= data.sum()  # Normalize
        return data
    
    # Inspect functions
    def imshow(self, name=''):
        fig = plt.figure()
        fig.suptitle("Overlay Image ref %s:" % name)
        plt.imshow(ref_objects.overlay_image(ref_img_obj['Brightfield',CROPy_ref,CROPx_ref], dim = ref_objects.circles_dim), cmap='Greys_r')
        plt.draw()

    def plot(self):
        self.ref_data_object.plot()

class BeadDecode(object):
    def __init__(self, spectra, target, assay_channels=None):
        self._spectra = spectra
        self._target = target
        self._assay_channels = assay_channels
        self._result = None

    def icp(self):
        icp=ICP(matrix_method='std', max_iter=100, tol=1e-4, outlier_pct=0.01, train=False)
        icp.fit(bead_set.loc[filter_all, ('rat_dy', 'rat_sm', 'rat_tm')], target)
        bead_set = bead_set.join(icp.transform())
        print("Tranformation matrix: ", icp.matrix)
        print("Offset vector: ", icp.offset)

    def gmm(self):
        gmix = Classify(target, tol=1e-5, min_covar=1e-7, sigma=1e-5, train=False)
        gmix.decode(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')])
        bead_set = bead_set.join(gmix.output)
        print("Number of unique codes found:", gmix.found)
        print("Missing codes:", gmix.missing)

    def decode(self):
        self.icp()
        self.gmm()
        if self._assay_channels is not None:
            pass

    @property
    def result(self):
        return self._result

    @staticmethod
    def assay(image, mask, method=np.median):
        idx = np.arange(1, len(np.unique(mask)))
        data = ndi.labeled_comprehension(image, mask, idx, method, float, -1)
        return data

    @classmethod
    def assays(cls, images, mask, method=np.median):
        data = []
        for image in images:
            data.append(cls.assay(image, mask, method=np.median))
        return data
