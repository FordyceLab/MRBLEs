# !/usr/bin/env python
# -*- coding: utf-8 -*-

# [Future imports]
from __future__ import print_function, division

# [File header]     | Copy and edit for each file in this project!
# title             : simp.py
# description       : MRBLEs - Simplified and condensed functions.
# author            : Bjorn Harink
# credits           :
# date              : 20170219

# [Modules]
# General Python
import sys
from math import sqrt
# Data Structure
import numpy as np
import pandas as pd
import xarray as xd
import weightedstats as ws
# Image Processing
from scipy import ndimage as ndi
# Graphs
from matplotlib import pyplot as plt
# Project
from mrbles.core import FindBeadsImaging
from mrbles.data import Spectra, ImageSetRead, DataOutput

# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    from future.standard_library import install_aliases
    from __builtin__ import *  # NOQA
    install_aliases()


# Classes

class Extract(object):
    """
    """
    pass

    def __init__(self):
        pass



class ReferenceSpectra(object):
    """Reference Spectra class.

    Parameters
    ----------


    """

    def __init__(self, files, object_channel, channels,
                 bead_size=16, dark_noise=0):
        """Instatiate Reference Spectra class."""
        super(ReferenceSpectra, self).__init__()
        self.files = files
        self.object_channel = object_channel
        if isinstance(channels, list) and (len(channels) == 2):
            self._channels = self.set_slice(channels)
        else:
            self._channels = channels
        self.bead_size = bead_size
        self.darknoise = dark_noise
        self.crop_x = None
        self.crop_y = None
        # Default bead find values
        self.thr_block = 33
        self.thr_c = 11
        self.ref_objects = FindBeadsImaging(bead_size, border_clear=True)
        # Dataframe
        self.ref_data = Spectra()

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
            raise ValueError(
                "Use slice(value, value) or [value, value] for range! Input: %s" % values)

    def get_spectra(self):
        """Get spectra."""
        # for name, file in self.files.iteritems():
        for name, file in self.files.items():
            print("Spectrum: %s" % name)
            img_obj = ImageSetRead(file)
            self.ref_objects.find(
                img_obj[self.object_channel, self._crop_y, self._crop_x])
            print("No beads:", self.ref_objects.bead_num)
            channels = img_obj[self._channels, self._crop_y, self._crop_x]
            if isinstance(self._channels, slice):
                channel_names = img_obj.c_names[np.where(img_obj.c_names ==
                                                         self._channels.start)[0][0]:
                                                np.where(img_obj.c_names ==
                                                         self._channels.stop)[0][0] + 1]
            elif isinstance(self._channels, list) and len(self._channels) > 2:
                channel_names = self._channels
            # data = self.get_spectrum(self.darknoise, channels, self.ref_objects.labeled_mask)
            data = self.get_spectrum(
                self.darknoise, channels, self.ref_objects.mask_inside)
            self.ref_data.spec_add(name, data=data, channels=channel_names)

    def set_back(self, file, channels, roi_x, roi_y):
        """Set background spectrum."""
        if isinstance(channels, list) and len(channels) == 2:
            channels = self.set_slice(channels)
        BACK_CROPx = self.set_slice(roi_x)
        BACK_CROPy = self.set_slice(roi_y)
        print("Spectrum Bkg: %s, %s" % (BACK_CROPx, BACK_CROPy))
        bkg_img_obj = ImageSetRead(file)
        ref_data_tmp = np.array(
            [np.median(ch) for ch in bkg_img_obj[channels, BACK_CROPy, BACK_CROPx]])
        ref_data_tmp /= ref_data_tmp.sum()  # Normalize, no dark noise subtraction
        plt.figure()
        plt.imshow(bkg_img_obj[self.object_channel,
                               BACK_CROPy, BACK_CROPx], cmap='Greys_r')
        self.ref_data.spec_add('Bkg', ref_data_tmp)

    @staticmethod
    def get_spectrum(dark_noise, channels, mask):
        """Get spectrum from image set using mask.

        The median of the masked area is extracted, the camera dark noise
        subtracted, and normalized.

        Parameters
        ----------
        dark_noise : int
            Intrinsic dark noise of camera. Image taken when shutter closed.
        channels : slice, list
            Slice of channels

        """
        # Dark noise is subtracted from the lanthanide spectra and then
        # normalized with the sum of the data.
        data = np.array([ndi.median(ch, mask) for ch in channels])
        data -= dark_noise  # Dark noise subtract
        data /= data.sum()  # Normalize
        return data

    # Inspect functions
    def imshow(self, name=''):
        fig = plt.figure()
        fig.suptitle("Overlay Image ref %s:" % name)
        img_overlay = FindBeadsImaging.overlay_image(ref_img_obj[self.object_channel,
                                                                 self._crop_y,
                                                                 self._crop_x],
                                                     dim=self.ref_objects.circles_dim)
        plt.imshow(img_overlay, cmap='Greys_r')
        plt.draw()

    def plot(self):
        self.ref_data_object.plot()


class Images(DataOutput):
    """Load images."""

    def __init__(self, folders, files, output='xr'):
        self.folders = folders
        self.files = files
        self.output = output
        self._dataframe = None

    def load(self):
        """Initilize loading of images."""
        file_sets = self._find_images(self.folders, self.files)
        if file_sets is None:
            return False
        dict_data = [ImageSetRead(file_set).xdata
                     for key, file_set in file_sets.items()]
        self._dataframe = xd.concat(dict_data,
                                    dim=pd.Index(file_sets.keys(), name='set'))
        #self._dataframe.rename({'set': file_sets.keys()})

    def _find_images(self, folders, files):
        if isinstance(folders, dict):
            files = {key: ImageSetRead.scan_path(folders[key], pattern)
                     for key, pattern in files.items()}
        elif isinstance(folders, str):
            files = {key: ImageSetRead.scan_path(folders, pattern)
                     for key, pattern in files.items()}
        else:
            return None
        return files


class Find(object):
    """Find MRBLEs in brightfield images."""

    def __init__(self, bead_size, border_clear=True):
        """Initialize."""
        self._bead_objects = FindBeadsImaging(
            bead_size, border_clear=border_clear)
        self.masks = None
        self.bead_data = None

    def find(self, object_images):
        """Execute finding images."""
        if isinstance(object_images, xd.DataArray):
            self.masks, self.bead_data = self.get_masks(
                object_images.values, self._bead_objects)
        else:
            self.masks, self.bead_data = self.get_masks(
                object_images, self._bead_objects)

    # def _multi(self, files):
    #    if isinstance(files, )

    @staticmethod
    def get_masks(object_images, bead_finder):
        masks = []
        circles_dim = []
        for idx, image in enumerate(object_images):
            bead_finder.find(image)
            masks.append(xd.DataArray([bead_finder.mask_bead,
                                       bead_finder.mask_inside,
                                       bead_finder.mask_ring,
                                       bead_finder.mask_outside,
                                       bead_finder.mask_bkg],
                                      dims=['c', 'y', 'x'],
                                      coords={'c': ['mask_full', 'mask_inside', 'mask_ring', 'mask_outside', 'mask_bkg']}))
            dim = pd.DataFrame(bead_finder.bead_dims_bead)
            dim.insert(0, 'image', idx)
            circles_dim.append(dim)
        bead_data = pd.concat(circles_dim, ignore_index=True)[
            circles_dim[0].columns]
        masks = xd.concat(masks, 'f')
        return masks, bead_data

    def combine(self, data):
        return xd.concat([self.masks, data], 'c')


class Decode(object):
    def __init__(self, spectra, target, assay_channels=None):
        self._spectra = spectra
        self._target = target
        self._assay_channels = assay_channels
        self._result = None

    def icp(self):
        icp = ICP(matrix_method='std', max_iter=100,
                  tol=1e-4, outlier_pct=0.01, train=False)
        icp.fit(bead_set.loc[filter_all,
                             ('rat_dy', 'rat_sm', 'rat_tm')], target)
        bead_set = bead_set.join(icp.transform())
        print("Tranformation matrix: ", icp.matrix)
        print("Offset vector: ", icp.offset)

    def gmm(self):
        gmix = Classify(target, tol=1e-5, min_covar=1e-7,
                        sigma=1e-5, train=False)
        gmix.decode(bead_set.loc[filter_all,
                                 ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')])
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


class Analyze():
    """Analyze data."""
    pass


def get_stats_per_channel_and_code(data, channels, codes=None):
    bead_sets = []
    for channel in channels:
        bead_sets.append(get_stats_per_code(data, channel, codes))
    final_bead_set = pd.concat(bead_sets, keys=channels)
    return final_bead_set


def get_stats_per_code(data, channel, norm=None, norm_channel=None, codes=None):
    """Get statistical values for each code in you data set.

    Parameters
    ----------
    data : Pandas DataFrame
        This is the per bead data.
    channel : string
        This is the channel (or column) to extract statistical values from.
    codes : int, list
        This value can be set if only a select (list) or one code (int) needs to be processed.
        If value is not set the unique codes from data, the Pandas DataFrame is used.
        Defaults to None.

    Returns
    -------
    bead_set : Pandas DataFrame
        This dataframe contains: AVG, SD, N, CV, SEM, and RSEM for each code.
        Codes start at 0. Code -1 represents weighted statistical values over all codes.

    """
    data_stats = []
    data_stats_norm = []
    n_codes = []
    if type(codes) is list:
        codes_set = codes
    elif (type(data) is pd.DataFrame) and (codes is None):
        codes_set = np.unique(data.code[data.code.notnull()])
    else:
        codes_set = range(codes)
    for code in codes_set:
        n_codes.append(code)
        data_code = data.loc[data.code == code, (channel)]
        stats = get_stats(data_code)
        data_stats.append(stats)
        if norm is not None:
            if norm_channel is not None:
                norm_code = norm.loc[norm.code == code, (norm_channel)]
            else:
                norm_code = norm.loc[norm.code == code, (channel)]
            stats_norm = get_stats_norm(data_code, norm_code)
            data_stats_norm.append(stats_norm)
    # n_codes.append(-1)
    # data_stats.append(get_weighted_stats(data_stats))
    final_codes = pd.DataFrame(n_codes, columns=['code'])
    final_stats = pd.DataFrame(
        data_stats, columns=['AVG', 'SD', 'N', 'CV', 'SEM', 'RSEM', 'MED'])
    if norm is not None:
        # data_stats_norm.append(get_weighted_stats(data_stats_norm))
        final_stats_norm = pd.DataFrame(data_stats_norm, columns=[
                                        'AVG_NORM', 'SD_NORM', 'N_NORM', 'CV_NORM', 'SEM_NORM', 'RSEM_NORM', 'MED_NORM'])
    bead_set = final_codes.join(final_stats)
    if norm is not None:
        bead_set = bead_set.join(final_stats_norm)
    return bead_set


def get_stats_norm(data, norm, scale_norm=True):
    if scale_norm is True:
        norm /= norm.max()
    norm_mean = np.nanmean(norm)
    norm_sd = np.nanstd(norm)
    data_mean = np.nanmean(data)
    data_sd = np.nanstd(data)

    mean = data_mean / norm_mean
    sd = np.abs(mean) * (np.sqrt((data_sd / data_mean) ** 2 +
                                 (norm_sd / norm_mean)**2))
    n = len(data)
    cv = sd / mean
    sem = sd / sqrt(n)
    rsem = sem / mean
    median = np.nanmedian(data / norm_mean)
    return np.array([mean, sd, n, cv, sem, rsem, median])


def get_stats(data_array):
    """Get statistics from data.

    Parameters
    ----------
    data_array :

    Returns
    -------

    """
    data_mean = np.nanmean(data_array)
    data_median = np.nanmedian(data_array)
    data_sd = np.nanstd(data_array)
    data_n = len(data_array)
    data_cv = data_sd / data_mean
    data_sem = data_sd / sqrt(data_n)
    data_rsem = data_sem / data_mean
    stats_array = np.array([data_mean,
                            data_sd,
                            data_n,
                            data_cv,
                            data_sem,
                            data_rsem,
                            data_median])
    return stats_array


def get_weighted_stats(data_array):
    """Get weighted statistics.

    Parameters
    ----------
    data_array : NumPy array, list
        Array with data to return statistics from.

    Returns
    -------
    NumPy array
        Returns array with weighted stats over

    """
    data_mean = np.average(data_array[:, 0], weights=data_array[:, 2])
    data_sd = sqrt(np.average(
        (data_array[:, 0] - data_mean) ** 2, weights=data_array[:, 2]))
    data_n = np.sum(data_array[:, 2])
    data_cv = data_sd / data_mean
    data_sem = data_sd / sqrt(data_n)
    data_rsem = data_sem / data_mean
    data_median = ws.weighted_median(np.nan_to_num(
        data_array[:, 0]), weights=np.nan_to_num(data_array[:, 2]))
    return np.array([data_mean, data_sd, data_n, data_cv, data_sem, data_rsem, data_median])
