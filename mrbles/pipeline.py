# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline Classes and functions
==============================

This files contains the pipeline for the MRBLEs analysis.

Classes
-------
References

Images

Find

Ratio

Extract

Analyze

"""

# [Future imports]
from __future__ import (absolute_import, division, print_function)
from builtins import (str, super, range, int, object)

# [File header]     | Copy and edit for each file in this project!
# title             : simp.py
# description       : MRBLEs - Simplified and condensed functions.
# author            : Bjorn Harink
# credits           :
# date              : 20170219

# [Modules]
# General Python
import os
import re
# import sys
import gc
from math import sqrt
from random import randint

# Other
import numpy as np
import scipy as sp
from scipy import ndimage as ndi
from sklearn.metrics import silhouette_score
import pandas as pd
import weightedstats as ws
import xarray as xr
from matplotlib import pyplot as plt

# Intra-Package dependencies
from mrbles.core import FindBeadsImaging, ICP, Classify, SpectralUnmixing
from mrbles.data import ImageSetRead, ImageDataFrame, TableDataFrame


# General methods


class Settings(object):
    """Settings object."""

    def __init__(self, objects, object_names):
        """Set attributes for given objects."""
        for idx, obj in enumerate(objects):
            setattr(self, object_names[idx], obj)


# Classes


class References(TableDataFrame):
    """Create reference spectra."""

    def __init__(self, folders, files, object_channel, reference_channels,
                 bead_size=16, dark_noise=99, background='bkg', clean_up=True):
        """Init."""
        super(References, self).__init__()
        self.object_channel = object_channel
        self.reference_channels = reference_channels
        self.dark_noise = dark_noise
        self.background = background
        self.clean_up = clean_up
        self.bkg_roi = [slice(None), slice(None)]
        self._images = Images(folders, files)
        self._find = Find(bead_size=bead_size,
                          border_clear=True,
                          circle_size=None)
        self._dataframe = None
        self.crop_x = slice(None)
        self.crop_y = slice(None)
        # Settings options
        self.settings = Settings([self._images, self._find], ['icp', 'gmm'])
        self.settings.__doc__ = """Return Images and Find objects for settings purposes.

        Attributes
        ----------
        images : Images() object
            Returns Images() object for settings puproses.
            See class documentation for more information.
        find : Find() object
            Returns Find() object for settings puproses.
            See class documentation for more information.

        """

    def load(self):
        """Process."""
        self._images.load()
        spectra = list(self._images.data.keys())
        bkg_images = self._images[self.background, self.reference_channels,
                                  self.bkg_roi[0], self.bkg_roi[1]]
        self._bkg_image = self._images[self.background, self.object_channel,
                                       self.bkg_roi[0], self.bkg_roi[1]]
        spec_images = ImageDataFrame(self._images.data)
        spec_images._dataframe.pop(self.background, None)
        spec_images.crop_x = self.crop_x
        spec_images.crop_y = self.crop_y
        self._find.find(spec_images[:, self.object_channel])
        ref_channels = self._images[
            spectra[0], self.reference_channels].c.values
        data = [self.get_spectrum(self.dark_noise,
                                  spec_images[x_set, self.reference_channels],
                                  self._find[x_set, 'mask_inside'])
                for x_set in spectra if x_set != self.background]
        if self.background in spectra:
            spectra.remove(self.background)
            spectra.append(self.background)
            data.append(self._get_back(bkg_images))
        self._dataframe = pd.DataFrame(data=np.array(data).T,
                                       columns=spectra,
                                       index=ref_channels)
        self._dataframe.index.name = 'channels'
        if self.clean_up is True:
            self._clean_up()

    def _clean_up(self):
        del self._images
        del self._find
        gc.collect()

    def _get_back(self, bkg_images):
        mask = np.ones((bkg_images.y.size,
                        bkg_images.x.size))
        bkg_data = self.get_spectrum(0, bkg_images, mask)
        return bkg_data

    def plot(self, dpi=75):
        """Plot Reference spectra."""
        self._dataframe.plot()
        plt.figure(dpi=dpi)
        plt.title('Background slice')
        plt.imshow(self._bkg_image)

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
        mask : NumPy array
            Labeled mask.

        """
        data = np.array([ndi.median(ch, mask) for ch in channels])
        data -= dark_noise  # Dark noise subtract
        data /= data.sum()  # Normalize
        return data


class Images(ImageDataFrame):
    """Load OME-TIF images.

    Super-Class wrapper for mrbles.data.ImageSetRead.
    Please see this class documentation for more information.

    Parameters
    ----------
    folders : str, dict
        String of single folder or dict of multiple folders.
        Dict keys must match file_patterns dict.
    file_patterns : dict
        Dict of multiple file patterns.
        Dict keys must match folders keys, if multiple folders.
    output : str, optional
        Sets default output method. Options: 'np' for NumPy ndarray or 'xr'
        for xarray.
        Defaults to 'xr'.

    """

    def __init__(self, folders, file_patterns):
        """Instatiate object and search for images."""
        super(Images, self).__init__()
        self.folders = folders
        self.file_patterns = file_patterns
        self._dataframe = None
        self.files = self._find_images(self.folders, self.file_patterns)

    def load(self):
        """Load images in memory."""
        if self.files is None:
            return False
        self._dataframe = {key: ImageSetRead(file_set).xdata
                           for key, file_set in self.files.items()}

    def add_images(self, images):
        """Add images to dataframe."""
        self.combine(images)
        gc.collect()

    def rename_channel(self, old_name, new_name):
        """Rename channel name."""
        if isinstance(self._dataframe, dict):
            for key, data_array in self._dataframe.items():
                channels = data_array.coords['c'].values
                if old_name in channels:
                    self._dataframe[key] = self.rename_coord(data_array,
                                                             'c',
                                                             old_name,
                                                             new_name)
        else:
            self._dataframe = self.rename_coord(self._dataframe,
                                                'c',
                                                old_name,
                                                new_name)

    @staticmethod
    def rename_coord(dataframe, dim, old_name, new_name):
        """Rename dimension coordinate in an Xarray DataArray."""
        coordinates = dataframe.coords[dim].values
        np.place(coordinates, coordinates == old_name, new_name)
        renamed_dataframe = xr.DataArray(dataframe,
                                         dims=dataframe.dims,
                                         coords={'c': coordinates})
        return renamed_dataframe

    # Private methods
    @classmethod
    def _find_images(cls, folders, files):
        if isinstance(folders, dict):
            files = {key: cls.scan_path(folders[key], pattern)
                     for key, pattern in files.items()}
        elif isinstance(folders, str):
            files = {key: cls.scan_path(folders, pattern)
                     for key, pattern in files.items()}
        else:
            return None
        return files

    # Static methods
    @staticmethod
    def scan_path(path, pattern="*.tif"):
        """Scan folder recursively for files matching the pattern.

        Parameters
        ----------
        path : string
            Folder path as string, e.g. r'C:/folder/file.tif'.
        pattern : string
            File extenstion of general file pattern as search string,
            e.g. 'peptides_x_([0-9][0-9])_MMStack_Pos0.ome.tif'
            Defaults to '*.tif'.

        """
        image_files = []
        reg_object = re.compile(pattern)
        for root, _, files in os.walk(path):
            file_list = [os.path.join(root, x)
                         for x in files if reg_object.match(x)]
            if file_list:
                image_files.append(file_list)
        return np.hstack(image_files).tolist()


class Find(ImageDataFrame):
    """Find MRBLEs in brightfield images.

    Super-Class wrapper for mrbles.core.FindBeadsImaging.
    Please see this class documentation for more information.

    Parameters
    ----------
    bead_size : int
        Approximate width of beads (circles) in pixels.
    border_clear : boolean
        Beads touching border or ROI will be removed.
        Defaults to True.
    circle_size : int
        Set circle size for auto find circular ROI.

    """

    def __init__(self, bead_size,
                 border_clear=True, circle_size=None):
        """Initialize."""
        super(Find, self).__init__()
        self._bead_size = bead_size
        self._bead_objects = FindBeadsImaging(bead_size=bead_size,
                                              border_clear=border_clear,
                                              circle_size=circle_size)
        self._dataframe = None
        self._bead_dims = None

    def find(self, object_images, combine_data=None):
        """Execute finding images."""
        if isinstance(object_images, dict):
            self._dataframe, self._bead_dims = \
                self._find_multi_set(object_images)
        else:
            self._dataframe, self._bead_dims = \
                self._return_data(object_images)
        beads_radius_mean = self.bead_dims.radius.mean() * 2
        print("Bead radius AVG: %0.2f" % (beads_radius_mean))
        beads_radius_sd = self.bead_dims.radius.std()
        print("Bead radius SD: %0.2f" % (beads_radius_sd))
        beads_radius_cv = (beads_radius_sd / beads_radius_mean) * 100
        print("Bead radius CV: %0.2f%%" % (beads_radius_cv))
        if self.beads_per_set is not None:
            for key, value in self.beads_per_set.items():
                print("Number of beads in set %s: %i" % (key, value))
        print("Total number of beads: %i" % self.beads_total)
        if combine_data is not None:
            self.combine(combine_data)

    @property
    def masks(self):
        """Return masks."""
        return self.data

    @property
    def bead_dims(self):
        """Return MRBLEs dimensions."""
        return self._bead_dims

    @property
    def beads_total(self):
        """Return total bead count in set(s)."""
        return self._bead_dims.shape[0]

    @property
    def beads_per_set(self):
        """Return total bead count in set(s)."""
        if self.sets is not None:
            bead_no_list = {set_x: self._bead_dims.loc[set_x].shape[0]
                            for set_x in self.sets}
        else:
            bead_no_list = None
        return bead_no_list

    @property
    def sets(self):
        """Return list of sets."""
        return TableDataFrame.get_set_names(self._bead_dims)

    def inspect(self, set_name=None, fig_num=3):
        """Display random images from set for inspection."""
        if not isinstance(self._dataframe, dict):
            img_num = self._dataframe.f.size
            self._inspect(self._dataframe, img_num, fig_num)
        elif set_name is None:
            img_num = {key: data.f.size
                       for key, data in self._dataframe.items()}
            for key, data in self._dataframe.items():
                self._inspect(data, img_num[key], fig_num)
                plt.suptitle(key)
        else:
            self._inspect(self._dataframe[set_name], img_num, fig_num)
            plt.suptitle(set_name)

    @staticmethod
    def _inspect(data, img_num, fig_num):
        plt.figure()
        for plt_num in range(1, fig_num + 1):
            rand_fig = randint(0, img_num)
            plt.subplot(1, fig_num, plt_num)
            plt.imshow(data.loc[rand_fig])
            plt.title('Figure #%s' % rand_fig)

    def _find_multi_set(self, object_image_sets):
        sets = list(object_image_sets.keys())
        data = [self._return_data(image_set)
                for key, image_set in object_image_sets.items()]
        data_masks = [i[0] for i in data]
        data_dims = [i[1] for i in data]
        result_masks = {
            sets[idx]: value for idx, value in enumerate(data_masks)
        }
        result_dims = pd.concat(data_dims,
                                keys=sets,
                                names=['set'])
        return result_masks, result_dims

    def _return_data(self, object_images):
        self._bead_objects.find(object_images)
        dataframe = self._bead_objects.data
        bead_dims = self._bead_objects.bead_dims
        return [dataframe, bead_dims]

    @property
    def settings(self):
        """Return FindBeadsImaging object for settings purposes.

        See FindBeadsImaging documentation for detailed settings.

        Examples
        --------
        >>> find_object = Find(bead_size=18)
        >>> find_object.settings.area_min = 20
        >>> find_object.settings.area_max = 350
        >>> find_object.settings.eccen_max = 0.65
        >>> find_object.settings.circle_size = 350

        """
        return self._bead_objects


class Ratio(ImageDataFrame):
    """Generate spectrally unmix ratio images.

    Parameters
    ----------
    reference_spectra : list, ndarray, References object
        List, array, or References object of reference spectra for linear
        spectral unmixing.
    background : string
        Background key label.
        Defaults to 'bkg'.

    """

    def __init__(self, reference_spectra, background='bkg'):
        """Initialize reference spectra, unmixing, and background."""
        super(Ratio, self).__init__()
        self.reference_spectra = reference_spectra
        self.background = background
        self.spec_unmix = SpectralUnmixing(reference_spectra)

    def get(self, image_sets, reference, combine_data=None):
        """Calculate unmixed and ratio images.

        Parameters
        ----------
        image_set : Xarray DataArray, mblres ImageDataFrame
            These are the images (emission channels) to be unmixed.
        reference : string
            Reference key label, e.g. 'Eu', used for ratio images.
        comb_data : Xarray DataArray, mblres ImageDataFrame
            These are the images that are combined with 'image_set'.
            Must be same dimensions and type as 'image_set'.

        """
        if isinstance(image_sets, dict):
            self._dataframe = self._find_multi_set(image_sets, reference)
        else:
            self._dataframe = self._return_data(image_sets, reference)
        if combine_data is not None:
            self.combine(combine_data)

    def _find_multi_set(self, image_sets, reference):
        sets = list(image_sets.keys())
        data = [self._return_data(image_sets[s], reference)
                for s in sets]
        result = {
            sets[idx]: value for idx, value in enumerate(data)
        }
        return result

    def _return_data(self, images, reference):
        self.spec_unmix.unmix(images)
        unmix_images = self.spec_unmix.data
        sets = ImageDataFrame.get_set_names(unmix_images, set_dim='c')
        sets.remove(reference)
        if self.background in sets:
            sets.remove(self.background)
        ratio_images = unmix_images.loc[dict(c=sets)] / \
            unmix_images.loc[dict(c=reference)]
        ratio_images.coords['c'] = [s + '_ratio' for s in sets]
        result = xr.concat([unmix_images, ratio_images], dim='c')
        return result


class Extract(TableDataFrame):
    """Extract data from images using masks.

    Parameters
    ----------
    function : function
        This is the function used to extract data from each bead region. This
        can be replaced with any 1 parameter function, e.g. np.mean.
        Defaults to np.median.

    """

    def __init__(self, function=None):
        """Initialize function and variables."""
        super(Extract, self).__init__()
        if function is None:
            self._func = np.median
        else:
            self._func = function
        self._dataframe = None
        # self.flag_name = 'flag'
        self.flag_filt = True

    def get(self, images, masks):
        """Extract data from images using masks.

        Parameters
        ----------
        images : Xarray DataArray, mrbles ImageDataFrame
            Images of which to take values from using provided labeled masks.
            Must be same number of dimensions as masks.
        masks : Xarray DataArray, mrbles ImageDataFrame
            Labeled mask of regions to take values from the provided images.
            Must be same number of dimensions as images, even if one dimension
            is only singular. Therefore, use [] around mask selection, even
            when only using a single channel: ['inside'].

        Examples
        --------
        >>> per_mrble_data = mrbles.Extract(np.median)
        >>> per_mrble_data.run(images['100  nM', :, ['Cy5', 'Cy3']],
                               masks['100 nM', :, ['ring']])
        >>> per_mrble_data.data

        """
        if isinstance(images, xr.DataArray):
            if images.ndim == 4:
                f_list = ImageDataFrame.get_set_names(
                    images, set_dim=images.dims[0])
                data = [self._get_data_images(images.loc[f], masks.loc[f])
                        for f in f_list]
                self._dataframe = pd.concat(data, keys=f_list)
            else:
                self._dataframe = pd.DataFrame(
                    self._get_data_images(images, masks))
        else:
            data_append = []
            s_list = list(images.keys())
            for set_x in s_list:
                f_list = ImageDataFrame.get_set_names(
                    images[set_x], set_dim=images[set_x].dims[0])
                data = [self._get_data_images(images[set_x][f],
                                              masks[set_x][f])
                        for f in f_list]
                data_append.append(pd.concat(data, keys=f_list))
            self._dataframe = pd.concat(data_append, keys=s_list)
        self._dataframe[self.flag_name] = False

    def _get_data_images(self, images, masks):
        data = {
            str(image[images.dims[0]].values):
            self._get_data_masks(image, masks, self._func)
            for image in images
        }
        data_flatten = self.flatten_dict(data, prefix='.')
        dataframe = pd.DataFrame.from_dict(data_flatten, orient='columns')
        return dataframe

    @classmethod
    def _get_data_masks(cls, image, masks, func):
        if masks.ndim == 3:
            idx = cls._get_index(masks[0])
        else:
            idx = cls._get_index(masks)
        if idx is None:
            data = [None]
        else:
            if masks.ndim == 3:
                data = {
                    str(mask[masks.dims[0]].values):
                    cls._get_data(image, mask, func, idx)
                    for mask in masks
                }
            else:
                data = {
                    str(masks[masks.dims[0]].values):
                    cls._get_data(image, masks, func, idx)
                }
            data['mask_lbl'] = idx
        return data

    @classmethod
    def _get_data(cls, image, mask, func, idx):
        # idx = cls._get_index(mask)
        if idx is None:
            data = [None]
        else:
            data = ndi.labeled_comprehension(image,
                                             mask,
                                             idx,
                                             func,
                                             float,
                                             None)
        return data

    @staticmethod
    def _get_index(mask):
        if np.count_nonzero(mask) == 0:
            num = None
        else:
            num = np.unique(mask.values[mask.values > 0])
        return num

    def filter(self, bkg_factor=2.0, ref_factor=2.0,
               bkg='bkg.mask_full', ref='Eu.mask_inside'):
        """Filter and flag based on unmixed background and reference signal.

        Parameters
        ----------
        bkg_factor : float
            Filter factor of Standard Deviations away from mean background
            signal.
            Defaults to 2.0
        ref_factor : float
            Filter factor of Standard Deviations within mean of reference
            signal.
            Defaults to 2.0
        bkg : string
            Background signal column name.
            Defaults to 'bkg.mask_full'
        ref : string
            Reference signal column name.
            Defaults to 'Eu.mask_inside'

        """
        bkg_data = self._dataframe[bkg]
        bkg_mean = bkg_data.mean()
        bkg_sd = bkg_data.std()
        mask_bkg = ((bkg_data > (bkg_mean - bkg_factor * bkg_sd)) &
                    (bkg_data < (bkg_mean + bkg_factor * bkg_sd)))
        ref_data = self._dataframe[ref]
        ref_mean = ref_data.mean()
        ref_sd = ref_data.std()
        mask_ref = ((ref_data > (ref_mean - ref_factor * ref_sd)) &
                    (ref_data < (ref_mean + ref_factor * ref_sd)))
        filter_all = (mask_bkg & mask_ref)
        self._dataframe[self.flag_name] = ~filter_all
        num_out = len(self._dataframe[self._dataframe.flag == True])  # NOQA
        num_remain = len(self._dataframe[self._dataframe.flag == False])  # NOQA
        num_total = len(self._dataframe)
        num_percentage = ((num_out / num_total) * 100)
        print("Pre-filter: %i" % num_total)
        print("Post-filter: %i" % num_remain)
        print("Filtered: %i (%0.1f%%)" % (num_out, num_percentage))


class Decode(TableDataFrame):
    """Decode."""

    def __init__(self, target, decode_channels=None):
        """Initialize Decode."""
        super(Decode, self).__init__()
        self._target = target
        self._decode_channels = decode_channels
        # Instantiate ICP and GMM with default settings
        self._icp = ICP(target)
        self._gmm = Classify(target)
        self._dataframe = None
        # Setup settings object
        self.settings = Settings([self._icp, self._gmm], ['icp', 'gmm'])
        self.settings.__doc__ = """
            Return Decode object for settings purposes.

            See ICP and Classify documentation for detailed settings."""

    def decode(self, data, combine_data=None):
        """Decode MRBLEs."""
        if self._decode_channels is not None:
            pass
        self._icp.fit(data)
        icp_data = self._icp.transform()
        self._gmm.decode(icp_data)
        self._gmm_qc(data)
        self._dataframe = self._gmm.output
        self._dataframe = self._dataframe.combine_first(icp_data)
        if combine_data is not None:
            self.combine(combine_data)

    def _gmm_qc(self, data):
        print("Number of unique codes found:", self._gmm.found)
        print("Missing codes:", self._gmm.missing)
        s_score = silhouette_score(data, self._gmm.output.code)
        print("Silhouette Coefficient:", s_score)
        print("AIC:", self._gmm._gmix.aic(data))
        print("BIC:", self._gmm._gmix.bic(data))


class Analyze(TableDataFrame):
    """Analyze data.

    Parameters
    ----------
    functions : dict
        Dictionary of functions and their corresponding names.
        Default: {'mean': np.mean,
                  'median': np.median,
                  'sd': np.std,
                  'se': sp.stats.sem,
                  'N': len,
                  'CV': sp.stats.variation}
    norm_data: Pandas DataFrame or NumPy array.
        DataFrame containing per bead normalization data.
        First column must be codes column, subsequent column(s) must be signal.
        Defaults to None, no normalization.

    """

    def __init__(self, seq_list=None, flag_filt=True):
        """Set up statistics functions and set normalizartion data."""
        super(Analyze, self).__init__()
        self.seq_list = seq_list
        self.flag_filt = flag_filt
        self._dataframe = None
        self._data_per_bead = None
        # Default set of functions
        self.functions = {
            'mean': np.mean,
            'median': np.median,
            'sd': np.std,
            'se': sp.stats.sem,
            'N': len,
            'CV': sp.stats.variation
        }

    def analyze(self, data):
        """Analyze data."""
        if data.index.nlevels == 3:
            self._dataframe = self._multi(data)
        else:
            self._dataframe = self._single(data)

    def background(self):
        pass

    def normalize(self):
        pass

    @property
    def data_per_bead(self):
        """Return (normalized) data per bead."""
        return self._data_per_bead

    def _single(self, data):
        channels = list(data.columns)
        channels.remove('code')
        if self.flag_name in channels:
            channels.remove(self.flag_name)
        codes = np.unique(data.code.values).astype(int)
        result = {}
        for code in codes:
            for channel in channels:
                result[code] = self._iter_functions(
                    self.functions, data.loc[(data.code == code), channel])
        dataframe = pd.DataFrame.from_dict(result, orient='index')
        dataframe.index.rename('code', inplace=True)
        if self.seq_list is not None:
            for code in codes:
                if isinstance(self.seq_list, pd.DataFrame):
                    dataframe.loc[code, 'set.sequence'] = \
                        self.seq_list.sequence.iloc[code]
                    dataframe.loc[code, 'set.code'] = \
                        self.seq_list.code.iloc[code]
                else:
                    dataframe.loc[code, 'set.sequence'] = \
                        self.seq_list[code]
            dataframe['set.code'] = dataframe['set.code'].astype(int)
        return dataframe

    def _multi(self, data):
        levels = data.index.get_level_values(0).unique()
        result = [
            self._single(data.loc[level]) for level in levels
        ]
        dataframe = pd.concat(result, keys=levels)
        dataframe.index.rename(('set', 'code'), inplace=True)
        return dataframe

    def _get_stats_per_code(self, data, codes):
        pass

    def _norm_per_code(self):
        pass

    def _norm_per_bead(self):
        pass

    @staticmethod
    def _iter_functions(functions, data):
        data_na_omit = data.values[~np.isnan(data.values)]
        result = {
            key: func(data_na_omit) for (key, func) in functions.items()
        }
        return result


def get_stats_per_channel_and_code(data, channels, codes=None):
    """Get stats per channel and code."""
    bead_sets = []
    for channel in channels:
        bead_sets.append(get_stats_per_code(data, channel, codes))
    final_bead_set = pd.concat(bead_sets, keys=channels)
    return final_bead_set


def get_stats_per_code(data, channel,
                       norm=None, norm_channel=None, codes=None):
    """Get statistical values for each code in you data set.

    Parameters
    ----------
    data : Pandas DataFrame
        This is the per bead data.
    channel : string
        This is the channel (or column) to extract statistical values from.
    codes : int, list
        This value can be set if only a select (list) or one code (int) needs
        to be processed. If value is not set the unique codes from data, the
        Pandas DataFrame is used.
        Defaults to None.

    Returns
    -------
    bead_set : Pandas DataFrame
        This dataframe contains: AVG, SD, N, CV, SEM, and RSEM for each code.
        Codes start at 0. Code -1 represents weighted statistical values over
        all codes.

    """
    data_stats = []
    data_stats_norm = []
    n_codes = []
    if isinstance(codes, list):
        codes_set = codes
    elif (isinstance(data, pd.DataFrame)) and (codes is None):
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
        final_stats_norm = pd.DataFrame(data_stats_norm,
                                        columns=['AVG_NORM',
                                                 'SD_NORM',
                                                 'N_NORM',
                                                 'CV_NORM',
                                                 'SEM_NORM',
                                                 'RSEM_NORM',
                                                 'MED_NORM'])
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
    stats_array

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
    return np.array([data_mean,
                     data_sd,
                     data_n,
                     data_cv,
                     data_sem,
                     data_rsem,
                     data_median])
