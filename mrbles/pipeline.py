# !/usr/bin/env python
# -*- coding: utf-8 -*-

# [Future imports]
from __future__ import division, print_function

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
import sys
from math import sqrt

# Other
import numpy as np
import pandas as pd
import weightedstats as ws
import xarray as xr
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

# Intra-Package dependencies
from mrbles.core import FindBeadsImaging, ICP, Classify, SpectralUnmixing
from mrbles.data import DataOutput, ImageSetRead, Spectra


# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    from future.standard_library import install_aliases  # NOQA
    from __builtin__ import *  # NOQA
    install_aliases()


# General methods


def get_set_names(data_set, set_dim='set'):
    """Return list of sets."""
    if isinstance(data_set, pd.DataFrame):
        sets_list = list(data_set.groupby(set_dim).groups.keys())
    elif isinstance(data_set, xr.DataArray):
        sets_list = list(data_set.coords[set_dim].values)
    else:
        sets_list = None
    return sets_list


def combine_in_place(data_array_1, data_array_2):
    """Combine in place."""
    combined_data = data_array_1.combine_first(data_array_2).astype(float)
    return combined_data


def flatten_dict(d, prefix='.'):
    def items():
        # A clojure for recursively extracting dict like values
        for key, value in d.items():
            if isinstance(value, dict):
                for sub_key, sub_value in flatten_dict(value).items():
                    # Key name should imply nested origin of the dict,
                    # so we use a default prefix of __ instead of _ or .
                    yield key + prefix + sub_key, sub_value
            else:
                yield key, value
    return dict(items())


# Classes


class ReferenceSpectra(object):
    """Reference Spectra class.

    Parameters
    ----------
    files :
    object_channel :
    bead_size :
    dark_noise :

    """

    def __init__(self, files, object_channel, channels,
                 bead_size=16, dark_noise=0):
        """Instatiate Reference Spectra class."""
        super(ReferenceSpectra, self).__init__()
        self.files = files
        self.object_channel = object_channel
        if isinstance(channels, list) and (len(channels) == 2):
            self._channels = self._set_slice(channels)
        else:
            self._channels = channels
        self.bead_size = bead_size
        self.darknoise = dark_noise
        # Default bead find values
        self.thr_block = 33
        self.thr_c = 11
        self.ref_objects = FindBeadsImaging(bead_size,
                                            border_clear=True,
                                            output='np')
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
            data = self.get_spectrum(
                self.darknoise, channels, self.ref_objects.mask('inside'))
            self.ref_data.spec_add(name, data=data, channels=channel_names)

    def set_back(self, file, channels, roi_x, roi_y):
        """Set background spectrum."""
        if isinstance(channels, list) and len(channels) == 2:
            channels = self._set_slice(channels)
        BACK_CROPx = self._set_slice(roi_x)
        BACK_CROPy = self._set_slice(roi_y)
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
        """Plot reference sepctra."""
        self.ref_data_object.plot()

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


class References(DataOutput):
    """Create reference spectra."""

    def __init__(self, folders, files, object_channel, reference_channels,
                 bead_size=18, dark_noise=99, background='bkg', bkg_roi=None):
        """Init."""
        super(References, self).__init__()
        self.object_channel = object_channel
        self.reference_channels = reference_channels
        self.dark_noise = dark_noise
        self.background = background
        self.bkg_roi = bkg_roi
        self._images = Images(folders, files)
        self._find = Find(bead_size=bead_size,
                          border_clear=True,
                          circle_size=None)
        self._dataframe = None

    def load(self):
        """Process."""
        self._images.crop_x = self.crop_x
        self._images.crop_y = self.crop_y
        self._images.load()
        self._find.find(self._images[:, self.object_channel], ref=True)
        spectra = get_set_names(self._images.data)
        ref_channels = self._images[:, self.reference_channels].c.values
        data = [self.get_spectrum(self.dark_noise,
                                  self._images[x_set, self.reference_channels],
                                  self._find[x_set, 'inside'])
                for x_set in spectra if x_set not in self.background]
        if self.background in spectra:
            spectra.remove(self.background)
            spectra.append(self.background)
            data.append(self._get_back())
        self._dataframe = pd.DataFrame(data=np.array(data).T,
                                       columns=spectra,
                                       index=ref_channels)
        self._dataframe.index.name = 'channels'

    def _get_back(self):
        mask = np.ones((self._images.data.y.size,
                        self._images.data.x.size))
        if self.bkg_roi is None:
            bkg_images = self._images[self.background,
                                      self.reference_channels]
        else:
            bkg_images = self._images[self.background,
                                      self.reference_channels,
                                      self.bkg_roi[0],
                                      self.bkg_roi[1]]
        bkg_data = self.get_spectrum(0, bkg_images, mask)
        return bkg_data

    # TODO
    def settings(self):
        """Return Images and Find objects for settings purposes.

        Attributes
        ----------
        images : Images() object
            Returns Images() object for settings puproses.
            See class documentation for more information.
        find : Find() object
            Returns Find() object for settings puproses.
            See class documentation for more information.

        """
        pass

    def plot(self):
        """Plot Reference spectra."""
        self._dataframe.plot()

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


class Images(DataOutput):
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

    def __init__(self, folders, file_patterns, output='xr'):
        """Instatiate object and search for images."""
        super(Images, self).__init__()
        self.folders = folders
        self.file_patterns = file_patterns
        self.output = output
        self._dataframe = None
        self.files = self._find_images(self.folders, self.file_patterns)

    def __repr__(self):
        """Return xarray dataframe representation."""
        return repr([self._dataframe])

    def __getitem__(self, index):
        """Get method."""
        return self.data.loc[index]

    def load(self):
        """Load images in memory."""
        if self.files is None:
            return False
        dict_data = [ImageSetRead(file_set).xdata
                     for key, file_set in self.files.items()]
        self._dataframe = xr.concat(
            dict_data, dim=pd.Index(self.files.keys(), name='set'))

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
            e.g. '20160728_MOL_*'
            Defaults to '*.tif'.

        """
        image_files = []
        r = re.compile(pattern)
        for root, _, files in os.walk(path):
            file_list = [os.path.join(root, x) for x in files if r.match(x)]
            if file_list:
                image_files.append(file_list)
        return np.hstack(image_files).tolist()


class Find(DataOutput):
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
                                              circle_size=circle_size,
                                              output='xr')
        self._dataframe = None
        self._bead_dims = None

    # TODO change method for ref
    def find(self, object_images, combine_data=None, ref=False):
        """Execute finding images."""
        if (object_images.data.ndim > 3 and object_images.data.shape[0] > 1) or ref is True:
            self._dataframe, self._bead_dims = \
                self._find_multi_set(object_images)
        else:
            self._dataframe, self._bead_dims = \
                self._return_data(object_images)
        if combine_data is not None:
            dataframe = combine_in_place(combine_data, self._dataframe)
            data_object = DataOutput(data=dataframe)
            return data_object

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
        bead_no_list = {set_x: self._bead_dims.loc[set_x].shape[0]
                        for set_x in self.sets}
        return bead_no_list

    @property
    def sets(self):
        """Return list of sets."""
        # sets_list = list(self._bead_dims.groupby('set').groups.keys())
        return get_set_names(self._bead_dims)

    def _find_multi_set(self, object_image_sets):
        sets = get_set_names(object_image_sets)
        data = [self._return_data(image_set)
                for image_set in object_image_sets[:]]
        data_masks = [i[0] for i in data]
        data_dims = [i[1] for i in data]
        result_masks = xr.concat(data_masks,
                                 dim=pd.Index(sets, name='set'))
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


class Ratio(DataOutput):
    """Generate spectrally unmix ratio images."""

    def __init__(self, reference_spectra, background='bkg'):
        """Initialize Unmix."""
        super(Ratio, self).__init__()
        self.reference_spectra = reference_spectra
        self.background = background
        self.spec_unmix = SpectralUnmixing(reference_spectra, output='xr')

    def get(self, image_sets, reference, combine_data=None):
        """Get."""
        if image_sets.data.ndim > 3 and image_sets.data.shape[0] > 1:
            self._dataframe = self._find_multi_set(image_sets, reference)
        else:
            self._dataframe = self._return_data(image_sets, reference)
        if combine_data is not None:
            dataframe = combine_in_place(combine_data, self._dataframe)
            data_object = DataOutput(data=dataframe)
            return data_object

    def _find_multi_set(self, image_sets, reference):
        sets = get_set_names(self.reference_spectra.data)
        data = [self._return_data(image_sets[x_set], reference)
                for x_set in sets]
        result = xr.concat(data, dim='set')
        return result

    def _return_data(self, images, reference):
        self.spec_unmix.unmix(images)
        unmix_images = self.spec_unmix.xdata
        sets = get_set_names(unmix_images, set_dim='c')
        sets.remove(reference)
        if self.background in sets:
            sets.remove(self.background)
        ratio_images = np.divide(unmix_images.loc[sets],
                                 unmix_images.loc[reference])
        ratio_images.coords['c'] = [s + '_ratio' for s in sets]
        result = xr.concat([unmix_images, ratio_images], dim='c')
        return result


class Extract(DataOutput):
    """Extract data from images using masks."""

    def __init__(self, function=None):
        """Initialize Extract."""
        super(Extract, self).__init__()
        if function is None:
            self._func = np.median
        else:
            self._func = function
        self._dataframe = None

    def run(self, image_sets, mask_sets):
        """Run."""
        if image_sets.data.ndim == 4 and image_sets.data.shape[0] > 1:
            f_list = get_set_names(image_sets, set_dim='f')
            data = [self._get_data_images(image_sets[f], mask_sets[f])
                    for f in f_list]
            self._dataframe = pd.concat(data, keys=f_list)
        elif image_sets.data.ndim == 5 and image_sets.data.shape[0] > 1:
            pass
        else:
            self._dataframe = self._get_data_images(image_sets, mask_sets)

    def _get_data_images(self, images, masks):
        data = {
            str(image.c.values): {
                str(mask.c.values): self._get_data(image, mask)
                for mask in masks
            }
            for image in images
        }
        data_flatten = flatten_dict(data, prefix='.')
        dataframe = pd.DataFrame.from_dict(data_flatten, orient='columns')
        return dataframe

    @classmethod
    def _get_data(cls, image, mask, func=np.median):
        idx = cls._get_index(mask)
        data = ndi.labeled_comprehension(image, mask, idx, func, float, None)
        return data

    @staticmethod
    def _get_index(mask):
        return np.unique(mask.values[mask.values > 0])


class Decode(object):
    """Decode."""

    def __init__(self, spectra, target, assay_channels=None):
        """Initialize Decode."""
        self._spectra = spectra
        self._target = target
        self._assay_channels = assay_channels
        # Instantiate ICP and GMM with default settings
        self._icp = ICP(matrix_method='std',
                        max_iter=100,
                        tol=1e-4,
                        outlier_pct=0.01,
                        train=False)
        self._gmm = Classify(target,
                             tol=1e-5,
                             min_covar=1e-7,
                             sigma=1e-5,
                             train=False)
        self._result = None

    def decode(self):
        """Decode MRBLEs."""
        self._icp.fit()
        self._gmm.decode()
        if self._assay_channels is not None:
            pass

    @property
    def settings(self):
        """Return ICP and GMM objects for settings purposes."""
        icp = self._icp
        gmm = self._gmm

    def _icp(self):
        icp = ICP(matrix_method='std',
                  max_iter=100,
                  tol=1e-4,
                  outlier_pct=0.01,
                  train=False)
        icp.fit(bead_set.loc[filter_all,
                             ('rat_dy', 'rat_sm', 'rat_tm')], target)
        bead_set = bead_set.join(icp.transform())
        print("Tranformation matrix: ", icp.matrix)
        print("Offset vector: ", icp.offset)

    def _gmm(self):
        gmix = Classify(target, tol=1e-5, min_covar=1e-7,
                        sigma=1e-5, train=False)
        gmix.decode(bead_set.loc[filter_all,
                                 ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')])
        bead_set = bead_set.join(gmix.output)
        print("Number of unique codes found:", gmix.found)
        print("Missing codes:", gmix.missing)

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
    """Get stats per channel and code."""
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
