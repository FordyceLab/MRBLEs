# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core Classes and Functions.

This file stores the core classes and functions for the MRBLEs Analysis module.
"""

# [File header]     | Copy and edit for each file in this project!
# title             : core.py
# description       : MRBLEs - Core Functions
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20160308

# [Future imports]
from __future__ import (absolute_import, division, print_function)
from builtins import (super, range, zip, round, int, object)

# [Modules]
# General Python
import multiprocessing as mp
import sys
import types
import warnings
from math import ceil, sqrt
# Other
import cv2
import numpy as np
import pandas as pd
import photutils
import skimage as sk
import skimage.morphology
import skimage.segmentation
import xarray as xr
from matplotlib import pyplot as plt
from packaging import version
from scipy import ndimage as ndi
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture

# Intra-Package dependencies
from mrbles.data import ImageDataFrame

# Function compatibility issues
# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    warnings.warn(
        "mrbles: Please use Python >3.6 for multiprocessing.")
# NumPy compatibility issue
if version.parse(np.__version__) < version.parse("1.14.0"):
    warnings.warn('mrbles: Please upgrade module NumPy >1.14.0!')
    RCOND = -1
else:
    RCOND = None


# Decorators
def accepts(*types):  # NOQA
    """Check input parameters for data types."""
    def _check_accepts(func):
        assert len(types) == func.__code__.co_argcount

        def _new_func(*args, **kwds):
            for (arg_f, type_f) in zip(args, types):
                assert isinstance(arg_f, type_f), \
                    "arg %r does not match %s" % (arg_f, type_f)
            return func(*args, **kwds)
        _new_func.func_name = func.__name__
        return _new_func
    return _check_accepts


# Classes
class FindBeadsImaging(ImageDataFrame):
    """Find and identify beads and their regions using imaging.

    Parallel computing version.

    Parameters
    ----------
    bead_size : int
        Approximate width of beads (circles) in pixels.
    border_clear : boolean
        Beads touching border or ROI will be removed.
        Defaults to True.
    circle_size : int
        Set circle size for auto find circular ROI.

    Attributes
    ----------
    area_min : float
        Sets the minimum area in pixels. Set to minimum size inside of ring.
        Defaults to 0.1 * area of set bead_size.
    area_max : float
        Sets the maximum area in pixels. Set maximum size outside of ring.
        Defaults to 1.5 * area of set bead_size.
    eccen_max : float
        Get or set maximum eccentricity of beads in value 0 to 1, where a
        perfect circle is 0.
        Defaults to 0.65.

    """

    def __init__(self, bead_size,
                 border_clear=True, circle_size=None, parallelize=False):
        """Find and identify beads and their regions using imaging."""
        super(FindBeadsImaging, self).__init__()
        self._bead_size = bead_size
        self.border_clear = border_clear
        self.circle_size = circle_size
        self.parallelize = parallelize
        # Default values for filtering
        self._area_min = 0.25 * self.circle_area(bead_size)
        self._area_max = 1.5 * self.circle_area(bead_size)
        self._eccen_max = 0.65
        # Default values for local background
        self.mask_bkg_size = 11
        self.mask_bkg_buffer = 2
        # Data set
        self._dataframe = None
        self._bead_dims = None
        # Adaptive Thersholding
        self.thr_block = 15
        self.thr_c = 11

    # Properties - Settings
    @property
    def bead_size(self):
        """Get or set approximate width of beads (circles) in pixels."""
        return self._bead_size

    @property
    def area_min(self):
        """Get or set minimum area of beads (circles) in pixels."""
        return self._area_min

    @area_min.setter
    def area_min(self, value):
        self._area_min = value

    @property
    def area_max(self):
        """Get or set minimum area of beads (circles) in pixels."""
        return self._area_max

    @area_max.setter
    def area_max(self, value):
        self._area_max = value

    @property
    def eccen_max(self):
        """Get or set maximum eccentricity of beads from 0 to 1.

        A perfect circle is 0 and parabola is 1.
        """
        return self._eccen_max

    @eccen_max.setter
    def eccen_max(self, value):
        self._eccen_max = value

    # Main function
    def find(self, image):
        """Execute finding beads image(s)."""
        if image.ndim == 3:
            if (sys.version_info >= (3, 0)) and (self.parallelize is True):
                mp_worker = mp.Pool()
                result = mp_worker.map(self._find, image)
                mp_worker.close()
                mp_worker.join()
            else:
                result = list(map(self._find, image))
            r_m = [i[0] for i in result]
            r_d = [i[1] for i in result]
            self._dataframe = xr.concat(r_m, dim='f')
            self._bead_dims = pd.concat(r_d,
                                        keys=list(range(len(r_d))),
                                        names=['f', 'bead_index'])
        else:
            self._dataframe, self._bead_dims = self._find(image)

    def _find(self, image):
        if self.circle_size is not None:
            img, roi_mask = self.circle_roi(image, self.circle_size)
        else:
            img = self._img2ubyte(image)
        bin_img = self.img2bin(img, self.thr_block, self.thr_c)
        mask_inside, _ = self._find_inside(bin_img)
        if np.unique(mask_inside).size <= 1:
            blank_img = np.zeros_like(bin_img)
            mask_bead = blank_img
            mask_ring = blank_img
            mask_outside = blank_img
            mask_inside = blank_img
            mask_bkg = blank_img
            bead_dims = None
            overlay_image = blank_img
        else:
            mask_bead, mask_bead_neg = self._find_watershed(mask_inside,
                                                            bin_img)
            # Create and update final masks
            mask_ring = mask_bead - mask_inside
            mask_ring[mask_ring < 0] = 0
            mask_inside[mask_bead_neg < 0] = 0
            # Create outside and buffered background areas around bead
            mask_outside = self.make_mask_outside(mask_bead,
                                                  self.mask_bkg_size,
                                                  buffer=0)
            mask_bkg = self.make_mask_outside(mask_bead_neg,
                                              self.mask_bkg_size,
                                              buffer=self.mask_bkg_buffer)
            if self.circle_size is not None:
                mask_bkg[~roi_mask] = 0
                mask_bkg[mask_bkg < 0] = 0
            bead_dims = self.get_dimensions(mask_bead)
            if bead_dims is None:
                blank_img = np.zeros_like(bin_img)
                mask_bead = blank_img
                mask_ring = blank_img
                mask_outside = blank_img
                mask_inside = blank_img
                mask_bkg = blank_img
                bead_dims = None
                overlay_image = blank_img
            else:
                bead_dims_overlay = bead_dims.loc[:, ('x_centroid',
                                                      'y_centroid',
                                                      'radius')]
                overlay_image = self.cross_overlay(img,
                                                   bead_dims_overlay,
                                                   color=False)
        masks = xr.DataArray(data=np.array([mask_bead,
                                            mask_ring,
                                            mask_inside,
                                            mask_outside,
                                            mask_bkg,
                                            overlay_image],
                                           dtype=np.uint16),
                             dims=['c', 'y', 'x'],
                             coords={'c': ['mask_full',
                                           'mask_ring',
                                           'mask_inside',
                                           'mask_outside',
                                           'mask_bkg',
                                           'mask_check']})
        return [masks, bead_dims]

    def _find_inside(self, bin_img):
        seg_img = self._bin2seg(bin_img)
        filter_params_inside = [[self._area_min, self._area_max]]
        filter_names_inside = ['area']
        slice_types_inside = ['outside']
        mask_inside, mask_inside_neg = self.filter_mask(seg_img,
                                                        filter_params_inside,
                                                        filter_names_inside,
                                                        slice_types_inside,
                                                        border_clear=False)
        return mask_inside, mask_inside_neg

    def _find_watershed(self, mask_inside, bin_img):
        bin_img_invert = self._img_invert(bin_img)
        mask_all_bin = mask_inside + bin_img_invert
        mask_all_bin[mask_all_bin > 0] = 1
        dist_trans = ndi.distance_transform_edt(mask_all_bin, sampling=3)
        mask_full = sk.morphology.watershed(np.negative(dist_trans),
                                            markers=mask_inside,
                                            mask=mask_all_bin)
        filter_params = [self._eccen_max,
                         [self.area_min, self.area_max]]
        filter_names = ['eccentricity', 'area']
        slice_types = ['up', 'outside']
        mask_bead, mask_bead_neg = self.filter_mask(mask_full,
                                                    filter_params,
                                                    filter_names,
                                                    slice_types,
                                                    self.border_clear)
        return mask_bead, mask_bead_neg

    # Functions
    def _data_return(self, value):
        if self._dataframe.ndim > 3:
            data = self._dataframe.loc[:, value].values
        else:
            data = self._dataframe.loc[value].values
        return data

    def mask(self, mask_type):
        """Return labeled mask of the specified mask type."""
        return self._data_return(mask_type)

    @property
    def mask_types(self):
        """Return list of mask types."""
        return self._dataframe.c.values.tolist()

    # Properties - Output values
    @property
    def bead_num(self):
        """Return number of beads labeled mask."""
        return self.get_unique_count(self._data_return("mask_full"))

    @property
    def bead_labels(self):
        """Return all positive labels of labeled mask."""
        return self.get_unique_values(self._data_return("mask_full"))

    @property
    def bead_dims(self):
        """Return found bead dimensions."""
        return self._bead_dims

    # Class methods
    @classmethod
    def make_mask_outside(cls, mask, size, buffer=0):
        """Return labeled mask of area around bead."""
        if buffer > 0:
            mask_min = cls._morph_mask_step(buffer, mask)
        else:
            mask_min = mask
        mask_outside = cls._morph_mask_step(size, mask)
        mask_outside[mask_min > 0] = 0
        return mask_outside

    @classmethod
    def img2bin(cls, image, thr_block=15, thr_c=11):
        """Convert and adaptive threshold image."""
        img = cls._img2ubyte(image)
        img_thr = cv2.adaptiveThreshold(src=img,
                                        maxValue=1,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # NOQA
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=thr_block,
                                        C=thr_c)
        return img_thr

    @classmethod
    def filter_mask(cls, mask, filter_params, filter_names, slice_types,
                    border_clear=False):
        """Filter labeled mask based on provided parameters."""
        # Get dimensions from the mask
        props = cls.get_dimensions(mask)
        # Get labels to be removed
        lbls_out = cls.filter_properties(
            props, filter_params, filter_names, slice_types)
        # Create new masks
        mask_pos = mask.copy()
        mask_neg = mask.copy()
        # Set mask to 0 or negative label for labels outside limits.
        if lbls_out.size > 0:
            for lbl in lbls_out:
                mask_pos[mask == lbl] = 0
                mask_neg[mask == lbl] = -lbl
        if border_clear is True:
            sk.segmentation.clear_border(mask_pos, in_place=True)
            sk.segmentation.clear_border(mask_neg, bgval=-1, in_place=True)
        return mask_pos, mask_neg

    @classmethod
    def filter_properties(cls, properties, filter_params, filter_names,
                          slice_types):
        """Get labels of areas outside of limits."""
        lbls_out_tmp = [cls.filter_property(properties, param, name, stype)
                        for param, name, stype in zip(filter_params,
                                                      filter_names,
                                                      slice_types)]
        lbls_out = np.unique(np.hstack(lbls_out_tmp))
        return lbls_out

    @classmethod
    def circle_roi(cls, image, circle_size, hough_settings=None):
        """Apply a circular image ROI.

        Parameters
        ----------
        image : NumPy array image

        hough_settings : list, int
            Settings for HoughCircles in list.
            list[0] = dp, list[1] = param1, list[2] = param2

        """
        img = cls._img2ubyte(image)
        # Default Hough settings
        if hough_settings is None:
            hough_dp = 2
            hough_param1 = 10
            hough_param2 = 7
        else:
            hough_dp = hough_settings[0]
            hough_param1 = hough_settings[1]
            hough_param2 = hough_settings[2]
        dims = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                                dp=hough_dp,
                                minDist=img.shape[0],
                                minRadius=circle_size,
                                maxRadius=img.shape[0],
                                param1=hough_param1,
                                param2=hough_param2)
        if len(dims[0]) != 1:
            mask_img = img
            mask = np.zeros_like(img, dtype=np.uint8)
            warnings.warn("No circular ROI found. Defaulting to whole image. "
                          "Please adjust circle_size, not use circle_size, or "
                          "crop images.")
        else:
            circle_y, circle_x, _ = np.round(np.ravel(dims[0])).astype(np.int)
            mask = cls.sector_mask(img.shape,
                                   [circle_x, circle_y],
                                   circle_size)
            mask_img = img.copy()
            mask_img[~mask] = 0
        return mask_img, mask

    # Static methods
    @staticmethod
    def sector_mask(shape, center, radius):
        """Return a boolean mask for a circular ROI."""
        mesh_x, mesh_y = np.ogrid[:shape[0], :shape[1]]
        center_x, center_y = center
        # convert cartesian --> polar coordinates
        r_2 = (mesh_x - center_x) * (mesh_x - center_x) + \
              (mesh_y - center_y) * (mesh_y - center_y)
        circmask = r_2 <= radius * radius
        return circmask

    @staticmethod
    def get_unique_values(mask):
        """Get all unique positive values from labeled mask."""
        values = np.unique(mask[mask > 0])
        if values.size == 0:
            values = None
        return values

    @staticmethod
    def get_unique_count(mask):
        """Get count of unique positive values from labeled mask."""
        return np.unique(mask[mask > 0]).size

    @staticmethod
    def filter_property(properties, filter_param, filter_name, slice_type):
        """Get labels of beads outside/inside/up/down of propert limits.

        Parameters
        ----------
        properties : photutils table
            Table with feature properties from labeled mask.
            >>> from photutils import source_properties, properties_table
            >>> tbl = properties_table(properties)
            >>> properties = source_properties(mask, mask)
        filter_param : float, int, list
            Parameters to filter by.
            If provided a list it will filter by range, inside or outside).
            If provided a value it filter up or down that value.
        slice_type : string
            'outside' : < >
            'inside'  : >= <=
            'up'      : >
            'down'    : <

        """
        if isinstance(filter_param, list):
            if slice_type == 'outside':
                lbls_out = properties[(properties[filter_name]
                                       < filter_param[0])
                                      | (properties[filter_name]
                                         > filter_param[1])].label.values
            elif slice_type == 'inside':
                lbls_out = properties[(properties[filter_name]
                                       >= filter_param[0])
                                      & (properties[filter_name]
                                         <= filter_param[1])].label.values
        else:
            if slice_type == 'up':
                lbls_out = properties[properties[filter_name]
                                      > filter_param].label.values
            elif slice_type == 'down':
                lbls_out = properties[properties[filter_name]
                                      < filter_param].label.values
        return lbls_out

    @staticmethod
    def _morph_mask_step(steps, mask):
        """Morph mask step-by-step using erosion or dilation.

        This function will erode or dilate step-by-step, in a loop, each
        labeled feature in labeled mask array.

        Parameters
        ----------
        steps : int
            Set number of dilation (positive value, grow outward) or erosion
            (negative value, shrink inward) steps.
        mask : NumPy array
            Labeled mask to be dilated or eroded.

        """
        morph_mask = mask.copy()
        if steps < 0:
            for _ in range(abs(steps)):
                morph_mask = sk.morphology.erosion(morph_mask)
        elif steps > 0:
            for _ in range(steps):
                morph_mask = sk.morphology.dilation(morph_mask)
        return morph_mask

    @staticmethod
    def _bin2seg(image):
        """Convert and adaptive threshold image."""
        ellipse_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                                   ksize=(3, 3))
        seg_img = ndi.label(image, structure=ellipse_kernel)[0]
        return seg_img

    @staticmethod
    def get_dimensions(mask):
        """Get dimensions of labeled regions in labeled mask."""
        properties = photutils.source_properties(mask, mask)
        if not properties:
            return None
        tbl = properties.to_table()  # Convert to table
        lbl = np.array(tbl['min_value'], dtype=np.int16)
        reg_x = tbl['xcentroid']
        reg_y = tbl['ycentroid']
        reg_r = tbl['equivalent_radius']
        reg_area = tbl['area']
        perimeter = tbl['perimeter']
        eccentricity = tbl['eccentricity']
        pdata = np.array([lbl, reg_x, reg_y, reg_r, reg_area,
                          perimeter, eccentricity]).T
        dims = pd.DataFrame(data=pdata,
                            columns=['label',
                                     'x_centroid',
                                     'y_centroid',
                                     'radius',
                                     'area',
                                     'perimeter',
                                     'eccentricity'])
        return dims

    @staticmethod
    def cross_overlay(image, dims, color=True):
        """Create image with overlay crosses."""
        img = skimage.color.gray2rgb(image)
        if isinstance(dims, pd.DataFrame):
            dims = np.array(np.round(dims.values.astype(np.float)),
                            dtype=np.int)
        for center_x, center_y, radius in zip(dims[:, 0],
                                              dims[:, 1],
                                              dims[:, 2]):
            line_y = slice(int(round(center_y) - round(radius)),
                           int(round(center_y) + round(radius)))
            line_x = slice(int(round(center_x) - round(radius)),
                           int(round(center_x) + round(radius)))
            width_x = int(round(center_x))
            width_y = int(round(center_y))
            img[width_y, line_x] = (20, 20, 220)
            img[line_y, width_x] = (20, 20, 220)
        if color is False:
            img = skimage.color.rgb2gray(img)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = skimage.img_as_uint(img)
        return img

    @staticmethod
    def show_image_overlay(image, image_blend,
                           alpha=0.3, cmap1='Greys_r', cmap2='jet'):
        """Overlay of 2 images using alpha blend.

        Parameters
        ----------
        image : NumPy array
            Base image.
        image_blend : NumPy arra
            Image to blend over base image.
        aplha : float
            Amount of blending. Value between 0 and 1.
            Defaults to 0.3.
        c_map1 : cmap
            Color scheme using cmap. See matplotlib for color schemes.
            Defaults to 'Greys_r', which are reversed grey values.

        """
        plt.axis('off')
        plt.imshow(image, cmap=cmap1)
        plt.imshow(image_blend, cmap=cmap2, interpolation='none', alpha=alpha)

    @staticmethod
    @accepts((np.ndarray, xr.DataArray))
    def _img2ubyte(image):
        """Convert image to ubuyte (uint8) and rescale to min/max.

        Parameters : NumPy or xarray
            Image to be converted and rescaled to ubuyte.

        """
        if isinstance(image, xr.DataArray):
            image = image.values
        img_dtype = image.dtype
        if img_dtype is np.dtype('uint8'):
            return image
        img_min = image - image.min()
        img_max = img_min.max()
        img_conv = np.array((img_min / img_max) * 255, dtype=np.uint8)
        return img_conv

    @staticmethod
    def _img_invert(img_thr):
        """Invert boolean image.

        Parameters
        ----------
        img_thr : NumPy array
            Boolean image in NumPy format.

        """
        img_inv = (~img_thr.astype(bool)).astype(int)
        return img_inv

    @staticmethod
    def circle_area(diameter):
        """Return area of circle.

        Parameters
        ----------
        diameter : float
            Diameter of circle.

        """
        radius = ceil(diameter / 2)
        return np.pi * radius**2

    @staticmethod
    def eccentricity(axis_a, axis_b):
        """Return eccentricity by major axes.

        Parameters
        ----------
        axis_a : float
            Size major axis a.
        axis_b : float
            Size major axis b.

        """
        major = max([axis_a, axis_b])
        minor = min([axis_a, axis_b])
        return sqrt(1 - (minor**2 / major**2))


class FindBeadsCircle(FindBeadsImaging):
    """Find and identify bead objects from image.

    Parameters changes for each setup/magnification/bead-set.

    Parameters
    ----------
    min_r : int
        Sets the minimum diameter of the bead in pixels.
    max_r : int
        Sets the maximum diameter of the bead in pixels.
    param_1 : int
        Sets the gradient steepness. CHECK
    param_2 : int
        Sets the sparsity. CHECK
    annulus_width : int, optional
        Sets the width of the annulus in pixels.
        Defaults to 2 pixels.
    min_dist : int, optional
        Sets the minimal distance between the centers of the beads in pixels.
        Defaults to 2x of the minimum diameter (min_r).
    enlarge : float, optional
        Enlarges the found diameter by this factor.
        1 remains equal, 1.1 enlarges by 10% and 0.9 shrinks by 10%.
        Defaults to 1, no enlarging/shrinking.

    """

    def __init__(self, bead_size, min_r, max_r,
                 param_1=99, param_2=7,
                 annulus_width=2,
                 min_dist=None, enlarge=1,
                 auto_filt=True, border_clear=False,
                 parallelize=True):
        """Instantiate FindBeadsCircle."""
        super(FindBeadsCircle, self).__init__(bead_size)
        self.min_r = min_r
        self.max_r = max_r
        self.annulus_width = annulus_width
        self.param_1 = param_1
        self.param_2 = param_2
        self.enlarge = enlarge
        self.auto_filt = auto_filt
        self.border_clear = border_clear
        self.parallelize = parallelize
        # Default values for local background
        self.mask_bkg_size = 11
        self.mask_bkg_buffer = 2
        if min_dist is not None:
            self.min_dist = min_dist
        else:
            self.min_dist = 2 * min_r
        self._labeled_mask = None
        self._labeled_annulus_mask = None
        self._circles_dim = None
        self._dataframe = None

    # @property
    # def labeled_mask(self):
    #     """Return labeled mask."""
    #     return self._labeled_mask

    # @property
    # def labeled_annulus_mask(self):
    #     """Return labeled annulus mask."""
    #     return self._labeled_annulus_mask

    # @property
    # def circles_dim(self):
    #     """Return circle dimensions."""
    #     return self._circles_dim

    # @staticmethod
    # def convert(image):
    #     """8 Bit Convert.

    #     Checks image data type and converts if necessary to uint8 array.
    #     image : M x N image array
    #     """
    #     try:
    #         img_type = image.dtype
    #     except ValueError:
    #         print("Not a NumPy array of image: %s" % image)
    #     else:
    #         if img_type == 'uint16':
    #             image = np.array(((image / 2**16) * 2**8), dtype='uint8')
    #     return image

    @staticmethod
    def circle_mask(image, min_dist, min_r, max_r, param_1, param_2, enlarge):
        """Find initial circles using Hough transform and return mask."""
        try:  # TO-DO: HACK - Fix later
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1,
                                       minDist=min_dist,
                                       param1=param_1,
                                       param2=param_2,
                                       minRadius=min_r,
                                       maxRadius=max_r)[0]
        except ValueError:
            return None
        mask = np.zeros(image.shape, np.uint8)  # Make mask
        for c in circles:
            x, y, r = c[0], c[1], int(np.ceil(c[2] * enlarge))
            # Draw circle on mask (line width -1 fills circle)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        return mask, circles

    @staticmethod
    def circle_separate(mask, circles):
        """Find and separate circles using watershed on initial mask."""
        D = ndi.distance_transform_edt(mask, sampling=1)
        markers_circles = np.zeros_like(mask)
        for _, circle in enumerate(circles):
            markers_circles[int(circle[1]), int(circle[0])] = 1
        markers = ndi.label(markers_circles, structure=np.ones((3, 3)))[0]
        labels = sk.morphology.watershed(np.negative(D), markers, mask=mask)
        # print("Number of unique segments found: {}".format(
        #     len(np.unique(labels)) - 1))
        return labels

    def _get_dimensions(self, labels):
        """Find center of circle and return dimensions."""
        idx = np.arange(1, len(np.unique(labels)))
        circles_dim = np.empty((len(np.unique(labels)) - 1, 3))
        for label in idx:
            # Create single object mask
            mask_detect = np.zeros(labels.shape, dtype="uint8")
            mask_detect[labels == label] = 255
            # Detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask_detect.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)
            # Get circle dimensions
            ((x, y), r) = cv2.minEnclosingCircle(c)
            circles_dim[label - 1, 0] = int(x)
            circles_dim[label - 1, 1] = int(y)
            circles_dim[label - 1, 2] = int(r)
        return circles_dim

    # Main function
    def find(self, image):
        """Execute finding beads image(s)."""
        if isinstance(image, xr.DataArray):
            image = image.values
        if image.ndim == 3:
            if (sys.version_info >= (3, 0)) and (self.parallelize is True):
                mp_worker = mp.Pool()
                result = mp_worker.map(self._find, image)
                mp_worker.close()
                mp_worker.join()
            else:
                result = list(map(self._find, image))
            r_m = [i[0] for i in result]
            r_d = [i[1] for i in result]
            self._dataframe = xr.concat(r_m, dim='f')
            self._bead_dims = pd.concat(r_d,
                                        keys=list(range(len(r_d))),
                                        names=['f', 'bead_index'])
        else:
            self._dataframe, self._bead_dims = self._find(image)

    def _find(self, image):
        """Find objects in image and return data."""
        img = self._img2ubyte(image)
        mask, circles = self.circle_mask(img, self.min_dist,
                                         self.min_r,
                                         self.max_r,
                                         self.param_1,
                                         self.param_2,
                                         self.enlarge)
        if mask is None:
            return None
        self._labeled_mask = self.circle_separate(mask, circles)
        self._circles_dim = self._get_dimensions(self._labeled_mask)
        bead_dims = self.get_dimensions(self._labeled_mask)
        self._labeled_annulus_mask = self.create_annulus_mask(
            self._labeled_mask)
        if self.auto_filt is True:
            self._filter()
        mask_outside = self.make_mask_outside(self._labeled_mask,
                                              self.mask_bkg_size,
                                              buffer=0)
        mask_bkg = self.make_mask_outside(self._labeled_mask,
                                          self.mask_bkg_size,
                                          buffer=self.mask_bkg_buffer)
        mask_inside = self._labeled_mask - self._labeled_annulus_mask
        mask_inside[mask_inside < 0] = 0
        bead_dims_overlay = bead_dims.loc[:, ('x_centroid',
                                              'y_centroid',
                                              'radius')]
        overlay_image = self.cross_overlay(img,
                                           bead_dims_overlay,
                                           color=False)
        masks = xr.DataArray(data=np.array([self._labeled_mask,
                                            self._labeled_annulus_mask,
                                            mask_inside,
                                            mask_outside,
                                            mask_bkg,
                                            overlay_image],
                                           dtype=np.uint16),
                             dims=['c', 'y', 'x'],
                             coords={'c': ['mask_full',
                                           'mask_ring',
                                           'mask_inside',
                                           'mask_outside',
                                           'mask_bkg',
                                           'mask_check']})
        return [masks, bead_dims]

    def create_annulus_mask(self, labeled_mask):
        """Create annulus mask from regular mask."""
        labeled_annulus_mask = labeled_mask.copy()
        for cd in self._circles_dim:
            if (int(cd[2] - self.annulus_width)) < (
                    (self.min_r + 1) - self.annulus_width):
                r_dim = 1
            else:
                r_dim = int(cd[2] - self.annulus_width)
            cv2.circle(labeled_annulus_mask,
                       (int(cd[0]), int(cd[1])), r_dim,
                       (0, 0, 0), -1)
        return labeled_annulus_mask

    def overlay_image(self, image, annulus=None, dim=None):
        """Overlay image with circles of labeled mask."""
        img = image.copy()
        if dim is not None:
            circ_dim = dim
        else:
            circ_dim = self._circles_dim
        for dim in circ_dim:
            if annulus is True:
                if (int(dim[2] - self.annulus_width)) < (
                        (self.min_r + 1) - self.annulus_width):
                    r_dim = 1
                else:
                    r_dim = int(dim[2] - self.annulus_width)
                cv2.circle(img, (int(dim[0]), int(
                    dim[1])), r_dim, (0, 255, 0), 1)
            cv2.circle(img, (int(dim[0]), int(dim[1])),
                       int(dim[2]), (0, 255, 0), 1)
        return img

    def _filter(self):
        remove_list = np.where((self._circles_dim[:, 2] < self.min_r) & (
            self._circles_dim[:, 2] > self.max_r))[0]
        if remove_list.size > 0:
            self._circles_dim = np.delete(
                self._circles_dim, remove_list, axis=0)
            for remove in remove_list:
                self._labeled_mask[self._labeled_mask == remove + 1] = 0
                self._labeled_annulus_mask[self._labeled_mask ==
                                           remove + 1] = 0
        if self.border_clear is True:
            sk.segmentation.clear_border(self._labeled_mask,
                                         in_place=True)
            sk.segmentation.clear_border(self._labeled_annulus_mask,
                                         in_place=True)


class SpectralUnmixing(ImageDataFrame):
    """Spectrally unmix images using reference spectra.

    Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm images
    to Dy, Sm and Tm nanophospohorous lanthanides using reference spectra for
    each dye.

    Parameters
    ----------
    ref_data : list, ndarray, Pandas DataFrame, mrbles.data.References
        Reference spectra for each dye channel as Numpy Array: N x M, where N
        are the spectral channels and M the dye channels.

    """

    def __init__(self, ref_data):
        """Instantiate SpectralUnmixing."""
        super(SpectralUnmixing, self).__init__()
        self._ref_object = ref_data
        if isinstance(ref_data, pd.DataFrame):
            self._ref_data = ref_data.values
            self._names = list(ref_data.keys())
        elif hasattr(ref_data, 'data'):
            self._ref_data = ref_data.data.values
            self._names = list(ref_data.data.keys())
        else:
            raise TypeError(
                "Wrong type. Only mrbles dataframes, or Pandas DataFrame types.")  # NOQA
        self._ref_size = self._ref_data[0, :].size
        self._dataframe = None
        self._c_size = None
        self._y_size = None
        self._x_size = None

    def __repr__(self):
        """Return Xarray dataframe representation."""
        return repr([self._dataframe])

    def unmix(self, images):
        """Unmix images based on initiated reference data.

        Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm
        images to Dy, Sm and Tm nanophospohorous lanthanides using reference
        spectra for each dye.

        Parameters
        ----------
        image_data : NumPy array, Xarry DataArray, mrbles.Images
            Spectral images as NumPy array: N x M x P,
            where N are the spectral channels and M x P the image pixels
            (Y x X).

        """
        if isinstance(images, xr.DataArray):
            images = images.values
        if images.ndim > 3:
            data = [self._unmix(image) for image in images]
            self._dataframe = xr.concat(data, dim='f')
        else:
            self._dataframe = self._unmix(images)

    def _unmix(self, images):
        if self._ref_data.shape[0] != images.shape[0]:
            print("Number of channels not equal. Ref: ",
                  self._ref_data.shape[0], " Image: ", images.shape[0])
            raise IndexError
        self._sizes(images)
        img_flat = self._flatten(images)
        unmix_flat = np.linalg.lstsq(self._ref_data, img_flat, rcond=RCOND)[0]
        unmix_result = self._rebuilt(unmix_flat)
        dataframe = xr.DataArray(unmix_result,
                                 dims=['c', 'y', 'x'],
                                 coords={'c': self._names})
        return dataframe

    # Private functions
    def _sizes(self, images):
        """Get sizes images: Channels, Y, X."""
        self._c_size = images[:, 0, 0].size
        self._y_size = images[0, :, 0].size
        self._x_size = images[0, 0, :].size

    def _flatten(self, images):
        """Flatten X and Y of images in NumPy array."""
        images_flat = images.reshape(
            self._c_size, (self._y_size * self._x_size))
        return images_flat

    def _rebuilt(self, images_flat):
        """Rebuilt images to NumPy array."""
        images = images_flat.reshape(
            self._ref_size, self._y_size, self._x_size)
        return images


class ICP(object):
    """Iterative Closest Point (ICP).

    Iterative Closest Point (ICP) algorithm to minimize the difference
    between two clouds of points.

    Parameters
    ----------
    matrix_method : string/function/list, optional
        Transformation matrix method. Standard methods: 'max', 'mean', 'std'.
        Other options: own function or list of initial guesses.
        Defaults to 'std'.
    offset : list of float, optional
    max_iter : int, optional
        Maximum number of iterations.
        Defaults to 100.
    tol : float, optional
        Convergence threshold. ICP will stop after delta < tol.
        Defaults to 1e-4.
    outlier_pct : float, optional
        Discard percentile 0.x of furthest distance from target. Percentile
        given in fraction [0-1], e.g. '0.001'.
        Defaults to 0.
    train : boolean
        Turn on (True) or off (False) traning mode.
        This will keep the current tranformation from resetting to default
        initial values.
        Defaults to True.
    echo : boolean
        Turn on (True) or off (False) printing information while in process.
        Prints the delta for each iteration, the final number of iterations,
        and the final transformation and offset matrices.

    Attributes
    ----------
    matrix : NumPy array
        This stores the transformation matrix.
    offset : NumPy vector
        This stores the offset vector.

    Functions
    ---------
    fit : function
        Function to find ICP using set parameters and attributes.
    transform : function
        Function to apply transformat data using current transformation matrix
        and offset vector.

    """

    def __init__(self, target,
                 matrix_method='std',
                 offset=None,
                 max_iter=100,
                 tol=1e-4,
                 outlier_pct=0.01):
        """Instantiate Iterative Closest Point (ICP) object."""
        if isinstance(target, pd.DataFrame):
            target = target.values
        self._target = target
        self.matrix, self.matrix_func = self._set_matrix_method(matrix_method)
        self.max_iter = max_iter
        self.tol = tol
        self.outlierpct = outlier_pct
        self.offset = offset
        self.train = False
        self.echo = True
        self._pdata = None

    def _set_matrix_method(self, matrix_method):
        """Set matrix method."""
        matrix = None
        if matrix_method == 'max':
            matrix_func = np.max
        elif matrix_method == 'mean':
            matrix_func = np.mean
        elif matrix_method == 'std':
            matrix_func = np.std
        # Use own or other function
        elif isinstance(matrix_method, types.FunctionType):
            matrix_func = matrix_method
        # Use list of initial ratios
        elif isinstance(matrix_method, list):
            matrix_func = matrix_method
            naxes = len(matrix_method)
            matrix = np.eye(naxes)
            for n in range(naxes):
                matrix[n, n] = matrix_method[n]
        else:
            raise ValueError("Matrix method invalid: %s" % matrix_method)
        return matrix, matrix_func

    def _set_matrix(self, data, target):
        """Set initial guess matrix."""
        matrix = self.matrix_create(self.matrix_func, target, data)
        return matrix

    def _set_offset(self, data, target):
        """Set initial guess offset."""
        naxes = len(data[0, :])
        offset = np.ones(naxes)
        for n in range(naxes):
            offset[n] = np.min(target[:, n]) - np.min(data[:, n])
        return offset

    @staticmethod
    def matrix_create(func, input1, input2):
        """Create identity matrix and set values with function on inputs e.g 'np.mean'.

        Parameters
        ----------
        func : function
            Function to apply on input1 divided by input2, e.g. 'np.std'.
            Insert function without function call: ().
        input1 : list, ndarray
        input2 : list, ndarray

        Returns
        -------
        matrix : ndarray
            Returns func(input1/input2)

        """
        naxes1 = len(input1[0, :])
        naxes2 = len(input2[0, :])
        if naxes1 == naxes2:
            matrix = np.eye(naxes1)
            for n in range(naxes1):
                matrix[n, n] = func(input1[:, n]) / func(input2[:, n])
        else:
            raise ValueError(
                "Lengths of input1 = %s and input2 = %s do not match",
                naxes1,
                naxes2)
        return matrix

    def transform(self, data=None):
        """Apply transformation matrix to data."""
        if (self._pdata is not None) and data is None:
            self._pdata.reset_index(drop=True, inplace=True)
            tdata = np.dot(self._pdata.values, self.matrix) + self.offset
            result = pd.DataFrame()
            for num, val in enumerate(self._pdata.index):
                for n, v in enumerate(self._pdata.columns):
                    result.loc[val, ('%s_icp' % v)] = tdata[num, n]
        else:
            result = np.dot(data, self.matrix) + self.offset
        return result

    def fit(self, data, target=None):
        """Fit Iterative Closest Point."""
        if isinstance(data, pd.DataFrame):
            self._pdata = data
            data = data.values
        if target is None:
            target = self._target
        if (self.offset is None) or (self.train is False):
            self.offset = self._set_offset(data, target)
        else:
            warnings.warn("Training mode: ON")
        if (self.matrix is None) or (self.train is False):
            self.matrix = self._set_matrix(data, target)
        else:
            warnings.warn("Training mode: ON")

        delta = 1
        for i in range(self.max_iter):
            if delta < self.tol:
                print("Converged after:", i)
                break

            # Copy old to compare to new
            matrix_old = self.matrix
            offset_old = self.offset

            # Apply transform
            data_transform = self.transform(data)

            # Compare distances between tranformed data and target
            distances = pairwise_distances(data_transform, target)
            min_dist = np.min(distances, axis=1)
            # Filter percentile of furthest away points
            min_dist_pct = np.percentile(
                min_dist, [0, (1 - self.outlierpct) * 100])[1]
            min_dist_filt = np.argwhere(min_dist < min_dist_pct)[:, 0]
            # Match codes and levels
            matched_code = np.argmin(distances, axis=1)
            matched_levels = target[matched_code[min_dist_filt], :]
            # Least squares
            dist = np.c_[data[min_dist_filt], np.ones(
                len(data[min_dist_filt, 0]))]
            mat = np.linalg.lstsq(dist, matched_levels, rcond=RCOND)[0]

            # Store new tranformation matrix and offset vector
            self.matrix = mat[0:-1, :]
            self.offset = mat[-1, :]

            # Compare step by step delta
            d_compare = np.sum(np.square(self.matrix - matrix_old))
            d_compare = d_compare + np.sum(np.square(self.offset - offset_old))
            n_compare = np.sum(np.square(self.matrix)) + \
                np.sum(np.square(self.offset))
            delta = np.sqrt(d_compare / n_compare)
            print("Delta: ", delta)


class Classify(object):
    """Classification of beads by Gaussian Mixture Model.

    Parameters
    ----------
    target : list, NumPy array
        List of target ratios.
    tol : float
        Tolerance.
        Defaults to 1e-5.
    min_covar : float
        Minimum covariance.
        Defaults to 1e-7.
    sigma : float
        Minimum significance.
        Defaults to 1e-5.
    train : boolean
        Sets training mode. Remembers covariance matrix or resets to initial
        covariance matrix.
        Defaults to False.

    """

    def __init__(self, target,
                 sigma=1e-5, train=False, **kwargs):
        """Instantiate Classification object."""
        if isinstance(target, pd.DataFrame):
            target = target.values
        self._target = target
        kwargs.setdefault('covariance_type', 'full')
        kwargs.setdefault('tol', 1e-5)
        kwargs.setdefault('reg_covar', 1e-5)
        self.__dict__.update(kwargs)
        self._sigma = sigma
        self.train = train

        self._nclusters = len(self._target[:, 0])
        self._naxes = len(self._target[0, :])

        self._probs = None
        self._log_prob = None

        self._data = None
        self._predict = None
        self._log_proba = None

        self._init = True
        self._setup_gmix(**kwargs)

    def __repr__(self):
        """Return GaussianMixture object."""
        return repr([self._gmix])

    def _setup_gmix(self, **kwargs):
        if (self.train is False) or (self._init is True):
            self._gmix = GaussianMixture(n_components=self._nclusters,
                                         means_init=self._target,
                                         weights_init=self.init_weights,
                                         precisions_init=self.init_covars,
                                         **kwargs)
            self._init = False
        else:
            warnings.warn("Training mode: ON")

    @property
    def init_covars(self):
        """Return initial covariance matrix."""
        sigmas = np.eye(self._naxes) * self._sigma
        covars = np.tile(sigmas, (self._nclusters, 1, 1))
        return np.linalg.inv(covars)

    @property
    def init_weights(self):
        """Return initial weights."""
        weights = np.tile(1 / self._nclusters, (self._nclusters))
        return weights

    @property
    def stds(self):
        """Return Choleski based SD."""
        return np.linalg.cholesky(self._gmix.covariances_)

    @property
    def means(self):
        """Return means."""
        return self._gmix.means_

    @property
    def probs(self):
        """Return probabilities."""
        return self._probs

    def _set_probs(self, data):
        self._probs = 1 - np.exp(-self._gmix.score_samples(data))

    @property
    def log_prob(self):
        """Return log probabilities."""
        return self._log_proba

    def _set_log_prob(self, data):
        self._log_proba = self._gmix.score_samples(data)

    def ellipsoids(self, nsigma, resolution=100):
        """Create CI ellipsoids."""
        elps = []
        unit_sphere = self.unit_sphere(resolution)
        for n in range(self._nclusters):
            C = (nsigma * np.dot(self.stds,
                                 np.reshape(unit_sphere,
                                            (unit_sphere.shape[0],
                                             unit_sphere[1].size))))
            C += np.matlib.repmat(np.reshape(self.means[n], (3, 1)), 1,
                                  unit_sphere[0].size)
            elps.append(np.reshape(C, unit_sphere.shape))
        return elps

    @property
    def output(self):
        """Return codes, probability and log probability."""
        data = pd.DataFrame(columns=['code', 'prob', 'log_prob'])
        if isinstance(self._data, pd.DataFrame):
            for num, val in enumerate(self._data.index):
                data.loc[val, ('code')] = self._predict[num]
                data.loc[val, ('prob')] = self.probs[num]
                data.loc[val, ('log_probability')] = self.log_prob[num]
        else:
            data[('code')] = self._predict
            data[('prob')] = self.probs
            data[('log_probability')] = self.log_prob
        return data

    @property
    def found(self):
        """Return found codes."""
        return len(np.unique(self._predict))

    @property
    def missing(self):
        """Return missing codes."""
        if len(np.unique(self._predict)) != self._nclusters:
            missing = np.setxor1d(np.unique(self._predict),
                                  np.arange(0, self._nclusters))
        else:
            missing = None
        return missing

    def code_metrics(self, nsigma=3, resolution=100):
        """Return code metrics."""
        data = pd.DataFrame()
        data['means'] = self.means.tolist()
        data['stds'] = self.stds.tolist()
        data['ellipsoids'] = self.ellipsoids(nsigma, resolution)
        return data

    def decode(self, data):
        """Decode mrbles."""
        self._setup_gmix()
        self._data = data
        self._gmix.fit(data, self._target)
        self._predict = self._gmix.predict(data)
        self._set_probs(data)
        self._set_log_prob(data)

    @staticmethod
    def unit_sphere(resolution=100):
        """Return unit sphere."""
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi = np.linspace(0, np.pi, resolution)
        x = np.outer(np.cos(theta), np.sin(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.ones(resolution), np.cos(phi))
        return np.array((x, y, z))
