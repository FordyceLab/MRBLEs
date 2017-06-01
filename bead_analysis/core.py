# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division
from __builtin__ import *

# [File header]     | Copy and edit for each file in this project!
# title             : core.py
# description       : Bead Kinetics module - Core functions
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20160308
# version update    : 20170601
# version           : v0.5
# usage             : As module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [TO-DO]
# Check error exceptions
# Create error checking functions for clustering
# Update filterObjects

# [Modules]
# General Python
import sys
import os
import types
import warnings
from math import sqrt
# Data Structure
import numpy as np
import pandas as pd
import xarray as xd
# Image Processing
import cv2
from scipy import ndimage as ndi
from photutils import source_properties, properties_table
#from scipy.misc import bytescale
from sklearn.metrics.pairwise import pairwise_distances
from skimage.feature import peak_local_max
from skimage.morphology import watershed, dilation, erosion
from skimage.draw import circle
from skimage.external import tifffile as tff
# Classification
from sklearn.mixture import GaussianMixture
# Project
from bead_analysis.data import *


### Decorators

def accepts(*types):
    """Checks input parameters for data types.
    """
    def check_accepts(f):
        assert len(types) == f.func_code.co_argcount
        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                        "arg %r does not match %s" % (a,t)
            return f(*args, **kwds)
        new_f.func_name = f.func_name
        return new_f
    return check_accepts


### Classes

class FindBeads2(object):
    """

    Attributes
    ----------
    param1 : int
        First parameters of Hough circle find algorithm.
        Defaults to PARAM1 (100).
    param2 : int
        First parameters of Hough circle find algorithm.
        Defaults to PARAM2 (5)
    """
    ## Default values
    # Default values OpenCV Hough
    global PARAM1
    PARAM1 = 200
    global PARAM2
    PARAM2 = 10
    # Default values OpenCV Thershold and Filter
    global THR_BLOCK
    THR_BLOCK = 11
    global THR_C
    THR_C = 15
    global KERNEL
    KERNEL = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (3,3))
    global FILT_ITER
    FILT_ITER = 1

    def __init__(self, bead_size, *args, **kwargs):
        self.bead_size = bead_size
        self.circles = None
        self._lbl_mask = None
        self._lbl_mask_ann = None
        self._lbl_mask_bkg = None
        self.mask_bkg_size = 15
        self.mask_bkg_buffer = 3
        self.mask_ann_size = 2
        # Default values OpenCV Hough
        self.param1 = PARAM1
        self.param2 = PARAM2
        # Default values OpenCV Thershold
        self.thr_block = THR_BLOCK
        self.thr_c = THR_C
        self.kernel = KERNEL
        self.filt_iter = FILT_ITER

    @property
    def bead_size(self):
        return self._bead_size
    @bead_size.setter
    def bead_size(self, value):
        self._bead_size = value
        self.c_min, self.c_max, self.c_min_dist = self.get_bead_dims(value)

    @property
    def bead_num(self):
        if self._lbl_mask is not None:
            return self.get_bead_num(self._lbl_mask)
        else:
            return None

    @property
    def bead_labels(self):
        return self.get_bead_labels(self._lbl_mask)

    @staticmethod
    def get_bead_labels(mask):
        idx = np.unique(mask[mask>0])
        #idx = np.delete(idx, idx == 0)
        return idx

    @staticmethod
    def get_bead_num(mask):
        return len(np.unique(mask[mask>0])) - 1

    @property
    def mask_org(self):
        return self._lbl_mask

    @property
    def mask_ann(self):
        return self._lbl_mask_ann

    @property
    def mask_bkg(self):
        return self._lbl_mask_bkg

    @property
    def bead_dims(self):
        props = source_properties(self._lbl_mask, self._lbl_mask)
        tbl = properties_table(props)
        x = tbl['xcentroid']
        y = tbl['ycentroid']
        r = tbl['equivalent_radius']
        area = tbl['area']
        #lbl = tbl['max_value']
        #dims = pd.DataFrame([x, y, r, area, lbl], columns=['x', 'y', 'r', 'area', lbl])
        dims = np.array([x,y,r]).T
        return dims

    def find(self, image):
        #bin_img_mask, circles = self.img2bin(image, 
        #                            [self.c_min, self.c_max, self.c_min_dist], 
        #                            self.param1,
        #                            self.param2,
        #                            self.thr_block, 
        #                            self.thr_c, 
        #                            self.filt_iter, 
        #                            self.kernel)
        img = self.img2ubyte(image)
        img_thr = self.img2thr(img, self.thr_block, self.thr_c)

        labels = ndi.label(img_thr, structure=self.kernel)[0]        
        self._lbl_mask, self._lbl_mask_incl_neg = self.lbl_mask_flt( labels )

        img_thr_invert = np.invert(img_thr.copy())-254
        labels_all_bin = self._lbl_mask.copy() + img_thr_invert
        labels_all_bin[labels_all_bin > 0] = 1
        D = ndi.distance_transform_edt(labels_all_bin, sampling=3)
        labels_full = watershed(-D, markers=self._lbl_mask, mask=labels_all_bin)
        self._lbl_mask_ann, self._lbl_mask_ann_incl_neg = self.lbl_mask_flt( labels_full ) - self._lbl_mask
        self._lbl_mask_ann[self._lbl_mask_ann < 0] = 0
        self._lbl_mask[self._lbl_mask_ann < 0] = 0

        #self._lbl_mask = self.lbl_mask_flt( self.create_labeled_mask(bin_img_mask, circles) )
        #self._lbl_mask_ann = self.lbl_mask_ann(self._lbl_mask, self.mask_ann_size)
        self._lbl_mask_bkg_incl_neg = self.lbl_mask_bkg(self._lbl_mask_incl_neg+self._lbl_mask_incl_neg, 
                                               self.mask_bkg_size, 
                                               self.mask_bkg_buffer) / 2
        self._lbl_mask_bkg = self._lbl_mask_bkg_incl_neg.copy()
        self._lbl_mask_bkg[self._lbl_mask_bkg < 0] = 0

    @classmethod
    def lbl_mask_flt(cls, labels):
        idx = np.unique(labels)
        props = source_properties(labels, labels)
        tbl = properties_table(props)

        area_high = np.median(tbl['area'])*1.25
        area_low = np.median(tbl['area'])*0.75
        eccentricity = 0.55

        indices_ec = np.argwhere(tbl['eccentricity'] > eccentricity)
        indices_ar_max = np.argwhere(tbl['area'] > area_high)
        indices_ar_min = np.argwhere(tbl['area'] < area_low)
        indices_all = np.unique(np.concatenate((indices_ec,indices_ar_max,indices_ar_min)))
        lbl_filter = labels.copy()
        lbl_filter_incl_neg = labels.copy()
        if len(indices_all) > 0:
            for x in indices_all:
                lbl_filter[labels == idx[x+1]] = 0
                lbl_filter_incl_neg[labels == idx[x+1]] = -idx[x+1]
        return lbl_filter, lbl_filter_incl_neg

    def morph_filter(self):
        idx = cls.get_bead_labels(labels)
        props = source_properties(labels, labels)
        tbl = properties_table(props)


    #@classmethod
    #def lbl_mask_ann(cls, mask, size):
    #    mask_min = mask.copy()
    #    mask_max = cls.mask_morph_step(-size, mask)
    #    mask_min[mask_max > 0] = 0
    #    return mask_min

    @classmethod
    def lbl_mask_ann(cls, mask, size):
        mask_max = cls.mask_morph_step(size, mask)
        mask_max[mask > 0] = 0
        return mask_max

    @classmethod
    def lbl_mask_bkg(cls, mask, size, buffer=0):
        if buffer > 0:
            mask_min = cls.mask_morph_kernel(buffer, mask)
        else:
            mask_min = mask
        mask_max = cls.mask_morph_kernel(size, mask)
        mask_max[mask_min > 0] = 0
        return mask_max

    @classmethod
    def mask_morph_kernel(cls, size, mask):
        morph_mask = None
        kernel = cls.circle_kernel(abs(size))
        if size < 0:
            morph_mask = erosion(mask, kernel)
        elif size > 0:
            morph_mask = dilation(mask, kernel)
        return morph_mask

    @classmethod
    def mask_morph_step(cls, size, mask):
        morph_mask = mask.copy()
        if size < 0:
            for n in xrange(abs(size)):
                morph_mask = erosion(morph_mask)
        elif size > 0:
            for n in xrange(size):
                morph_mask = dilation(morph_mask)
        return morph_mask

    @staticmethod
    def circle_kernel(size):
        kernel = np.zeros((size, size), dtype=np.uint8)
        rr, cc = circle(np.floor(size/2), np.floor(size/2), np.ceil(size/2))
        kernel[rr, cc] = 1
        return kernel

    @staticmethod
    def create_labeled_mask(image, circles, kernel=KERNEL):
        img = image.copy()
        D = ndi.distance_transform_edt(img, sampling=3)
        markers_circles = np.zeros_like(img)
        for idx, c in enumerate(circles):
            markers_circles[int(c[1]),int(c[0])] = 1
        markers = ndi.label(markers_circles, structure=kernel)[0]
        labels = watershed(-D, markers, mask=img)
        return labels

    @staticmethod
    @accepts((np.ndarray, xd.DataArray))
    def img2ubyte(image):
        if type(image) is (xd.DataArray):
            image = image.values
        img_dtype = image.dtype
        if img_dtype is np.dtype('uint8'):
            return image
        img_min = image - image.min()
        img_max = img_min.max()
        img_conv = np.array( (img_min/img_max) * 255, dtype=np.uint8 )
        return img_conv

    @classmethod
    def img2thr(cls, image, thr_block=THR_BLOCK, thr_c=THR_C):
        img = cls.img2ubyte(image)
        img_thr = cv2.adaptiveThreshold(src = img,
                            maxValue = 1, 
                            adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            thresholdType = cv2.THRESH_BINARY,
                            blockSize = thr_block,
                            C = thr_c)
        return img_thr

    @staticmethod
    def thr2fill(image, circles, kernel=KERNEL):
        img_fill = image.copy()
        flood_mask = np.zeros((image.shape[0]+2, image.shape[1]+2), dtype='uint8')
        for idx, c in enumerate(circles):
            cv2.floodFill(image=img_fill, mask=flood_mask,
                          seedPoint = (c[1],c[0]), 
                          newVal = 2, 
                          loDiff = 0, 
                          upDiff = 1)
        img_fill[image == 0] = 0     # Add previous threshold image to filled image
        img_fill[img_fill == 0] = 1  # Set lines to 1
        img_fill[img_fill == 2] = 0  # Set background to 0
        img_fill_final = ndi.binary_fill_holes(img_fill, structure=kernel).astype(np.uint8)
        return img_fill_final

    @staticmethod
    def fill2filter(image, iter=FILT_ITER, kernel=KERNEL):
        img_filter = cv2.morphologyEx(image, 
                                      cv2.MORPH_OPEN, 
                                      kernel, 
                                      iterations = iter)
        return img_filter

    @staticmethod
    def get_bead_dims(bead_size):
        """Set default bead dimensions, min/max range, and min distance.
        """
        c_radius = bead_size / 2
        c_min = int(c_radius * 0.75)
        c_max = int(c_radius * 1.25)
        c_min_dist = (c_min * 2) - 1
        return c_min, c_max, c_min_dist

    #@classmethod
    #def img2bin(cls, image, 
    #            bead_size_param, param1=PARAM1, param2=PARAM2, 
    #            thr_block=THR_BLOCK, thr_c=THR_C, 
    #            iter=FILT_ITER, kernel=KERNEL):
    #    img = cls.img2ubyte(image)
    #    img_thr = cls.img2thr(img, thr_block, thr_c)
    #    circles = cls.circle_find(img, bead_size_param, param1, param2)
    #    img_fill = cls.thr2fill(img_thr, circles, kernel)
    #    img_final = cls.fill2filter(img_fill, iter=iter, kernel=kernel)
    #    return img_final, circles

    @classmethod
    def circle_find(cls, image, bead_size_parem, param1=PARAM1, param2=PARAM2):
        """Find circles using OpenCV Hough transform.
        """
        img = cls.img2ubyte(image)
        if type(bead_size_parem) is int:
            c_min, c_max, c_min_dist = cls.get_bead_dims(bead_size_parem)
        else:
            c_min, c_max, c_min_dist = bead_size_parem
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=c_min_dist,
                                   minRadius=c_min, 
                                   maxRadius=c_max,
                                   param1=param1,
                                   param2=param2)
        return circles[0]

class FindBeads(object):
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
    def __init__(self, min_r, max_r, param_1=10, param_2=10, annulus_width = 2, min_dist = None, enlarge = 1, auto_filt=True):
        self.min_r = min_r
        self.max_r = max_r
        self.annulus_width = annulus_width
        self.param_1 = param_1
        self.param_2 = param_2
        self.enlarge = enlarge
        self.auto_filt = auto_filt
        # TO-DO proper method
        if min_dist is not None:
            self.min_dist = min_dist
        else:
            self.min_dist = 2*min_r
        self._labeled_mask = None
        self._labeled_annulus_mask = None
        self._circles_dim = None

    @property
    def labeled_mask(self):
        return self._labeled_mask

    @property
    def labeled_annulus_mask(self):
        return self._labeled_annulus_mask

    @property
    def circles_dim(self):
        return self._circles_dim

    @staticmethod
    def convert(image):
        """8 Bit Convert
        Checks image data type and converts if necessary to uint8 array.
        image = M x N image array
        """
        try:
            img_type = image.dtype
        except ValueError:
            print("Not a NumPy array of image: %s" % image)
        except:
            print("Unexpected error:", sys.exc_info())
        else:
            if img_type == 'uint16':
                image = np.array( ((image / 2**16) * 2**8), dtype='uint8')
        finally:
            return image

    @staticmethod
    def circle_mask(image, min_dist, min_r, max_r, param_1, param_2, enlarge):
        """Find initial circles using Hough transform and return mask.
        """
        try:  # TO-DO: HACK - Fix later
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1,  
                                       minDist=min_dist, 
                                       param1=param_1, 
                                       param2=param_2, 
                                       minRadius=min_r, 
                                       maxRadius=max_r)[0]
        except:
            return None
        mask = np.zeros(image.shape, np.uint8)  # Make mask
        for c in circles:
            x, y, r = c[0], c[1], int(np.ceil(c[2]*enlarge))
            # Draw circle on mask (line width -1 fills circle)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        return mask, circles

    @staticmethod
    def circle_separate(mask, circles):
        """Find and separate circles using watershed on initial mask.
        """
        D = ndi.distance_transform_edt(mask, sampling=1)
        markers_circles = np.zeros_like(mask)
        for idx, circle in enumerate(circles):
            markers_circles[int(circle[1]),int(circle[0])] = 1
        markers = ndi.label(markers_circles, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=mask)
        print("Number of unique segments found: {}".format(
            len(np.unique(labels)) - 1))
        return labels

    def _get_dimensions(self, labels):
        """Find center of circle and return dimensions.
        """
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

    def find(self, image):
        """Find objects in image and return data.
        """
        img = self.convert(image)
        mask, circles = self.circle_mask(img, self.min_dist, 
                                     self.min_r, 
                                     self.max_r, 
                                     self.param_1, 
                                     self.param_2,
                                     self.enlarge)
        if mask is None: return None
        self._labeled_mask = self.circle_separate(mask, circles)
        self._circles_dim = self._get_dimensions(self._labeled_mask)
        self._labeled_annulus_mask = self.create_annulus_mask(self._labeled_mask)
        if self.auto_filt is True:
            self._filter()

    def create_annulus_mask(self, labeled_mask):
        """Create annulus mask from regular mask.
        """
        labeled_annulus_mask = labeled_mask.copy()
        for cd in self._circles_dim:
            if (int(cd[2] - self.annulus_width)) < ((self.min_r+1) - self.annulus_width): 
                r_dim = 1
            else:
                r_dim = int(cd[2] - self.annulus_width)
            cv2.circle(labeled_annulus_mask, 
                       (int(cd[0]), int(cd[1])), r_dim, 
                       (0, 0, 0), -1)
        return labeled_annulus_mask

    def overlay_image(self, image, annulus=None, dim=None):
        """Overlay Image
        Overlay image with circles of labeled mask
        """
        img = image.copy()
        if dim is not None:
            circ_dim = dim
        else:
            circ_dim = self._circles_dim
        for dim in circ_dim:
            if annulus is True:
                if (int(dim[2] - self.annulus_width)) < ((self.min_r+1) - self.annulus_width): 
                    r_dim = 1
                else:
                    r_dim = int(dim[2] - self.annulus_width)
                cv2.circle(img, (int(dim[0]), int(dim[1])), r_dim, (0, 255, 0), 1)
            cv2.circle(img, (int(dim[0]), int(dim[1])), int(dim[2]), (0, 255, 0), 1)
        return img

    def _filter(self):
        remove_list = np.where( (self._circles_dim[:,2] < self.min_r) & (self._circles_dim[:,2] > self.max_r) )[0]
        if remove_list.size > 0:
            self._circles_dim = np.delete(self._circles_dim, remove_list, axis=0)
            for remove in remove_list:
                self._labeled_mask[self._labeled_mask == remove + 1] = 0
                self._labeled_annulus_mask[self._labeled_mask == remove + 1] = 0


class SpectralUnmixing(FrozenClass):
    """Spectrally unmix images using reference spectra.
    
    Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm images to Dy, Sm and Tm nanophospohorous lanthanides using reference spectra for each dye.
    
    parameters
    ----------
    ref_data : list, ndarray, bead_analysis.data.Spectra
        Reference spectra for each dye channel as Numpy Array: N x M, where N are the spectral channels and M the dye channels.

    image_data : list, ndarray
        Spectral images as NumPy array: N x M x P, where N are the spectral channels and M x P the image pixels (Y x X)

    names : list
        List of channel names. When using Spectra object, names are imlied.
    """
    def __init__(self, ref_data, names=None):
        if isinstance(ref_data, Spectra):
            self._ref_object = ref_data
            self._ref_data = ref_data.ndata
            self._names = ref_data.spec_names
        elif isinstance(ref_data, np.ndarray):
            self._ref_data = ref_data
            if self._names is None:
                self._names = range(len(self._ref_data[0,:]))
        else:
            raise TypeError("Wrong type. Only Bead-Analysis Spectra or Numpy ndarray types.")
        self._dataframe = pd.Panel(items=self._names)
        #self._freeze()

    def __repr__(self):
        """Returns Pandas dataframe representation.
        """
        return repr([self._dataframe])

    def __getitem__(self, index):
        """Get method, see method 'spec_get'.
        """
        return self._dataframe.ix[index].values

    def unmix(self, images):
        """Unmix
        Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm images to Dy, Sm and Tm nanophospohorous lanthanides using reference spectra for each dye.
        ref_data = Reference spectra for each dye channel as Numpy Array: N x M, where N are the spectral channels and M the dye channels 
        image_data = Spectral images as NumPy array: N x M x P, where N are the spectral channels and M x P the image pixels (Y x X)
        """
        # Check if inputs are NumPy arrays and check if arrays have equal channel sizes
        self._ref_size = self._ref_data[0, :].size
        self._c_size = images[:, 0, 0].size
        self._y_size = images[0, :, 0].size
        self._x_size = images[0, 0, :].size
        if self._ref_data.shape[0] != images.shape[0]:
            print("Number of channels not equal. Ref: ", ref_shape, " Image: ", img_shape)
            raise IndexError
        img_flat = self._flatten(images)
        unmix_flat = np.linalg.lstsq(self._ref_data, img_flat)[0]
        unmix_result = self._rebuilt(unmix_flat)
        # TO-DO Change to proper insert
        self._dataframe = pd.Panel(unmix_result, items=self._names)
    
    @property
    def pdata(self):
        return self._dataframe
    @property
    def ndata(self):
        return self._dataframe.values

    # Private functions
    def _flatten(self, images):
        """Flatten
        Flatten X and Y of images in NumPy array
        """
        images_flat = images.reshape(self._c_size, (self._y_size * self._x_size))
        return images_flat

    def _rebuilt(self, images_flat):
        """Rebuilt
        Rebuilt images to NumPy array
        """
        images = images_flat.reshape(self._ref_size, self._y_size, self._x_size)
        return images


class ICP(object):
    """Iterative Closest Point (ICP) algorithm to minimize the difference between two clouds of points.

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
        Discard percentile 0.x of furthest distance from target. Percentile given in fraction [0-1], e.g. '0.001'.
        Defaults to 0.

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
        Function to apply transformat data using current transformation matrix and offset vector.
    """
    def __init__(self, matrix_method='std', offset=None, max_iter=100, tol=1e-4, outlier_pct=0, train=False):
        self.matrix, self.matrix_func = self._set_matrix_method(matrix_method)
        self.max_iter = max_iter
        self.tol = tol
        self.outlierpct = outlier_pct
        self.offset = offset
        self._train = train

        self._pdata = None

    def _set_matrix_method(self, matrix_method):
        """Set matrix method
        """
        matrix = None
        if matrix_method == 'max':
            matrix_func = np.max
        elif matrix_method == 'mean':
            matrix_func = np.mean
        elif matrix_method == 'std':
            matrix_func = np.std
        elif isinstance(matrix_method, types.FunctionType):  # Use own or other function
            matrix_func = matrix_method
        elif isinstance(matrix_method, types.ListType):  # Use list of initial ratios
            matrix_func = matrix_method
            naxes = len(matrix_method)
            matrix = np.eye(naxes)
            for n in xrange(naxes):
                matrix[n, n] = matrix_method[n]
        else:
            raise ValueError("Matrix method invalid: %s" % matrix_method)
        return matrix, matrix_func

    def _set_matrix(self, data, target):
        """Set initial guess matrix
        """
        matrix = self.matrix_create(self.matrix_func, target, data)
        return matrix

    def _set_offset(self, data, target):
        """Set initial guess offset
        """
        naxes = len(data[0, :])
        offset = np.ones(naxes)
        for n in xrange(naxes):
            offset[n] = np.min(target[:, n]) - np.min(data[:, n])
        return offset

    @staticmethod
    def matrix_create(func, input1, input2):
        """Create identity matrix and set values with function on inputs e.g 'np.mean'.

        parameters
        ----------
        naxes : int
            Number of axes or features.
        
        func : function
            Function to apply on input1 divided by input2, e.g. 'np.std'. 
            Insert function without function call: ().
        input1 : list, ndarray
        input2 : list, ndarray

        returns
        -------
        matrix : ndarray
            Returns func(input1/input2)
        """
        naxes1 = len(input1[0, :])
        naxes2 = len(input2[0, :])
        if naxes1 == naxes2:
            matrix = np.eye(naxes1)
            for n in xrange(naxes1):
                matrix[n, n] = func(input1[:,n]) / func(input2[:,n])
        else:
            raise ValueError("Lengths of input1 = %s and input2 = %s do not match", naxes1, naxes2)
        return matrix

    def transform(self, data=None):
        """Apply transformation matrix to data.
        """
        if (self._pdata is not None) and data is None:
            tdata = np.dot(self._pdata.values, self.matrix) + self.offset
            result = pd.DataFrame()
            for num, val in enumerate(self._pdata.index):
                for n, v in enumerate(self._pdata.columns):
                    result.loc[val, ('%s_icp' % v)] = tdata[num,n]
        else:
            result = np.dot(data, self.matrix) + self.offset
        return result

    def fit(self, data, target):
        """ICP
        Iterative Closest Point
        """
        if type(data) is pd.DataFrame:
            self._pdata = data
            data = data.values

        if (self.offset is None) or (self._train is False):
            self.offset = self._set_offset(data, target)
        else:
            warnings.warn("Training mode: ON")

        if (self.matrix is None) or (self._train is False):
            self.matrix = self._set_matrix(data, target)
        else:
            warnings.warn("Training mode: ON")

        delta = 1
        for i in xrange(self.max_iter):
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
            min_dist_pct = np.percentile(min_dist, [0, (1-self.outlierpct)*100])[1]
            min_dist_filt = np.argwhere(min_dist < min_dist_pct)[:, 0]
            # Match codes and levels
            matched_code = np.argmin(distances, axis=1)
            matched_levels = target[matched_code[min_dist_filt], :]
            # Least squaress
            d = np.c_[data[min_dist_filt], np.ones(len(data[min_dist_filt, 0]))]
            m = np.linalg.lstsq(d, matched_levels)[0]
            
            # Store new tranformation matrix and offset vector
            self.matrix = m[0:-1, :]
            self.offset = m[-1, :]

            # Compare step by step delta
            d_compare = np.sum(np.square(self.matrix - matrix_old))
            d_compare = d_compare + np.sum(np.square(self.offset - offset_old))
            n_compare = np.sum(np.square(self.matrix)) + np.sum(np.square(self.offset))
            delta = sqrt(d_compare / n_compare)
            print("Delta: ", delta)

class Classify(object):
    """Classification of beads by Gaussian Mixture Model.

    Parameters
    ----------
    target : list of ratios
        Bla.
    """
    def __init__(self, target, tol=1e-5, min_covar=1e-7, sigma=1e-5, train=False):
        self._target = target
        self._tol = tol
        self._min_covar = min_covar
        self._sigma = sigma
        self._train = train

        self._nclusters = len(self._target[:, 0])
        self._naxes = len(self._target[0, :])
        
        self._confs = None
        self._log_prob = None

        self._init = True
        self._setup_gmix()

    def __repr__(self):
        """Returns GaussianMixture object.
        """
        return repr([self._gmix])

    def _setup_gmix(self):
        if (self._train is False) or (self._init is True):
            self._gmix = GaussianMixture(covariance_type='full', 
                                         tol=self._tol, 
                                         reg_covar=self._min_covar, 
                                         n_components = self._nclusters, 
                                         means_init = self._target, 
                                         weights_init = self.init_weights, 
                                         precisions_init = self.init_covars)
            self._init = False
        else:
            warnings.warn("Training mode: ON")
        
    @property
    def init_covars(self):
        sigmas = np.eye(self._naxes) * self._sigma
        covars = np.tile(sigmas, (self._nclusters, 1, 1))
        return np.linalg.inv(covars)

    @property
    def init_weights(self):
        weights = np.tile(1 / self._nclusters, (self._nclusters))
        return weights

    @property
    def stds(self):
        return np.linalg.cholesky(self._gmix.covariances_)

    @property
    def means(self):
        return self._gmix.means_

    @property
    def confs(self):
        return self._confs
    def _set_confs(self, data):
        self._confs = 1-np.exp(-self._gmix.score_samples(data))

    @property
    def log_prob(self):
        return self._log_proba
    def _set_log_prob(self, data):
        self._log_proba = self._gmix.score_samples(data)

    def ellipsoids(self, nsigma, resolution=100):
        elps = []
        unit_sphere = self.unit_sphere(resolution)
        for n in xrange(self._nclusters):
            C = (nsigma * np.dot(self.stds, np.reshape(unit_sphere, (unit_sphere.shape[0], unit_sphere[1].size)))) + \
                np.matlib.repmat(np.reshape(self.means[n], (3,1)), 1, unit_sphere[0].size)
            elps.append(np.reshape(C, unit_sphere.shape))
        return elps

    @property
    def output(self):
        data =  pd.DataFrame(columns=['code','confidence','log_prob'])
        if type(self._data) is pd.DataFrame:
            for num, val in enumerate(self._data.index):
                data.loc[val, ('code')] = self._predict[num]
                data.loc[val, ('confidence')] = self.confs[num]
                data.loc[val, ('log_prob')] = self.log_prob[num]
        else:
            data[('code')] = self._predict
            data[('confidence')] = self.confs
            data[('log_prob')] = self.log_prob
        return data

    @property
    def found(self):
        return len(np.unique(self._predict))

    @property
    def missing(self):
        if len(np.unique(self._predict)) != self._nclusters:
            missing = np.setxor1d(np.unique(self._predict), np.arange(0,self._nclusters))
        else: 
            missing = None
        return missing

    #TO-DO    
    def code_metrics(self, nsigma=3, resolution=100):
        data =  pd.DataFrame()
        data['means'] = self.means.tolist()
        data['stds'] = self.stds.tolist()
        data['ellipsoids'] = self.ellipsoids(nsigma, resolution)
        return data

    def decode(self, data):
        self._setup_gmix()
        self._data = data
        self._gmix.fit(data, self._target)
        self._predict = self._gmix.predict(data)
        self._set_confs(data)
        self._set_log_prob(data)

    @staticmethod
    def unit_sphere(resolution=100):
        theta = np.linspace(0,2*np.pi,resolution)
        phi = np.linspace(0,np.pi,resolution)
        x = np.outer(np.cos(theta),np.sin(phi))
        y = np.outer(np.sin(theta),np.sin(phi))
        z = np.outer(np.ones(resolution),np.cos(phi))
        return np.array((x,y,z))