# !/usr/bin/env python

# [Future imports]
# Function compatibility between Python 2.x and 3.x
from __future__ import print_function, division
from future.standard_library import install_aliases
install_aliases()
import sys
if sys.version_info < (3,0): from __builtin__ import *

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
import os
import types
import warnings
from math import ceil, pi, sqrt
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
        #assert len(types) == f.func_code.co_argcount
        assert len(types) == f.__code__.co_argcount
        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                        "arg %r does not match %s" % (a,t)
            return f(*args, **kwds)
        #new_f.func_name = f.func_name
        new_f.func_name = f.__name__
        return new_f
    return check_accepts


### Classes

class FindBeadsImaging(object):
    """Find beads based on pure imaging.

    Parameters
    ----------
    bead_size : int
        Approximate width of beads (circles) in pixels.
        Defaults to 18.

    eccen_param : int, list of int
        Sets the maximum of eccentricity [0-1] of the beads (circles).
        Values close to 0 mean very circular, values closer to 1 mean very elliptical.
        Defaults to 0.55.

    area_param : int, list of int
        Sets the default min and max fraction for bead (circle) area. 
        Set as single int (1+/-: 0.XX) value or of 2 values [0.XX, 1.XX].
        E.g. area_param=0.5 or area_param=[0.5, 1.5] filters all below 75% and above 125% of area calculated by approximate bead_size.
        Defaults to 0.5, which equals to [0.5, 1.5].

    Attributes
    ----------

    """
    def __init__(self, bead_size, eccen_param=0.55, area_param=0.5):
        # Default values for filtering
        self._bead_size = bead_size
        self._eccen_param = eccen_param
        self._area_param = area_param
        self.set_area_limits(bead_size)
        self.filter_params = [self._eccen_param, [self.area_min, self.area_max]]
        self.filter_names = ['eccentricity', 'area']
        self.slice_types = ['up', 'outside']
        # Default values OpenCV Thershold
        self.thr_block = 11
        self.thr_c = 15
        self.kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (3,3))
        self.filt_iter = 1
        # Default values for local background
        self.mask_bkg_size = 15
        self.mask_bkg_buffer = 3


    # Parameter methods
    def set_area_limits(self, bead_size):
        """"Sets area limits dependent on given bead width (pixels).
        Sets: maximum and minimum area.
        """
        # Set limits
        radius = ceil(self._bead_size/2)
        area_avg = pi * radius**2
        self.area_min, self.area_max = self.min_max(area_avg, self._area_param)


    # Main method
    # TODO: Split inside filter and whole bead filter.
    def find(self, image):
        # Convert image to uint8
        img = self.img2ubyte(image)
        # Threshold to binary image
        img_thr = self.img2thr(img, self.thr_block, self.thr_c)
        # Label all separate parts
        mask_inside = ndi.label(img_thr, structure=self.kernel)[0]
        self._mask_inside, self._mask_inside_neg = self.mask_filter(mask_inside, self.filter_params, self.filter_names, self.slice_types)
        # Check if image not empty
        if np.unique(self._mask_inside).size <= 1:
            return False
        # Find full bead
        img_thr_invert = (~img_thr.astype(bool)).astype(int)
        mask_all_bin = self._mask_inside + img_thr_invert
        mask_all_bin[mask_all_bin > 0] = 1
        D = ndi.distance_transform_edt(mask_all_bin, sampling=3)
        mask_full = watershed(-D, markers=self._mask_inside, mask=mask_all_bin)
        self._mask_bead, self._mask_bead_neg = self.mask_filter(mask_full, self.filter_params, self.filter_names, self.slice_types)
        # Create and update final masks
        self._mask_ring = self._mask_bead - self._mask_inside
        self._mask_ring[self._mask_ring < 0] = 0
        self._mask_inside[self._mask_bead_neg < 0] = 0
        # Create outside and buffered background areas around bead
        self._mask_outside = self.get_mask_outside(self._mask_bead, self.mask_bkg_size, buffer=0)
        self._mask_bkg = self.get_mask_outside(self._mask_bead, self.mask_bkg_size, buffer=self.mask_bkg_buffer)
        return True


    # Properties - Settings
    @property
    def bead_size(self):
        """Get or set approximate width of beads (circles) in pixels.
        """
        return self._bead_size
    @bead_size.setter
    def bead_size(self, bead_size):
        self._bead_size = bead_size
        self.set_area_limits(bead_size)

    @property
    def area_param(self):
        """Get or set approximate width of beads (circles) in pixels.
        """
        return self._area_param
    @area_param.setter
    def area_param(self, value):
        self._area_param = value
        self.set_area_limits(self.bead_size)
        self.filter_params = [self._eccen_param, [self.area_min, self.area_max]]

    @property
    def eccen_param(self):
        """Get or set approximate width of beads (circles) in pixels.
        """
        return self._eccen_param
    @area_param.setter
    def eccen_param(self, value):
        self._eccen_param = value
        self.filter_params = [self._eccen_param, [self.area_min, self.area_max]]

    # Properties - Output masks
    @property
    def mask_bead(self):
        return self._mask_bead

    @property
    def mask_ring(self):
        return self._mask_ring
    
    @property
    def mask_inside(self):
        return self._mask_inside

    @property
    def mask_outside(self):
        return self._mask_outside

    @property
    def mask_bkg(self):
        return self._mask_bkg

    # Properties - Output values
    @property
    def bead_num(self):
        return self.get_unique_count(self._mask_bead)

    @property
    def bead_labels(self):
        return self.get_unique_values(self._mask_bead)

    @property
    def bead_dims_bead(self):
        return self.get_dimensions(self._mask_bead)

    @property
    def bead_dims_inside(self):
        return self.get_dimensions(self._mask_inside)

    
    # Class methods
    @classmethod
    def get_mask_outside(cls, mask, size, buffer=0):
        if buffer > 0:
            mask_min = cls.mask_morph_step(buffer, mask)
        else:
            mask_min = mask
        mask_outside = cls.mask_morph_step(size, mask)
        mask_outside[mask_min > 0] = 0
        return mask_outside

    @classmethod
    def mask_filter(cls, mask, filter_params, filter_names, slice_types):
        # Get dimensions from the mask
        props = cls.get_dimensions(mask)
        # Get labels to be removed
        lbls_out = cls.filters(props, filter_params, filter_names, slice_types)
        # Create new masks
        mask_pos = mask.copy()
        mask_neg = mask.copy()
        # Set mask to 0 or negative label for labels outside limits.
        if lbls_out.size > 0:
            for lbl in lbls_out:
                mask_pos[mask == lbl] = 0
                mask_neg[mask == lbl] = -lbl
        return mask_pos, mask_neg

    @classmethod
    def filters(cls, properties, filter_params, filter_names, slice_types):
        # Get labels of areas outside of limits.
        lbls_out_tmp = [cls.filter(properties, param, name, stype) for param, name, stype in zip(filter_params, filter_names, slice_types)]
        lbls_out = np.unique(np.hstack(lbls_out_tmp))
        return lbls_out

    @classmethod
    def img2thr(cls, image, thr_block, thr_c):
        img = cls.img2ubyte(image)
        img_thr = cv2.adaptiveThreshold(src = img,
                                        maxValue = 1, 
                                        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresholdType = cv2.THRESH_BINARY,
                                        blockSize = thr_block,
                                        C = thr_c)
        return img_thr


    # Static methods
    @staticmethod
    def mask_morph_step(size, mask):
        morph_mask = mask.copy()
        if size < 0:
            for n in range(abs(size)):
                morph_mask = erosion(morph_mask)
        elif size > 0:
            for n in range(size):
                morph_mask = dilation(morph_mask)
        return morph_mask

    @staticmethod
    def filter(properties, filter_param, filter_name, slice_type):
        # Get labels of areas outside of limits.
        if type(filter_param) is list:
            if slice_type == 'outside':
                lbls_out = properties[(properties[filter_name] < filter_param[0]) | (properties[filter_name] > filter_param[1])].label.values
            elif slice_type == 'inside':
                lbls_out = properties[(properties[filter_name] >= filter_param[0]) & (properties[filter_name] <= filter_param[1])].label.values
        else:
            if slice_type == 'up':
                lbls_out = properties[properties[filter_name] > filter_param].label.values
            elif slice_type == 'down':
                lbls_out = properties[properties[filter_name] < filter_param].label.values
        return lbls_out

    @staticmethod
    def min_max(value, min_max):
        if min_max is list:
            r_min = value * min_max[0]
            r_max = value * min_max[1]
        else:
            r_min = value * (1 - min_max)
            r_max = value * (1 + min_max)
        return r_min, r_max

    @staticmethod
    def eccentricity(a, b):
        major = max([a, b])
        minor = min([a, b])
        return sqrt(1 - (minor**2 / major**2))

    @staticmethod
    def get_unique_values(mask):
        """Get all unique positive values from an array.
        """
        values = np.unique(mask[mask>0])
        if values.size == 0:
            values = None
        return values

    @staticmethod
    def get_unique_count(mask):
        """Get count of unique positive values from an array.
        """
        return np.unique(mask[mask>0]).size

    @staticmethod
    def get_dimensions(mask):
        properties = source_properties(mask, mask)
        if not properties:
            return  None
        tbl = properties_table(properties)  # Convert to table
        lbl = np.array(tbl['min_value'], dtype=np.int)
        x = tbl['xcentroid']
        y = tbl['ycentroid']
        r = tbl['equivalent_radius']
        area = tbl['area']
        perimeter = tbl['perimeter']
        eccentricity = tbl['eccentricity']
        pdata = np.array([lbl, x, y, r, area, perimeter, eccentricity]).T
        dims = pd.DataFrame(data=pdata, columns=['label','x_centroid','y_centroid','radius','area','perimeter','eccentricity'])
        return dims

    @staticmethod
    @accepts((np.ndarray, xd.DataArray))
    def img2ubyte(image):
        """Convert image to ubuyte (uint8) and rescale to min/max.
        """
        if type(image) is (xd.DataArray):
            image = image.values
        img_dtype = image.dtype
        if img_dtype is np.dtype('uint8'):
            return image
        img_min = image - image.min()
        img_max = img_min.max()
        img_conv = np.array( (img_min/img_max) * 255, dtype=np.uint8 )
        return img_conv
# Backwards compatibility with previous name.
def FindBeads2(*args, **kwargs):
    """Depracation warning: class renamed to FindBeadsImaging.
    Name changed to distinguish between FindBeadsImaging (imaging based) and FindBeadsCircle (hough-circle based).
    See docstring of FindBeadsImaging for information.
    """
    warnings.warn("Depracation warning: class renamed to FindBeadsImaging.")
    return FindBeadsImaging(*args, **kwargs)


class FindBeadsCircle(object):
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
# Backwards compatibility with previous name.
def FindBeads(*args, **kwargs):
    """Depracation warning: class renamed to FindBeadsCircle.
    Name changed to distinguish between FindBeadsImaging (imaging based) and FindBeadsCircle (hough-circle based).
    See docstring of FindBeadsCircle for information.
    """
    warnings.warn("Depracation warning: class renamed to FindBeads.")
    return FindBeadsCircle(*args, **kwargs)


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
            for n in range(naxes):
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
        for n in range(naxes):
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
            for n in range(naxes1):
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
        for n in range(self._nclusters):
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