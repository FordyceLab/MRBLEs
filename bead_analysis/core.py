# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division
from __builtin__ import staticmethod, property

# [File header]     | Copy and edit for each file in this project!
# title             : core.py
# description       : Bead Kinetics module - Core functions
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20160308
# version update    : 20160808
# version           : v0.4
# usage             : As module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [Main header with project metadata] | Only in the main file!
# Copyright and credits
__copyright__   = ("Copyright 2016 - "
                   "The Encoded Beads Project - "
                   "ThornLab@UCSF and "
                   "FordyceLab@Stanford")
# Original author(s) of this Python project, like: ("...", 
__author__      = ("Bjorn Harink")  #               "name")
# People who contributed to this Python project, like: ["...",
__credits__     = ["Kurt Thorn",  #                     "name"] 
                   "Huy Nguyen"]
# Maintainer contact information
__maintainer__  = "Bjorn Harink" 
__email__       = "bjorn@harink.info" 
# Software information
__license__     = "MIT" 
__version__     = "v0.4"
__status__      = "Prototype"

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
# Image Processing
import cv2
from scipy import ndimage as ndi
from sklearn.metrics.pairwise import pairwise_distances
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.external import tifffile as tff
# Classification
from sklearn.mixture import GaussianMixture
# Project
from bead_analysis.data import *

class Filter(object):
    """Filter bead data sets.
    """

    
    @property
    def list(self):
        filter_all = (mask_size & mask_bkg & mask_ref)
        return filter_all

    @staticmethod
    def add(data_set, low, high):
        mask =  ( (data_set > (data_set.mean() - low * data_set.std()  )) & 
                  (data_set < (data_set.mean() + high * data_set.std() )) )
    

# TO-DO: UPDATE
def filterObjects(data, back, reference, objects_radius, back_std_factor=3, reference_std_factor=2, radius_min=3, radius_max=6):
    """Filter Objects
    Filter objects using x times SD from mean
    back = background data
    reference = reference data
    back_std_factor = x times SD from mean
    reference_std_factor = x times SD from mean
    """
    # Pre-filtered number
    pre_filter_no = data[:, 0].size

    # Mean and standard deviation of the background and the reference channel
    mean_reference = np.mean(reference)
    std_reference = np.std(reference)
    mean_back = np.mean(back)
    std_back = np.std(back)
    print(mean_reference, std_reference, mean_back, std_back)

    # Find indices of objects within search parameters
    # Check which objects are within set radius
    size_filter = np.logical_and(
        objects_radius >= radius_min, objects_radius <= radius_max)
    # Check which objects are within x SD from mean background signal
    back_filter = np.logical_and(back < (mean_back + back_std_factor * std_back),
                                 back > (mean_back - back_std_factor * std_back))
    # Check which objects are within x SD from mean reference signal
    refr_filter = np.logical_and(reference > (mean_reference - reference_std_factor * std_reference),
                                 reference < (mean_reference + reference_std_factor * std_reference))
    # Create list of indices of filtered-in objects
    filter_list = np.argwhere(np.logical_and(
        size_filter, np.logical_and(back_filter, refr_filter)))[:, 0]

    # Compare pre and post filtering object numbers
    post_filter_no = filter_list.size
    post_filter_per = int(
        ((pre_filter_no - post_filter_no) / post_filter_no) * 100)
    print("Pre-filter no:", pre_filter_no, ", Post-filter no:",
          post_filter_no, ", Filtered:", post_filter_per, "%")

    # Return list of indices of filtered-in objects
    return filter_list


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
    def __init__(self, min_r, max_r, param_1=10, param_2=10, annulus_width = 2, min_dist = None, enlarge = 1):
        self.min_r = min_r
        self.max_r = max_r
        self.annulus_width = annulus_width
        self.param_1 = param_1
        self.param_2 = param_2
        self.enlarge = enlarge
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
        """Make circle mask using Hough transform.
        """
        # Find initial circles using Hough transform
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1,  
                                   minDist=min_dist, 
                                   param1=param_1, 
                                   param2=param_2, 
                                   minRadius=min_r, 
                                   maxRadius=max_r)[0]
        if circles is None: return None
        # Make mask
        mask = np.zeros(image.shape, np.uint8)
        for c in circles:
            x, y, r = c[0], c[1], int(np.ceil(c[2]*enlarge))
            # Draw circle (line width -1 fills circle)
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
            circles_dim[label - 1, 0] = x
            circles_dim[label - 1, 1] = y
            circles_dim[label - 1, 2] = r
        return circles_dim

    def find(self, image):
        """Find objects in image and return data.
        """
        img = self.convert(image)
        # Find initial circles using Hough transform and make mask
        mask, circles = self.circle_mask(img, self.min_dist, 
                                     self.min_r, 
                                     self.max_r, 
                                     self.param_1, 
                                     self.param_2,
                                     self.enlarge)
        if mask is None: return None
        # Find and separate circles using watershed on initial mask
        self._labeled_mask = self.circle_separate(mask, circles)
        # Find center of circle and return dimensions
        self._circles_dim = self._get_dimensions(self._labeled_mask)
        # Create annulus mask
        self._labeled_annulus_mask = self.create_annulus_mask(self._labeled_mask)

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

    def overlay_image(self, image, annulus=None, dim=False):
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