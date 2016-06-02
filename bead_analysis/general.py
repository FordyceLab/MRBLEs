# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : general.py
# description       : Bead Kinetics module
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20160308
# version update    : 20160504
# version           : v0.3
# usage             : As module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General
import sys
import re
import os
import glob
# Math
from math import sqrt
import numpy as np
# Image import
import bioformats as bf
from bioformats import log4j
import javabridge as jb  # Used for bioformats
# Image processing
import cv2
from scipy import ndimage as ndi
import sklearn
from sklearn import metrics
from sklearn import mixture
from sklearn import preprocessing
#from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
# Image display
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# File processing
from sklearn.externals import joblib

# TO-DO
# Check error exceptions
# Create error checking functions for clustering

# Main functions and classes
# Software-package specific functions
def mmScanPath(path, pattern=".tif"):
    """Micro-Manager Scan Path
    Scan a Micro-Manager images directory structure
    path = Micro-Manager image path
    """
    files_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(pattern):
                files_list.append(os.path.join(root, file))
    return files_list


def getBack(image_data, square):
    """Get Background
    Get background reference of specified area
    image_data = single image used for background
    square = coordinates of region of interest [Y1, Y2, X1, X2]
    """
    c_size = image_data[:, 0, 0].size - 1
    channels = xrange(1, c_size + 1)
    ref_data = np.empty((c_size), dtype="float64")
    for ch in channels:
        img_tmp = image_data[ch, square[0]:square[1], square[2]:square[3]]
        ref_data[ch - 1] = np.median(img_tmp)
    sum = ref_data.sum()
    return np.divide(ref_data, sum)


def unmix(ref_data, image_data):
    """Unmix
    Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm images to Dy, Sm and Tm nanophospohorous lanthanides using reference spectra for each dye.
    ref_data = Reference spectra for each dye channel as Numpy Array: N x M, where N are the spectral channels and M the dye channels 
    image_data = Spectral images as NumPy array: N x M x P, where N are the spectral channels and M x P the image pixels (Y x X)
    """
    # Check if inputs are NumPy arrays and check if arrays have equal channel sizes
    try:
        ref_shape = ref_data.shape
        img_shape = image_data.shape
    except IOError:
        print("Input not NumPy array")
    if ref_shape[0] != img_shape[0]:
        print("Number of channels not equal. Ref: ", ref_shape, " Image: ", img_shape)
        raise IndexError
    c_size = image_data[:, 0, 0].size
    y_size = image_data[0, :, 0].size
    x_size = image_data[0, 0, :].size
    ref_size = ref_data[0, :].size
    img_flat = image_data.reshape(c_size, (y_size * x_size))
    unmix_flat = np.linalg.lstsq(ref_data, img_flat)[0]
    unmix_result = unmix_flat.reshape(ref_size, y_size, x_size)
    return unmix_result


def getSpectralMedianIntensities(labels, images):
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


def getRatios(labels, images, reference):
    """Get Ratios
    Get median ratio of each object.
    """
    idx = np.arange(1, len(np.unique(labels)))
    data_size = len(np.unique(labels)) - 1
    channel_no = images[:, 0, 0].size
    channels = xrange(channel_no)
    ratio_data = np.empty((data_size, channel_no))
    for ch in channels:
        # Get pixel-by-pixel ratios
        image_tmp = np.divide(images[ch, :, :], reference)
        # Get median ratio of each object
        ratio_data[:, ch] = ndi.labeled_comprehension(
            image_tmp, labels, idx, np.median, float, -1)
    return ratio_data


def icp(data, target, max_iter=100, tol=1e-4, input_matrix=None, input_offset=None):
    """ICP
    Iterative Closest Point
    """
    nLn = len(data[0, :])
    nData = len(data[:, 0])
    if input_offset is None:
        offset = np.ones(nLn)
        for n in xrange(nLn):
            offset[n] = np.min(target[:, n]) - np.min(data[:, n])

    if input_matrix is None:
        matrix = np.eye(nLn)
        for n in xrange(nLn):
            matrix[n, n] = (np.mean(target[:, n])) / np.mean(data[:, n])
    else:
        matrix = input_matrix

    delta = 1
    for i in xrange(max_iter):
        if delta < tol:
            print("Converged after:", i)
            break

        # Copy old to compare to new
        matrix_old = matrix
        offset_old = offset
        result = np.dot(data, matrix) + offset

        distances = metrics.pairwise.pairwise_distances(result, target)
        matched_code = np.argmin(distances, axis=1)
        matched_levels = target[matched_code, :]
        d = np.c_[data, np.ones(len(data[:, 0]))]
        m = np.linalg.lstsq(d, matched_levels)[0]
        matrix = m[0:-1, :]
        offset = m[-1, :]

        d_compare = np.sum(np.square(np.subtract(matrix, matrix_old)))
        d_compare = d_compare + np.sum(np.square(np.subtract(offset, offset_old)))
        n_compare = np.sum(np.square(matrix)) + np.sum(np.square(offset))
        delta = sqrt(d_compare / n_compare)
        print("Delta: ", delta)
    return result

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

# TO-DO
def trainGMM(base_path, retrain=False):
    """Training GMM
    Load pre-trained GMM or train GMM with training data.
    """
    gmm_trained = None
    stored_gmm_file = base_path + "/" + "StoredModel.jbl"
    # Checking if file exists
    try:
        os.path.isfile(base_path)
    except:
        pass
    else:
        gmm_trained = joblib.load(stored_gmm_file)

    if gmm_trained == None or retrain == True:
        image_files = mmScanPath(base_path)

    return gmm_trained


def getUnmixedData(file_path, CROP, ref_data, ref_channel, object_channel=0):
    """Get Unmixed Data

    """
    # Load and read image set
    image_set = ImageSet(file_path)
    image_data = image_set.readSet()
    # Crop image set
    image_data = image_data[:, CROP[0]:CROP[1], CROP[2]:CROP[3]]

    objects = Objects(image_data[object_channel])
    labels, labels_annulus, circles_dim = objects.findObjects()
    unmixed = unmix(ref_data, image_data[1:, :, :])
    median_data = getSpectralMedianIntensities(labels, unmixed)
    ratio_channels = range(median_data[0, :].size)
    ratio_channels.remove(ref_channel)      # Reference channel
    ratio_channels.remove(object_channel)   # Object (brightfield) channel
    ratio_data = getRatios(
        labels, unmixed[ratio_channels], unmixed[ref_channel])
    # Clean up objects
    del image_set
    del objects
    return labels, labels_annulus, circles_dim, median_data, ratio_data


def multiImageSetData(base_path):
    """Multi Image Set
    Load multiple image sets from base path(s) recursively.
    """
    if isinstance(base_path, basekeyword):
        image_files = mmScanPath(base_path)
    elif len(base_path) > 1:
        image_files = map(mmScanPath, base_path)
    else:
        print("Can't resolve base path(s).")

class ImageSet(object):
    """Image Set
    Load image set from file(s)
    file_path = File path(s) in list [path, path]
    """
    def __init__(self, file_path):
        """Initialize Bioformats & Java and set properties"""
        # Initiate JAVA environment and basic logging for image reader
        self.loadJVE(heap_size='8G')
        # Load image reader for images
        self._image_reader, self.metadata = self.loadImageReader(file_path)

        # Extract set information from metadata
        self.sizeC = self.getMetaDataNumber("SizeC", self.metadata)  # No. channels
        self.sizeT = self.getMetaDataNumber("SizeT", self.metadata)  # No. timepoints
        self.sizeZ = self.getMetaDataNumber("SizeZ", self.metadata)  # No. Z slices
        # TO-DO self.sizeP = self.getMetaDataNumber("SizeP", self.metadata)  # No. positions
        self.sizeI = self.sizeC * self.sizeT * self.sizeZ  # No. images total        
        self.arrayOrder = self.identifyOrder()
        # Load initial image to get dimensions
        init_image = self.readImage(idx=0)
        self.imageY = init_image[:,0].size  # Y dim first, since columns
        self.imageX = init_image[0,:].size  # X dim second, since columns
        del init_image  # Free up memory

    def __close__(self):
        """Destructor of ImageSet"""
        self._image_reader.close()
        return 0

    @staticmethod
    def loadJVE(heap_size='2G'):
        """Load JVE
        Initiate JAVA environment and basic logging
        heap_size = Maximum size of JAVA heap, e.g. '2G' or '2024M'
        """
        jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)
        log4j.basic_config()

    @staticmethod
    def loadImageReader(file_path):
        """Initialize Bioformats reader and get metadata"""
        # Getting metadata and load image reader
        try:
            os.path.isfile(file_path)
        except IOError as io:
            print("Cannot open file:", file_path)
        except:
            print("Unexpected error:", sys.exc_info())
        else:
            image_reader = bf.ImageReader(file_path)
            metadata = bf.get_omexml_metadata(file_path)            
            return image_reader, metadata

    @staticmethod
    def getMetaDataNumber(search_keyword, metadata):
        """Extract Metadata
        Extract from bioformats metadata the number after = and between "" following the given keyword.
        search_keyword = Keyword to be searched for: "keyword"
        """
        search_string = search_keyword + r'=\"\d+\"'
        found_string = re.findall(search_string, metadata)
        if len(found_string) >= 1:
            extracted_number = int(re.findall(r'\d+', found_string[0])[0])
        else:
            extracted_number = None
        return extracted_number

    def identifyOrder(self):
        """Identify Order
        Identify the order of the image set
        """
        if self.sizeT > 1 and self.sizeC > 1:
            return "[t,c,y,x]"
        elif self.sizeT == 1 and self.sizeC > 1:
            return "[c,y,x]"
        elif self.sizeT > 1 and self.sizeC == 1:
            return "[t,y,x]"
        elif self.sizeT == 1 and self.sizeC == 1:
            return "[y,x]"
        else:
            return None

    def getIndex(self, c=0, t=0, idx=0):
        """Get Index Number
        Return index number for given channel and/or timepoint
        c = Channel number starting with 0
        t = Timepoint starting with 0
        """
        if c > 0 and c < self.sizeC or t > 0 and t < self.sizeT:
            index = c + (t * self.sizeC)
            return index
        elif c >= self.sizeC or t >= self.sizeT:
            raise IndexError
        else: return idx

    def readImage(self, idx=0, c=0, t=0, rescale=False):
        """Read Image
        Read and return single image from image set
        c = Channel number starting with 0
        t = Timepoint starting with 0
        idx = Index number starting with 0
        """
        image = self._image_reader.read(
            index=self.getIndex(c=c, t=t, idx=idx), rescale=rescale)
        return image

    def readSet(self, idx=None, c=None, t=None, rescale=False):
        """Read Set
        Read defined image set and return data array
        """
        # Check array order
        if self.arrayOrder == "[y,x]" or "":
            print("Sigle image or not an image set:", self.arrayOrder)
            raise IOError
        
        # Set length timepoints and channels
        timepoints = xrange(self.sizeT)
        channels = xrange(self.sizeC) 
        # Iterate over timepoints and channels
        if self.arrayOrder == "[t,c,y,x]":
            image_set = np.array( [np.array( [self.readImage(c=ch, t=tp, rescale=rescale) for ch in channels] ) for tp in timepoints] )
        elif self.arrayOrder == "[c,y,x]":
            image_set = np.array( [self.readImage(c=ch, rescale=rescale) for ch in channels] )
        elif self.arrayOrder == "[t,y,x]":
            image_set = np.array( [self.readImage(t=tp, rescale=rescale) for tp in timepoints] )
        return image_set

class Objects(object):
    """Objects
    Identify objects from image and store
    """
    def __init__(self, image):
        """
        Initialization after instantiation
        Set local variables
        """
        # Check and/or convert image to 8 bit array. This is required for
        # object search
        self.image = self.imageConvert(image)
        self.labeled_mask = None
        self.labeled_annulus_mask = None
        self.circles_dim = None

    def __close__(self):
        """Destructor of Objects"""
        return 0

    @staticmethod
    def imageConvert(image):
        """8 Bit Convert
        Checks image data type and converts if necessary to uint8 array.
        image = M x N image array
        """
        try:
            img_type = image.dtype
        except IOError:
            print("Not a NumPy array of image", image)
        except:
            print("Unexpected error:", sys.exc_info())
        else:
            if img_type == 'uint8':
                return image
            else:
                image = np.array(((image / 2**16) * 2**8), dtype='uint8')
                return image

    def findObjects(self, image=None, sep_min_dist=2, min_dist=None, param_1=20, param_2=9, min_r=3, max_r=6, ring_size=2):
        """Find Objects
        Find objects in image and return data
        """
        # Check if image is set, if not use initial image
        if image == None: img = self.image
        else: img = self.imageConvert(image)
        
        # Check if min_dist is set and set to 2 x minimuj radius
        if min_dist == None: min_dist = 2 * min_r
        # Find initial circles using Hough transform and make mask
        mask = self.makeCircleMask(min_dist=min_dist, param_1=param_1, param_2=param_2, min_r=min_r, max_r=max_r)
        # Find and separate circles using watershed on initial mask
        labels = self.separateCircles(mask)
        # Find center of circle and return dimensions
        circles_dim = self.getCircleDimensions(labels)        
        
        # Create annulus mask
        labels_annulus = labels.copy()  
        for cd in circles_dim:
            cv2.circle(labels_annulus, (int(cd[0]), int(cd[1])),
                       int(cd[2] - ring_size), (0, 0, 0), -1)
        self.labeled_mask = labels
        self.labeled_annulus_mask = labels_annulus
        self.circles_dim = circles_dim
        return labels, labels_annulus, circles_dim

    def makeCircleMask(self, image = None, min_dist=None, param_1=20, param_2=9, min_r=3, max_r=6):
        """Make Circle Mask
        """
        # Check if image is set, if not use initial image
        if image == None: img = self.image
        else: img = self.imageConvert(image)

        # Check if min_dist is set and set to 2 x minimuj radius
        if min_dist == None: min_dist = 2 * min_r
        # Find initial circles using Hough transform
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=param_1, param2=param_2, minRadius=min_r, maxRadius=max_r)[0]
        # Make mask
        mask = np.zeros(img.shape, np.uint8)
        for c in circles:
            x, y, r = c[0], c[1], int(np.ceil(c[2] + 0.1))
            # Draw circle (line width -1 fills circle)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        return mask

    def separateCircles(self, mask, sep_min_dist=2):
        """Separate Circles
        """
        # Find and separate circles using watershed on initial mask
        D = ndi.distance_transform_edt(mask)
        localMax = peak_local_max(D, indices=False, 
                                  min_distance=sep_min_dist, 
                                  exclude_border=True, 
                                  labels=mask)
        markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=mask)
        print("Number of unique segments found: {}".format(
            len(np.unique(labels)) - 1))
        return labels

    def getCircleDimensions(self, labels, image = None):
        """
        Find center of circle and return dimensions
        """
        # Check if image is set, if not use initial image
        if image == None: img = self.image
        else: img = self.imageConvert(image)

        idx = np.arange(1, len(np.unique(labels)))
        circles_dim = np.empty((len(np.unique(labels)) - 1, 3))
        for label in idx:
            # Create single object mask
            mask_detect = np.zeros(img.shape, dtype="uint8")
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

    def overlayImage(self, dim, img=None, ring_size=0):
        """Overlay Image
        Overlay image with circles of labeled mask
        """
        # Check if image is set, if not a copy is made. Numpy array namespaces
        # are memory locators. If no copy is made the original data is
        # manipulated.
        if img == None:
            img = self.image.copy()
        for d in dim:
            if ring_size > 0:
                cv2.circle(img, (int(d[0]), int(d[1])), int(
                    d[2] - ring_size), (0, 255, 0), 1)
            cv2.circle(img, (int(d[0]), int(d[1])), int(d[2]), (0, 255, 0), 1)
        return img

class RefSpec(object):
    """Reference Spectra
    Generate reference spectra
    """
    def __init__(self, image_files, crop = [100, 400, 100, 400], size_param = [3, 9, 10, 10, 7, 10]):
        """
        Initialization after instantiation and set local variables
        """
        self.image_files = image_files
        self.crop = crop
        self.ref_spec_set = None
        self.objects = None
        self.size_param = size_param

    def __close__(self):
        """Destructor of RefSpec"""
        return 0

    def readSpectra(self):
        """Read Spectra
        """
        ref_spec_set = np.array( [self.readSpectrum(file, 0,  self.size_param[idx], crop = self.crop) for idx, file in enumerate(self.image_files)] )
        self.ref_spec_set = ref_spec_set
        return ref_spec_set.T

    def readSpectrum(self, file, object_channel, size_param = [3, 9, 10, 10, 7, 10], crop = None):
        """Read Spectrum
        """
        if size_param is None:
            size_param = [3, 9, 10, 10, 7, 10]
        if crop == None: crop = self.crop
        ref_set = ImageSet(file)
        ref_set_data = ref_set.readSet()[:, crop[0]:crop[1], crop[2]:crop[3]]
        objects = self.getRefObjects(ref_set_data[object_channel], 
                                     sep_min_dist=size_param[0], min_dist=size_param[1], 
                                     param_1=size_param[2], param_2=size_param[3], 
                                     min_r=size_param[4], max_r=size_param[5])
        channels = range(ref_set_data[:, 0, 0].size)
        channels.remove(object_channel)
        ref_data = self.getRef(ref_set_data[channels])
        return ref_data

    def getRefObjects(self, object_image, sep_min_dist=3, min_dist=9, param_1=10, param_2=10, min_r=7, max_r=10):
        """Get Reference Objects
        """
        objects = Objects(object_image)
        labels, labels_annulus, circles_dim = objects.findObjects(
            sep_min_dist=sep_min_dist, min_dist=min_dist, 
            param_1=param_1, param_2=param_2, min_r=min_r, max_r=max_r)
        self.objects = labels
        return labels

    def getRef(self, image_data, back = 451):
        """Get Reference
        Get reference spectra from image set
        """
        channels = range(image_data[:, 0, 0].size)
        ref_data = np.array( [self.getMedianObjects(image_data[ch], self.objects) for ch in channels], dtype="float64" )
        ref_data = ref_data - back
        sum = ref_data.sum()
        return np.divide(ref_data, sum)

    def getMedianObjects(self, image_data, objects):
        """Get Median Objects"""
        data = ndi.median(image_data, objects)
        return data

    def getBack(image_data, square):
        """Get Background
        Get background reference of specified area
        image_data = single image used for background
        square = coordinates of region of interest [Y1, Y2, X1, X2]
        """
        c_size = image_data[:, 0, 0].size - 1
        channels = xrange(1, c_size + 1)
        ref_data = np.empty((c_size), dtype="float64")
        for ch in channels:
            img_tmp = image_data[ch, square[0]:square[1], square[2]:square[3]]
            ref_data[ch - 1] = np.median(img_tmp)
        sum = ref_data.sum()
        return np.divide(ref_data, sum)

class ICP(object):
    """Iterative Closest Points
    """
    def __init__(self):
        """
        Initialization after instantiation and set local variables
        """

    @staticmethod
    def matrix_func(naxes, func, input1, input2):
        """Matrix
        Create identity matrix and replace 1 values with function e.g np.mean
        naxes = number
        func = function
        input1 = matrix 1
        input2 = matrix 2
        """
        matrix = np.eye(naxes)
        for n in xrange(naxes):
            matrix[n, n] = func(input1[:,n]) / func(input2[:,n])
        return matrix

    def matrix_mean(self, naxes, input1, input2):
        func = np.mean
        matrix = self.matrix_func(naxes, func, input1, input2)
        return matrix

    @staticmethod
    def icp(data, target, max_iter=100, tol=1e-4, input_matrix=None, input_offset=None):
        """ICP
        Iterative Closest Point
        """
        nLn = len(data[0, :])
        nData = len(data[:, 0])
        if input_offset is None:
            offset = np.ones(nLn)
            for n in xrange(nLn):
                offset[n] = np.min(target[:, n]) - np.min(data[:, n])

        if input_matrix is None:
            matrix = np.eye(nLn)
            for n in xrange(nLn):
                matrix[n, n] = (np.mean(target[:, n])) / np.mean(data[:, n])
        else:
            matrix = input_matrix

        delta = 1
        for i in xrange(max_iter):
            if delta < tol:
                print("Converged after:", i)
                break

            # Copy old to compare to new
            matrix_old = matrix
            offset_old = offset
            result = np.dot(data, matrix) + offset

            distances = metrics.pairwise.pairwise_distances(result, target)
            matched_code = np.argmin(distances, axis=1)
            matched_levels = target[matched_code, :]
            d = np.c_[data, np.ones(len(data[:, 0]))]
            m = np.linalg.lstsq(d, matched_levels)[0]
            matrix = m[0:-1, :]
            offset = m[-1, :]

            d_compare = np.sum(np.square(np.subtract(matrix, matrix_old)))
            d_compare = d_compare + np.sum(np.square(np.subtract(offset, offset_old)))
            n_compare = np.sum(np.square(matrix)) + np.sum(np.square(offset))
            delta = sqrt(d_compare / n_compare)
            print("Delta: ", delta)
        return result

class DecodeBeads(object):
    """Decode Beads
    Decode beads based on reference spectra and image set
    """
    def __init__(self, objects, reference_spectra):
        """
        Initialization after instantiation and set local variables
        """

    def __close__(self):
        """Destructor of DecodeBeads"""
        return 0
    
    def unmix(ref_data, image_data):
        """Unmix
        Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm images to Dy, Sm and Tm nanophospohorous lanthanides using reference spectra for each dye.
        ref_data = Reference spectra for each dye channel as Numpy Array: N x M, where N are the spectral channels and M the dye channels 
        image_data = Spectral images as NumPy array: N x M x P, where N are the spectral channels and M x P the image pixels (Y x X)
        """
        # Check if inputs are NumPy arrays and check if arrays have equal channel sizes
        try:
            ref_shape = ref_data.shape
            img_shape = image_data.shape
        except IOError:
            print("Input not NumPy array")
        if ref_shape[0] != img_shape[0]:
            print("Number of channels not equal. Ref: ", ref_shape, " Image: ", img_shape)
            raise IndexError
        c_size = image_data[:, 0, 0].size
        y_size = image_data[0, :, 0].size
        x_size = image_data[0, 0, :].size
        ref_size = ref_data[0, :].size
        img_flat = image_data.reshape(c_size, (y_size * x_size))
        unmix_flat = np.linalg.lstsq(ref_data, img_flat)[0]
        unmix_result = unmix_flat.reshape(ref_size, y_size, x_size)
        return unmix_result

    def getRatios(labels, images, reference):
        """Get Ratios
        Get median ratio of each object.
        """
        idx = np.arange(1, len(np.unique(labels)))
        data_size = len(np.unique(labels)) - 1
        channel_no = images[:, 0, 0].size
        channels = xrange(channel_no)
        ratio_data = np.empty((data_size, channel_no))
        for ch in channels:
            # Get pixel-by-pixel ratios
            image_tmp = np.divide(images[ch, :, :], reference)
            # Get median ratio of each object
            ratio_data[:, ch] = ndi.labeled_comprehension(
                image_tmp, labels, idx, np.median, float, -1)
        return ratio_data