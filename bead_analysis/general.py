# !/usr/bin/env python

### Future imports
from __future__ import print_function   # "print" function compatibility between Python 2.x and 3.x
from __future__ import division         # Use Python 3.x "/" for division in Pyhton 2.x

### File header     | Copy and edit for each file in this project!
# title             : BeadKinetics.py       [filename]
# description       : Bead Kinetics module
# author            : Bjorn Harink          [Original author/starter of this file]
# date              : 20160308              [Initial date yyyymmdd]
# last update       : 20160502              [Last update yyyymmdd]
# version           : v0.2
# usage             : As module
# notes             : 
# python_version    : 2.7

### TO-DO

### Modules
# General
import sys
import re
import os
import glob
# Math
import numpy as np
import math
# Image import
import bioformats as bf
from bioformats import log4j
import javabridge as jb # Used for bioformats
# Image processing
import cv2
from scipy import ndimage as ndi
import sklearn
from sklearn import metrics
from sklearn import mixture
from sklearn import preprocessing
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
# Image display
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# File processing
from sklearn.externals import joblib


### Settings


### Main functions and classes
## General functions

## Software-package specific functions
def mmScanPath(path, pattern=None):
    """Micro-Manager Scan Path
    Scan a Micro-Manager images directory structure
    path = Micro-Manager image path
    """
    if pattern != None:
        pass
    else:
        pattern = ".tif"

    files_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(pattern):
                 files_list.append(os.path.join(root, file))
    return files_list

def getRef(image_data, back = 0, sep_min_dist = 3, min_dist = 9, param_1 = 10, param_2 = 10, min_r = 7, max_r = 10):
    """Get Reference
    Get reference spectra from image set
    """
    # Object image must be 0
    objects = Objects(image_data[0])
    labels, labels_annulus, circles_dim = objects.findObjects(sep_min_dist = sep_min_dist, min_dist = min_dist, param_1 = param_1, param_2 = param_2, min_r = min_r, max_r = max_r)
    del objects
    c_size = image_data[:,0,0].size-1
    ref_data = np.empty((c_size), dtype="float64")
    channels = xrange(1, c_size+1)
    for ch in channels:
        data = ndi.median(image_data[ch], labels)
        ref_data[ch-1] = data - back # min background noise
    sum = ref_data.sum()
    return np.divide(ref_data, sum)

def getBack(image_data, square):
    c_size = image_data[:,0,0].size-1
    channels = xrange(1,c_size+1)
    ref_data = np.empty((c_size), dtype="float64")
    for ch in channels: 
        img_tmp = image_data[ch,square[0]:square[1],square[2]:square[3] ]
        ref_data[ch-1] = np.median(img_tmp)
    sum = ref_data.sum()
    return np.divide(ref_data, sum)

def unmix(ref_data, image_data):
    """Unmix
    Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm images to Dy, Sm and Tm nanophospohorous lanthanides using reference spectra for each dye.
    ref_data = Reference spectra for each dye channel as Numpy Array: N x M, where N are the spectral channels and M the dye channels 
    image_data = Spectral images as NumPy array: N x M x P, where N are the spectral channels and M x P the image pixels (Y x X)
    """
    #TD add array size check and select last 3 dimensions
    c_size = image_data[:,0,0].size
    y_size = image_data[0,:,0].size
    x_size = image_data[0,0,:].size
    ref_size = ref_data[0,:].size
    img_flat = image_data.reshape(c_size, (y_size*x_size))
    unmix_flat = np.linalg.lstsq(ref_data, img_flat)[0]
    unmix_result = unmix_flat.reshape(ref_size, y_size, x_size)
    
    return unmix_result
    
def checkArrayShape():
    """Array Shape
    """
    pass

def getSpectralMedianIntensities(labels, images):
    """Get Median Intensities of each object location from the given image.
    labels = Labeled mask of objects
    images = image set of spectral images
    """
    idx=np.arange(1, len(np.unique(labels)))
    data_size = len(np.unique(labels))-1
    channel_no = images[:,0,0].size
    channels = xrange(channel_no)
    medians_data = np.empty((data_size, channel_no))
    for ch in channels:
        # Get median value of each object
        medians_data[:,ch] = ndi.labeled_comprehension(images[ch,:,:], labels, idx, np.median, float, -1)
    return medians_data

def getRatios(labels, images, reference):
    """Get Ratios
    Get median ratio of each object.
    """
    idx=np.arange(1, len( np.unique(labels)))
    data_size = len(np.unique(labels))-1
    channel_no = images[:,0,0].size
    channels = xrange(channel_no)
    ratio_data = np.empty((data_size, channel_no))
    for ch in channels:
        # Get pixel-by-pixel ratios
        image_tmp = np.divide(images[ch,:,:], reference)
        # Get median ratio of each object
        ratio_data[:,ch] = ndi.labeled_comprehension(image_tmp, labels, idx, np.median, float, -1)
    return ratio_data

def icp(data, target, max_iter = 100, tol = 1e-3, input_matrix = None, input_offset = None):
    """ICP
    Iterative Closest Point
    """
    nLn = len(data[0,:])
    nData = len(data[:,0])
    if input_offset == None:
        offset = np.ones(nLn)
        for n in xrange(nLn):
            offset[n] = np.min(target[:,n]) - np.min(data[:,n])
        
    if input_matrix == None:
        matrix = np.eye(nLn)
        for n in xrange(nLn):
            matrix[n,n] = np.mean(target[:,n]) / np.mean(data[:,n])
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
        d = np.c_[data, np.ones(len(data[:,0]))]
        m = np.linalg.lstsq(d, matched_levels)[0]
        matrix = m[0:-1,:]
        offset = m[-1,:]
            
        d_compare = np.sum( np.square( np.subtract(matrix, matrix_old) ) )
        d_compare = d_compare + np.sum( np.square( np.subtract(offset, offset_old) ) )
        n_compare = np.sum(np.square(matrix)) + np.sum(np.square(offset))
        delta = math.sqrt(d_compare / n_compare)
        print("Delta: ", delta)
    return result

def filterObjects(data, back, reference, objects_radius, back_std_factor = 3, reference_std_factor = 2, radius_min = 3, radius_max = 6):
    """Filter Objects
    Filter objects using x times SD from mean
    back = background data
    reference = reference data
    back_std_factor = x times SD from mean
    reference_std_factor = x times SD from mean
    """
    # Pre-filtered number
    pre_filter_no = data[:,0].size
    
    # Mean and standard deviation of the background and the reference channel
    mean_reference = np.mean(reference)
    std_reference = np.std(reference)
    mean_back = np.mean(back)
    std_back = np.std(back)
    print(mean_reference, std_reference, mean_back, std_back)
    
    ## Find indices of objects within search parameters
    # Check which objects are within set radius
    size_filter = np.logical_and( objects_radius >= radius_min, objects_radius <= radius_max )
    # Check which objects are within x SD from mean background signal
    back_filter = np.logical_and( back < (mean_back + back_std_factor*std_back), back > (mean_back - back_std_factor*std_back) )
    # Check which objects are within x SD from mean reference signal
    refr_filter = np.logical_and( reference > (mean_reference - reference_std_factor*std_reference), reference < (mean_reference + reference_std_factor*std_reference) )
    # Create list of indices of filtered-in objects
    filter_list = np.argwhere( np.logical_and( size_filter, np.logical_and(back_filter, refr_filter) ) )[:,0]

    # Compare pre and post filtering object numbers
    post_filter_no = filter_list.size
    post_filter_per = int( ( (pre_filter_no - post_filter_no) / post_filter_no ) * 100 )
    print("Pre-filter no:", pre_filter_no, ", Post-filter no:", post_filter_no, ", Filtered:", post_filter_per, "%")
    
    # Return list of indices of filtered-in objects
    return filter_list

def trainGMM(base_path, retrain = False):
    """Training GMM
    Load and load pre-trained GMM or train GMM with training data.
    """
    gmm_trained = None
    stored_gmm_file = base_path + "\\" + "StoredModel.jbl"
    # Checking if file exists
    try:
        os.path.isfile(base_path)
    except:
        pass
    else:
        gmm_trained = joblib.load(stored_gmm_file)

    if gmm_trained == None or retrain == True:
        image_files = bk.mmScanPath(base_path)

        
    return gmm_trained

def getUnmixedData(file_path, CROP, ref_data, ref_channel, object_channel = 0):
    """Get Unmixed Data
    
    """
    # Load and read image set
    image_set = ImageSet(file_path)
    image_data = image_set.readSet()
    # Crop image set
    image_data = image_data[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    
    objects = Objects(image_data[object_channel])
    labels, labels_annulus, circles_dim = objects.findObjects()
    unmixed =  unmix(ref_data, image_data[1:,:,:])
    median_data = getSpectralMedianIntensities(labels, unmixed)
    ratio_channels = range(median_data[0,:].size)
    ratio_channels.remove(ref_channel)      # Reference channel
    ratio_channels.remove(object_channel)   # Object (brightfield) channel
    ratio_data = getRatios(labels, unmixed[ratio_channels], unmixed[ref_channel])
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

## Data and handling classes
class ImageSet(object):
    """Image Set
    Load image set from file(s)
    file_path = File path(s) in list [path, path]
    """  
    def __init__(self, file_path):
        """Initialize Bioformats & Java and set properties"""
        # Properties
        self.file_path = file_path
        self.imageX = 0
        self.imageY = 0
        self.sizeC = 0
        self.sizeT = 0
        self.sizeI = 0
        self.image_reader = None
        self.init_image = None
        
        # Initiate JAVA environment and basic logging
        self.loadJVE(heap_size = '8G')
        # Getting metadata and load image reader
        self.loadImageReader()
        # Extracting data from metadata and set properties
        self.extractMetaData()  

    def __close__(self):
        """Destructor of ImageSet"""
        self.image_reader.close()
        return 0
        
    @staticmethod
    def loadJVE(heap_size = '8G'):
        """Load JVE
        Initiate JAVA environment and basic logging
        heap_size = Maximum size of JAVA heap, e.g. '8G' or '8096M'
        """
        jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)
        log4j.basic_config()
    
    def loadImageReader(self):
        """Initialize Bioformats reader and get metadata"""
        # Getting metadata and load image reader
        try:
            os.path.isfile(self.file_path)
        except IOError as io:
            print("Cannot open file:", self.file_path)
        except:
            print("Unexpected error:", sys.exc_info())
        else:
            self.metadata = bf.get_omexml_metadata(self.file_path)
            self.image_reader = bf.ImageReader(self.file_path)
            self.init_image = self.image_reader.read(index=0)

    def extractMetaData(self):
        """Extract Meta Data
        Extracting image set numbers from metadata
        """
        self.imageY = self.init_image[:,0].size # NumPy array of image is [Y, X]
        self.imageX = self.init_image[0,:].size # NumPy array of image is [Y, X]
        self.sizeC = self.getSizeC()
        self.sizeT = self.getSizeT()
        self.sizeI = self.sizeC * self.sizeT

    @staticmethod
    def getMetaDataNumber(search_keyword, metadata):
        """Extract Metadata
        Extract from metadata the number after = and between "" following the given keyword.
        search_keyword = Keyword to be searched for: "keyword"
        """
        search_string = search_keyword + r'=\"\d+\"'
        found_string = re.findall(search_string, metadata)
        extracted_number = int( re.findall(r'\d+', found_string[0])[0] )
        return extracted_number
        
    def getSizeC(self):
        """Get Size Channels
        Return number of channels
        """
        size = self.getMetaDataNumber("SizeC", self.metadata)
        return size

    def getSizeT(self):
        """Get Size Timepoints
        Return the number of timepoints
        """
        size = self.getMetaDataNumber("SizeT", self.metadata)
        return size

    def getIndex(self, c=0, t=0):
        """Get Index Number
        Return index number for given channel and/or timepoint
        c = Channel number starting with 0
        t = Timepoint starting with 0
        """
        if c >= 0 and c <= self.sizeC and t >= 0 and t <= self.sizeT:
            index = c + (t * self.sizeC)
            return index
        else:
            return 0

    def readImage(self, idx=None, c=None, t=None, rescale = False):
        """Read Image
        Read and return single image from image set
        c = Channel number starting with 0
        t = Timepoint starting with 0
        idx = Index number starting with 0
        """
        if c == None and t == None and idx >= 0:
            return self.image_reader.read(index=idx, rescale = rescale)
        elif c >= 0 or t >= 0 and idx == None:
            if c == None: c = 0
            if t == None: t = 0
            return self.image_reader.read(index=self.getIndex(c=c,t=t), rescale = rescale)
        else:
            return 0

    def readSet(self, idx=None, c=None, t=None, rescale = False):
        """Read Set
        Read defined image set and return data array
        """
        #TD# convert to map() and deconstruct function
        try:
            idx_len = len(idx)
        except:
            idx_len = 1 
        if c == None and t == None and idx == None:
            if self.sizeT > 1 and self.sizeC > 1:
                image_set = np.empty( (self.sizeT, self.sizeC, self.imageY, self.imageX))
                time_points = xrange(0, self.sizeT)
                channels = xrange(0, self.sizeC)
                for tp in time_points:
                    for ch in channels:
                        image_set[tp,ch,:,:] = self.readImage(c=ch, t=tp, rescale = rescale)
            elif self.sizeT == 1 and self.sizeC > 1:
                image_set = np.empty( (self.sizeC, self.imageY, self.imageX))
                channels = xrange(0, self.sizeC)
                for ch in channels:
                    image_set[ch,:,:] = self.readImage(c=ch, rescale = rescale)
            elif self.sizeT > 1 and self.sizeC == 1:
                image_set = np.empty( (self.sizeT, self.imageY, self.imageX))
                time_points = xrange(0, self.sizeT)
                for tp in timepoints:
                    image_set[tp,:,:] = self.readImage(t=tp, rescale = rescale)
            else: return None
        elif c == None and t == None and idx_len > 1:
            image_set = np.empty((len(idx), self.imageY, self.imageX))
            for i in idx:
                image_set[i,:,:] = self.readImage(idx=i, rescale = rescale)
        elif c > 1 or t > 1 and idx == None:
            if c == None: 
                image_set = np.empty( (len(t), self.imageY, self.imageX))
                time_points = t
                for tp in time_points:
                    image_set[tp,:,:] = self.readImage(t=tp, rescale = rescale)
            elif t == None: 
                image_set = np.empty( (len(c), self.imageY, self.imageX))
                channels = c
                for ch in channels:
                        image_set[ch-1,:,:] = self.readImage(c=ch, t=tp, rescale = rescale)
            else:
                image_set = np.empty( (len(t), len(c), self.imageY, self.imageX))
                time_points = t
                channels = c
                for tp in time_points:
                    for ch in channels:
                        image_set[tp,ch,:,:] = self.readImage(c=ch, t=tp, rescale = rescale)
        else:
            image_set = self.readImage(idx=0, rescale = rescale)
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
        # Check and/or convert image to 8 bit array. This is required for object search
        self.image = self.imageConvert(image)

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
                image = np.array( ((image/2**16) * 2**8), dtype='uint8')
                return image

    def findObjects(self, image=None, sep_min_dist = 2, min_dist=None, param_1 = 20, param_2 = 9, min_r = 3, max_r = 6, ring_size = 2):
        """Find Objects
        Find objects in image and return data
        """
        # Check if image is set, if not use initial image
        if image == None: 
            img = self.image
        else: 
            img = self.imageConvert(image)
        # Set search parameters
        if min_dist == None: 
            min_dist = 2 * min_r # Minimal distance is the minimal diameter
            
        # Find initial circles using Hough transform
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=param_1, param2=param_2, minRadius=min_r, maxRadius=max_r)[0]
        # Make mask
        mask = np.zeros(img.shape,np.uint8)
        for c in circles:
            x, y, r = c[0], c[1], int(np.ceil(c[2]+0.1))
            # Draw circle and fill (-1)
            cv2.circle(mask, (x,y), r, (255,255,255), -1) # -1 fills circle

        # Find and separate circles using watershed on initial mask
        #thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        D = ndi.distance_transform_edt(mask)
        localMax = peak_local_max(D, indices=False, min_distance=sep_min_dist, exclude_border=True, labels=mask)
        markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=mask)
        labels_annulus = labels.copy() # Must make copy otherwise just reference to same memory posistion
        print("Number of unique segments found: {}".format(len(np.unique(labels)) - 1))

        idx = np.arange(1, len(np.unique(labels)))
        circles_dim = np.empty( (len( np.unique(labels)) - 1, 3) )
        for label in idx:
            # Create single object mask
            mask_detect = np.zeros(img.shape, dtype="uint8")
            mask_detect[labels == label] = 255
        
            # Detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask_detect.copy(), cv2.RETR_EXTERNAL,
	            cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)
           
            # Get circle dimensions
            ((x, y), r) = cv2.minEnclosingCircle(c)
            circles_dim[label-1,0] = x
            circles_dim[label-1,1] = y
            circles_dim[label-1,2] = r

            circles_dim = np.array(circles_dim)
            
            # Update annulus labels mask
            cv2.circle(labels_annulus, (int(x), int(y)), int(r-ring_size), (0, 0, 0), -1)

        return labels, labels_annulus, circles_dim
        
    def makeCircleMask(img):
        # Find initial circles using Hough transform
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=param_1, param2=param_2, minRadius=min_r, maxRadius=max_r)[0]
        # Make mask
        mask = np.zeros(img.shape,np.uint8)
        for c in circles:
            x, y, r = c[0], c[1], int(np.ceil(c[2]+0.1))
            # Draw circle (line width -1 fills circle)
            cv2.circle(mask, (x,y), r, (255,255,255), -1)
        
    def separateCircles():
        pass

    def overlayImage(self, dim, img=None, ring_size=0):
        """Overlay Image
        Overlay image with circles of labeled mask
        """
        # Check if image is set, if not a copy is made. Numpy array namespaces are memory locators. If no copy is made the original data is manipulated. 
        if img == None: img = self.image.copy()
        
        #label = 0
        for d in dim:
            if ring_size > 0:
                cv2.circle(img, (int(d[0]), int(d[1])), int(d[2]-ring_size), (0, 255, 0), 1)
            cv2.circle(img, (int(d[0]), int(d[1])), int(d[2]), (0, 255, 0), 1)
            #cv2.putText(img, str(label), (int(d[0]) - 10, int(d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
            #label = label + 1
        return img