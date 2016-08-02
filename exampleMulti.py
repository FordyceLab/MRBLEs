# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : main.py               [filename]
# description       : Main file to call and use BeadKinetics module
# author            : Bjorn Harink          [Original author(s) of this file]
# credits           : Kurt Thorn, Huy Nguyen[Contributors to this file]
# date              : 20160308              [Initial date yyyymmdd]
# version update    : 20160504              [Last version update yyyymmdd]
# version           : v0.3
# usage             : As startup file
# notes             : This is an example file for the module
# python_version    : 2.7

# [Main header with project metadata] | Only in the main file!
# Copyright and credits
__copyright__   = ("Copyright 2016 - "
                   "The Encoded Beads Project - "
                   "ThornLab@UCSF and "
                   "FordyceLab@Stanford")
# Original author(s) of this Python project, like: ("...", 
__author__      = ("Bjorn Harink")  #        "name")
# People who contributed to this Python project, like: ["...",
__credits__     = ["Kurt Thorn",  #              "name"] 
                   "Huy Nguyen"]
# Maintainer contact information
__maintainer__  = "Bjorn Harink" 
__email__       = "bjorn@harink.info" 
# Software information
__license__     = "MIT" 
__version__     = "v0.3"
__status__      = "Prototype"

# [TO-DO]

# [Modules]
# General
import sys
# Math
import numpy as np
from scipy import ndimage as ndi
# Machine learning
from sklearn import mixture
# Image disply
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# Project
import bead_analysis as ba

# [Settings]
# Reference image location
#REF_FILES = {"CeTb": "Z:/Huy/20150706/20150709_CeTbSolo_1/20150709_CeTbSolo_1_MMStack_Pos0.ome.tif",
#             "Dy": "Z:/Huy/20150706/20150709_DySolo_1/20150709_DySolo_1_MMStack_Pos0.ome.tif",
#             "Sm": "Z:/Huy/20150706/20150709_SmSolo_3/20150709_SmSolo_3_MMStack_Pos0.ome.tif",
#             "Tm": "Z:/Huy/20150706/20150709_TmSolo_3/20150709_TmSolo_3_MMStack_Pos0.ome.tif",
#             "Eu": "Z:/Huy/20150706/20150709_EuSolo_1/20150709_EuSolo_1_MMStack_Pos0.ome.tif",
#             "Trp": "Z:/Huy/20160314_WBlank/20160314_W_blank_4/20160314_W_blank_4_MMStack.ome.tif"}

REF_FILES = ["Z:/Huy/20160314_WBlank/20160314_W_blank_4/20160314_W_blank_4_MMStack.ome.tif",    # Trp
             "Z:/Huy/20150706/20150709_DySolo_1/20150709_DySolo_1_MMStack_Pos0.ome.tif",        # Dy
             "Z:/Huy/20150706/20150709_SmSolo_3/20150709_SmSolo_3_MMStack_Pos0.ome.tif",        # Sm
             "Z:/Huy/20150706/20150709_TmSolo_3/20150709_TmSolo_3_MMStack_Pos0.ome.tif",        # Tm
             "Z:/Huy/20150706/20150709_EuSolo_1/20150709_EuSolo_1_MMStack_Pos0.ome.tif"]        # Eu

# Target file location
TARGET_FILE = "Z:/Code Sets/20150714_DySmTm_48codes.csv"
#TARGET_FILE = "Z:/Code Sets/20160226_DySmTm_48Codes.csv"

# Training set location
TRAIN_PATH = "Z:/Huy/20160225"
TRAIN_BACK = "Z:/Huy/20160225/20160226_48codes_1to48Test_1/20160226_48codes_1to48Test_1_MMStack.ome.tif"
# [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
TRAIN_BACK_CROP = [280, 300, 230, 270]

# Image set channels
IMAGE_CHANNELS = {"BF": 1,
                  "435": 2,
                  "474": 3,
                  "527": 4,
                  "536": 5,
                  "546": 6,
                  "572": 7,
                  "620": 8,
                  "630": 9,
                  "650": 10,
                  "Cy3": 11,
                  "Cy5": 12,
                  "FITC": 13}

# General Region or interest
# [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
CROP = [70, 430, 70, 430]

### Main function ###
def main():
    print(__copyright__)

    # Set target file of code set and choose targets
    target = np.genfromtxt(TARGET_FILE, delimiter=',')
    target = target[:, 1:]  # Target set to Dy, Sm and Tm (no CeTb)

    ##################################################
    # Get reference spectra of solo lanthanide beads #
    ##################################################

    print("Create reference spectra")
    size_parameters = [[3, 4, 10 ,11, 2, 5], None, None, None, None]
    ref_data_object = ba.RefSpec(REF_FILES, size_param = size_parameters)
    ref_data = ref_data_object.readSpectra()
    
    ##################################################
    # Get training data and check clustering         #
    ##################################################    
    
    print("Generate training data")
    # Get all image files of training set
    image_files = ba.ImageSet.scanPath(TRAIN_PATH)
    # Background spectrum of training set
    image_set = ba.ImageSet(TRAIN_BACK)
    image_data_set = image_set.readSet()
    ref_data[:, 0] = ba.getBack(image_data_set, TRAIN_BACK_CROP)
    del image_set

    # Data all
    raw_data = []
    data_ratios = []
    data_circles_dim = []

    # Run on all found files
    def run_paths(ipath):
        labels, labels_annulus, circles_dim, median_data, ratio_data = ba.getUnmixedData(
            ipath, CROP, ref_data, 5)
        raw_data.append(median_data)
        data_ratios.append(ratio_data)
        data_circles_dim.append(circles_dim)
    map(run_paths, image_files)
    
    fig = plt.figure()
    fig.suptitle("Reference Spectra  - Training Set")
    plt.plot(ref_data)
    plt.draw()

    rflatten_1 = [item for sublist in raw_data for item in sublist]
    data_raw = np.array(rflatten_1)

    rat_flatten_1 = [item for sublist in data_ratios for item in sublist]
    ratio_data = np.array(rat_flatten_1)

    c_flatten_1 = [item for sublist in data_circles_dim for item in sublist]
    circles_dim = np.array(c_flatten_1)

    background = data_raw[:, 0]  # Device background
    reference = data_raw[:, 5]   # Internal reference: Eu

    # Filter objects based on background and reference
    data_filter_list = ba.filterObjects(ratio_data, background, reference, circles_dim[:, 2])
    # Omit CeTb and Trp ratios
    data_filtered = ratio_data[data_filter_list, 1:4]
    circles_dim_filt = circles_dim[data_filter_list]

    # Iterative closest point
    data_icp = ba.icp(data_filtered, target, tol=1e-4, max_iter=10)

    # Gaussian Mixture Modeling
    nclusters = len(target[:, 0])
    naxes = len(target[0, :])
    sigma = np.eye(naxes) * 1e-5
    gmix = mixture.GMM(n_components=nclusters, covariance_type='full',
                       min_covar=1e-7, tol=1e-7, init_params='', params='wmc')
    gmix.means_ = target
    gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
    gmix.weights_ = np.tile(1 / 48, (nclusters))
    gmix.fit(data_icp, target)
    predict = gmix.predict(data_icp)

    np.set_printoptions(threshold=1e4)
    print("Number of unique beads found:", len(np.unique(predict)))
    colors = np.multiply(predict, 5)

    fig = plt.figure()
    fig.suptitle("Clustering pre-ICP - Training Set")
    ax = fig.gca(projection='3d')
    ax.scatter(data_filtered[:, 0], data_filtered[:, 1],
               data_filtered[:, 2], c=colors, alpha=0.8)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], alpha=0.8)
    plt.draw()

    fig = plt.figure()
    fig.suptitle("Clustering post-ICP - Training Set")
    ax = fig.gca(projection='3d')
    ax.scatter(data_icp[:, 0], data_icp[:, 1],
               data_icp[:, 2], c=colors, alpha=0.8)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], alpha=0.8)
    plt.draw()

    plt.show()

    return 0

### Main loop ###
if __name__ == '__main__':
    status = main()
    sys.exit(status)