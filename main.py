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
#TARGET_FILE = "Z:/Code Sets/20150714_DySmTm_48codes.csv"
TARGET_FILE = "Z:/Code Sets/20160226_DySmTm_48Codes.csv"

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

# Kinetics image set location
# Calcineurin
#CALC_PATH_ON_1 = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc_3/20160331 - Start Antibody Calc_3_MMStack.ome.tif"
CALC_PATH_ON_1 = "Z:/Huy/20160513/20160513_48B_BR0.5_1/20160513_48B_BR0.5_1_MMStack.ome.tif"
CALC_PATH_ON_S = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc - Seq_1/20160331 - Start Antibody Calc - Seq_1_MMStack.ome.tif"
CALC_PATH_OFF_1 = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc - Seq End 4hr - Off start_1/20160331 - Start Antibody Calc - Seq End 4hr - Off start_1_MMStack.ome.tif"
CALC_PATH_OFF_S = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc - Seq End 4hr - Off start_2/20160331 - Start Antibody Calc - Seq End 4hr - Off start_2_MMStack.ome.tif"
CALC_BACK = "Z:/Bjorn/20160321 - BeadReactor 0.4 - Calcineurin/BR04 - BR Background UV-Focussed_1/BR04 - BR Background UV-Focussed_1_MMStack.ome.tif"
# [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
CALC_BACK_CROP = [100, 400, 261, 273]

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
    
    #print("Generate training data")
    ## Get all image files of training set
    #image_files = ba.mmScanPath(TRAIN_PATH)
    ## Background spectrum of training set
    #image_set = ba.ImageSet(TRAIN_BACK)
    #image_data_set = image_set.readSet()
    #ref_data[:, 0] = ba.getBack(image_data_set, TRAIN_BACK_CROP)
    #del image_set

    ## Data all
    #raw_data = []
    #data_ratios = []
    #data_circles_dim = []

    ## Run on all found files
    #def run_paths(ipath):
    #    labels, labels_annulus, circles_dim, median_data, ratio_data = ba.getUnmixedData(
    #        ipath, CROP, ref_data, 5)
    #    raw_data.append(median_data)
    #    data_ratios.append(ratio_data)
    #    data_circles_dim.append(circles_dim)
    #map(run_paths, image_files)
    
    #fig = plt.figure()
    #fig.suptitle("Reference Spectra  - Training Set")
    #plt.plot(ref_data)
    #plt.draw()

    #rflatten_1 = [item for sublist in raw_data for item in sublist]
    #data_raw = np.array(rflatten_1)

    #rat_flatten_1 = [item for sublist in data_ratios for item in sublist]
    #ratio_data = np.array(rat_flatten_1)

    #c_flatten_1 = [item for sublist in data_circles_dim for item in sublist]
    #circles_dim = np.array(c_flatten_1)

    #background = data_raw[:, 0]  # Device background
    #reference = data_raw[:, 5]   # Internal reference: Eu

    ## Filter objects based on background and reference
    #data_filter_list = ba.filterObjects(ratio_data, background, reference, circles_dim[:, 2])
    ## Omit CeTb and Trp ratios
    #data_filtered = ratio_data[data_filter_list, 1:4]
    #circles_dim_filt = circles_dim[data_filter_list]

    ## Iterative closest point
    #data_icp = ba.icp(data_filtered, target, tol=1e-4, max_iter=10)

    ## Gaussian Mixture Modeling
    #nclusters = len(target[:, 0])
    #naxes = len(target[0, :])
    #sigma = np.eye(naxes) * 1e-5
    #gmix = mixture.GMM(n_components=nclusters, covariance_type='full',
    #                   min_covar=1e-7, tol=1e-7, init_params='', params='wmc')
    #gmix.means_ = target
    #gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
    #gmix.weights_ = np.tile(1 / 48, (nclusters))
    #gmix.fit(data_icp, target)
    #predict = gmix.predict(data_icp)

    #np.set_printoptions(threshold=1e4)
    #print("Number of unique beads found:", len(np.unique(predict)))
    #colors = np.multiply(predict, 5)

    #fig = plt.figure()
    #fig.suptitle("Clustering pre-ICP - Training Set")
    #ax = fig.gca(projection='3d')
    #ax.scatter(data_filtered[:, 0], data_filtered[:, 1],
    #           data_filtered[:, 2], c=colors, alpha=0.8)
    #ax.scatter(target[:, 0], target[:, 1], target[:, 2], alpha=0.8)
    #plt.draw()

    #fig = plt.figure()
    #fig.suptitle("Clustering post-ICP - Training Set")
    #ax = fig.gca(projection='3d')
    #ax.scatter(data_icp[:, 0], data_icp[:, 1],
    #           data_icp[:, 2], c=colors, alpha=0.8)
    #ax.scatter(target[:, 0], target[:, 1], target[:, 2], alpha=0.8)
    #plt.draw()

    ###################################################
    # Get kinetics data clustering and classification #
    ################################################### 

    print("Kinetics data")
    bead_no = 48
    
    calc_on = ba.ImageSet(CALC_PATH_ON_1)

    calc_on_set = calc_on.readSet()
    calc_on_set_object = calc_on_set[0]
    calc_on_set = calc_on_set[:, CROP[0]:CROP[1], CROP[2]:CROP[3]]
    calc_on_objects = ba.Objects(calc_on_set_object[CROP[0]:CROP[1], CROP[2]:CROP[3]])

    labels_on, labels_annulus_on, circles_dim_on = calc_on_objects.findObjects()

    ba_img_set = ba.ImageSet(CALC_BACK)
    ba_img_read = ba_img_set.readSet()
    ref_back = ba.getBack(ba_img_read[0:10], CALC_BACK_CROP)
    del ba_img_set
    #crop_area = [253, 274, 167, 171]  # 48B
    #crop_area = [11, 27, 128, 143]  # 48
    #ref_back = ba.getBack(calc_on_set[0:10], crop_area)
    
    ref_data = np.insert(ref_data, 0, ref_back, axis = 1)

    fig = plt.figure()
    fig.suptitle("Reference Spectra - Python")
    plt.plot(ref_data)
    plt.draw()

    ## Use Matlab refs
    ref_data = np.genfromtxt("ref_spectra.csv", delimiter=',').T
    ref_data[:,0] = ref_back

    fig = plt.figure()
    fig.suptitle("Reference Spectra - Matlab")
    plt.plot(ref_data)
    plt.draw()

    # Unmix images by least squares
    unmixed_on = ba.unmix(ref_data, calc_on_set[1:10, :, :])

    median_data_on = ba.getSpectralMedianIntensities(labels_on, unmixed_on)
    # Omit CeTb and Trp ratios
    ratio_data_on = ba.getRatios(labels_on, unmixed_on[2:5], unmixed_on[5])

    background_on = median_data_on[:, 0]  # Device background
    reference_on = median_data_on[:, 5]  # Internal reference: Eu

    # Filter objects based on background and reference
    data_filter_list_on = ba.filterObjects(ratio_data_on, background_on, reference_on, circles_dim_on[:, 2], back_std_factor = 1.5, reference_std_factor = 1)
    data_filtered_on = ratio_data_on[data_filter_list_on]
    circles_dim_on_filt = circles_dim_on[data_filter_list_on]

    fig = plt.figure()
    fig.suptitle("Overlay Image Pre-filter - Kinetics Set")
    plt.imshow(calc_on_objects.overlayImage(circles_dim_on))
    plt.draw()

    fig = plt.figure()
    fig.suptitle("Overlay Image Post-filter - Kinetics Set")
    plt.imshow(calc_on_objects.overlayImage(circles_dim_on_filt))
    plt.draw()

    # Iterative closest point
    d_size = data_filtered_on[:, 0].size

    # Initial guess matrix to prevent collapse
    nLn = len(data_filtered_on[0, :])
    matrix = np.eye(nLn)
    for n in xrange(nLn):
        matrix[n, n] = np.divide(
            np.max(target[:, n]), np.max(data_filtered_on[:, n]))
    #matrix[0, 0] = 5
    #matrix[1, 1] = 1.5
    #matrix[2, 2] = 5

    # Iterative Closest Point
    data_icp_on = ba.ICP.icp(data_filtered_on, target, input_matrix = matrix, tol=1e-4)
    

    # GMM Setup
    nclusters = len(target[:, 0])
    naxes = len(target[0, :])
    sigma = np.eye(naxes) * 1e-5

    # New GMM
    gmix_on = mixture.GMM(n_components=nclusters, covariance_type='full',
                          min_covar=1e-7, tol=1e-5, init_params='', params='wmc')
    gmix_on.means_ = target
    gmix_on.covars_ = np.tile(sigma, (nclusters, 1, 1))
    gmix_on.weights_ = np.tile(1 / 48, (nclusters))
    gmix_on.fit(data_icp_on, target)
    predict_on = gmix_on.predict(data_icp_on)

    # Re-use trained GMM
    #gmix.fit(data_icp_on, target)
    #predict_on = gmix.predict(data_icp_on)

    print("Number of unique beads found:", len(np.unique(predict_on)))
    colors_on = np.multiply(predict_on, 5)

    fig = plt.figure()
    fig.suptitle("Clustering pre-ICP - Kinetics Set")
    ax = fig.gca(projection='3d')
    ax.scatter(data_filtered_on[:, 0], data_filtered_on[
               :, 1], data_filtered_on[:, 2], c=colors_on, alpha=0.8)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], alpha=0.8)
    plt.draw()

    fig = plt.figure()
    fig.suptitle("Clustering post-ICP - Training Set")
    ax = fig.gca(projection='3d')
    ax.scatter(data_icp_on[:, 0], data_icp_on[:, 1],
               data_icp_on[:, 2], c=colors_on, alpha=0.8)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], alpha=0.8)
    plt.draw()


    ###################################################
    # Get kinetics data ON-Rate                       #
    ################################################### 
    #calc_on_s = ba.ImageSet(CALC_PATH_ON_S)
    #calc_on_s_set = calc_on_s.readSet()
    #calc_on_s_set = calc_on_s_set[:, :, CROP[0]:CROP[1], CROP[2]:CROP[3]]

    #t_images = xrange(0, calc_on_s.sizeT)
    #data_set = []
    #idx = np.arange(1, len(np.unique(labels_on)))
    #for i in t_images:
    #    # Median from data
    #    #data = ndi.labeled_comprehension(calc_on_s_set[i,2,:,:], 
    #    #                                 labels_on, idx, np.median, float, -1)
    #    # Mean from data
    #    data = ndi.mean(calc_on_s_set[i, 2, :, :],
    #                    labels=labels_annulus_on, index=idx)
    #    data_set.append(data)

    #data_set = np.array(data_set)
    #data_set = data_set[:, data_filter_list_on]
    #beads = xrange(0, bead_no - 1)
    ## Unmix kinetics data
    #bead_unmixed = np.empty((calc_on_s.sizeT, bead_no))
    #for t in t_images:
    #    data_array = data_set[t]
    #    for x in beads:
    #        where = np.argwhere(predict_on == x)
    #        bead_unmixed[t, x] = np.mean(data_array[where[:, 0]])

    ####################################################
    ## Get kinetics data OFF-Rate                      #
    #################################################### 
    ## Combine ON-rate with OFF-rate
    #bead_unmixed_all = bead_unmixed
    #data_set_all = data_set

    #calc_off_s = ba.ImageSet(CALC_PATH_OFF_S)
    #calc_off_s_set = calc_off_s.readSet()
    #calc_off_s_set = calc_off_s_set[:, :, CROP[0]:CROP[1], CROP[2]:CROP[3]]

    #t_images = xrange(0, calc_off_s.sizeT)

    #data_set = []
    #idx = np.arange(1, len(np.unique(labels_on)))
    #for i in t_images:
    #    #data = ndi.labeled_comprehension(calc_off_s_set[i,1,:,:], labels_on, idx, np.median, float, -1)
    #    data = ndi.mean(calc_off_s_set[i, 1, :, :],
    #                    labels=labels_annulus_on, index=idx)
    #    data_set.append(data)

    #data_set = np.array(data_set)
    #data_set = data_set[:, data_filter_list_on]

    #beads = xrange(0, bead_no - 1)

    ## Unmix kinetics data
    #bead_unmixed = np.empty((calc_off_s.sizeT, bead_no))
    #for t in t_images:
    #    data_array = data_set[t]
    #    for x in beads:
    #        where = np.argwhere(predict_on == x)
    #        bead_unmixed[t, x] = np.mean(data_array[where[:, 0]])

    ## Combine ON-rate with OFF-rate
    #bead_unmixed_all = np.append(bead_unmixed_all, bead_unmixed, axis=0)
    #bead_unmixed_all = np.subtract(bead_unmixed_all, np.min(bead_unmixed_all))
    #data_set_all = np.append(data_set_all, data_set, axis=0)
    #data_set_all = np.subtract(data_set_all, np.min(data_set_all))

    ## Plots
    ## X Split
    #x_s = circles_dim_on_filt[:, 0]
    #colors = []
    #min_c = np.min(x_s)
    #max_c = np.max(x_s)
    #for x in x_s:
    #    colors.append([(x - min_c) / (max_c - min_c), (x - min_c) /
    #                   (max_c - min_c), (x - min_c) / (max_c - min_c)])

    #fig = plt.figure()
    #fig.suptitle("Training Set - X-Coloring")
    #ax = fig.add_subplot(111)
    #ax.plot(data_set_all[:98])
    #[ax.lines[i].set_color(color) for i, color in enumerate(colors)]
    #plt.draw()
    
    ## Split in 3 sections to see distance variance
    ##x_filter_1 = np.argwhere(np.logical_and(x_s >= 0, x_s < 100))
    ##x_filter_2 = np.argwhere(np.logical_and(x_s >= 100, x_s < 200))
    ##x_filter_3 = np.argwhere(np.logical_and(x_s >= 200, x_s < 300))
    ##plt.figure()
    ##plt.plot(data_set_all[:98, x_filter_1[:,0]])
    ##plt.axis([0, 98, 0, 16000])
    ##plt.draw()
    ##plt.figure()
    ##plt.plot(data_set_all[:98, x_filter_2[:,0]])
    ##plt.axis([0, 98, 0, 16000])
    ##plt.draw()
    ##plt.figure()
    ##plt.plot(data_set_all[:98, x_filter_3[:,0]])
    ##plt.axis([0, 98, 0, 16000])
    ##plt.draw()

    ## Plot individual beads data
    #fig = plt.figure()
    #fig.suptitle("Training Set - All Beads")
    #plt.plot(data_set_all[:98])
    #plt.axis([0, 98, 0, 16000])
    #plt.draw()

    ## Plot unmixed codes data
    #fig = plt.figure()
    #fig.suptitle("Training Set - Unmixed")
    #plt.plot(bead_unmixed_all[:98])
    #plt.axis([0, 98, 0, 16000])
    #plt.draw()

    plt.show()

    return 0


### Main loop ###
if __name__ == '__main__':
    status = main()
    sys.exit(status)
