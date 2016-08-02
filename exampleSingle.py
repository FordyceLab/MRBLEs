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

# [TO-DO]

# [Modules]
# General
import sys
from itertools import cycle
import re
# Math
import numpy as np
from scipy import ndimage as ndi
# Machine learning
from sklearn import mixture
# Image disply
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import axes3d
import bioformats as bf

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
#TARGET_FILE = "Z:/Code Sets/Copy of 20160226_DySmTm_48Codes_adjustedAraH2.csv"

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
CALC_PATH_ON_1 = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc_3/20160331 - Start Antibody Calc_3_MMStack.ome.tif"
#CALC_PATH_ON_1 = "Z:/Huy/20160513/20160513_48B_BR0.5_1/20160513_48B_BR0.5_1_MMStack.ome.tif"
CALC_PATH_ON_S = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc - Seq_1/20160331 - Start Antibody Calc - Seq_1_MMStack.ome.tif"
CALC_PATH_OFF_1 = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc - Seq End 4hr - Off start_1/20160331 - Start Antibody Calc - Seq End 4hr - Off start_1_MMStack.ome.tif"
CALC_PATH_OFF_S = "Z:/Bjorn/20160331 - Calcineurin/20160331 - Start Antibody Calc - Seq End 4hr - Off start_2/20160331 - Start Antibody Calc - Seq End 4hr - Off start_2_MMStack.ome.tif"
CALC_BACK = "Z:/Bjorn/20160321 - BeadReactor 0.4 - Calcineurin/BR04 - BR Background UV-Focussed_1/BR04 - BR Background UV-Focussed_1_MMStack.ome.tif"
# [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
CALC_BACK_CROP = [100, 400, 261, 273]

#CALC_PATH_ON_1 = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/On-Rate-T0.ome.tif"
#CALC_PATH_ON_S = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/On-Rate.ome.tif"
#CALC_PATH_OFF_1 = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/Off-Rate-T0.ome.tif"
#CALC_PATH_OFF_S = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/Off-Rate.ome.tif"

#CALC_BACK = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/20160615_Empty_BR05_1/20160615_Empty_BR05_1_MMStack.ome.tif"
#CALC_BACK_CROP = [100, 400, 261, 273]

# General Region or interest
# [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
CROP = [70, 430, 70, 430]

### Main function ###
def main():
    print(ba.__copyright__)

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
    
    ###################################################
    # Get kinetics data clustering and classification #
    ################################################### 

    print("Kinetics data")
    bead_no = 48
    
    calc_on = ba.ImageSet(CALC_PATH_ON_1)

    calc_on_set = calc_on.readSet()
    calc_on_set = calc_on_set[:, CROP[0]:CROP[1], CROP[2]:CROP[3]]
    calc_on_objects = ba.Objects(calc_on_set[0])

    labels_on, labels_annulus_on, circles_dim_on = calc_on_objects.findObjects()

    ba_img_set = ba.ImageSet(CALC_BACK)
    xml = ba_img_set.metadata.replace(u'\xb5',u'micro')
    o = bf.omexml.OMEXML(xml)
    
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
    #ref_data = np.genfromtxt("ref_spectra.csv", delimiter=',').T
    #ref_data[:,0] = ref_back

    #fig = plt.figure()
    #fig.suptitle("Reference Spectra - Matlab")
    #plt.plot(ref_data)
    #plt.draw()

    # Unmix images by least squares
    unmixed_on = ba.unmix(ref_data, calc_on_set[1:10, :, :])

    median_data_on = ba.getSpectralMedianIntensities(labels_on, unmixed_on)
    # Omit CeTb and Trp ratios
    ratio_data_on = ba.getRatios(labels_on, unmixed_on[2:5], unmixed_on[5])

    background_on = median_data_on[:, 0]  # Device background
    reference_on = median_data_on[:, 5]  # Internal reference: Eu

    # Filter objects based on background and reference
    data_filter_list_on = ba.filterObjects(ratio_data_on, background_on, reference_on, circles_dim_on[:, 2], back_std_factor = 3, reference_std_factor = 1.2)
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

    # Iterative Closest Point
    icp_on = ba.ICP(matrix_method = 'max', max_iter=100, tol=1e-4)
    icp_on.fit(data_filtered_on, target)
    print(icp_on.matrix)
    print(icp_on.offset)
    data_icp_on = icp_on.transform(data_filtered_on)

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


    ####################################################
    ## Get kinetics data ON-Rate                       #
    #################################################### 
    calc_on_s = ba.ImageSet(CALC_PATH_ON_S)
    calc_on_s_set = calc_on_s.readSet()
    calc_on_s_set = calc_on_s_set[:, :, CROP[0]:CROP[1], CROP[2]:CROP[3]]

    t_images = xrange(0, calc_on_s.sizeT)
    data_set = []
    idx = np.arange(1, len(np.unique(labels_on)))
    for i in t_images:
        data = ndi.labeled_comprehension(calc_on_s_set[i,1,:,:], labels_annulus_on, idx, np.mean, float, -1)
        data_set.append(data)

    data_set = np.array(data_set)
    data_set = data_set[:, data_filter_list_on]

    #beads = xrange(0, bead_no - 1)
    beads = np.unique(predict_on)
    print(beads)

    # Unmix kinetics data
    bead_unmixed = np.empty((calc_on_s.sizeT, len(beads)))
    for t in t_images:
        data_array = data_set[t]
        for idx, val in np.ndenumerate(beads):
            where = np.argwhere(predict_on == val)
            bead_unmixed[t, idx] = np.mean(data_array[where[:, 0]])

    single_bead = 22
    bead_unmixed_single = []
    for t in t_images:
        data_array = data_set[t]
        where = np.argwhere(predict_on == single_bead)
        bead_unmixed_single.append(data_array[where[:, 0]])
    bead_unmixed_single = np.array(bead_unmixed_single)

    ###################################################
    # Get kinetics data OFF-Rate                      #
    ################################################### 
    # Make copy to combine data
    data_set_all = data_set.copy()
    bead_unmixed_all = bead_unmixed.copy()
    bead_unmixed_single_all = bead_unmixed_single.copy()

    calc_off_s = ba.ImageSet(CALC_PATH_OFF_S)
    calc_off_s_set = calc_off_s.readSet()
    calc_off_s_set = calc_off_s_set[:, :, CROP[0]:CROP[1], CROP[2]:CROP[3]]

    t_images = xrange(0, calc_off_s.sizeT)

    data_set = []
    idx = np.arange(1, len(np.unique(labels_on)))
    for i in t_images:
        data = ndi.labeled_comprehension(calc_off_s_set[i,1,:,:], labels_annulus_on, idx, np.mean, float, -1)
        data_set.append(data)

    data_set = np.array(data_set)
    data_set = data_set[:, data_filter_list_on]

    beads = np.unique(predict_on)

    # Unmix kinetics data
    bead_unmixed = np.empty((calc_off_s.sizeT, len(beads)))
    for t in t_images:
        data_array = data_set[t]
        for idx, val in np.ndenumerate(beads):
            where = np.argwhere(predict_on == val)
            bead_unmixed[t, idx] = np.mean(data_array[where[:, 0]])

    bead_unmixed_single = []
    for t in t_images:
        data_array = data_set[t]
        where = np.argwhere(predict_on == single_bead)
        bead_unmixed_single.append(data_array[where[:, 0]])
    bead_unmixed_single = np.array(bead_unmixed_single)

    # Combine ON-rate with OFF-rate
    #bead_unmixed_all = np.append(bead_unmixed_all, bead_unmixed, axis=0)
        #bead_unmixed_all = np.subtract(bead_unmixed_all, np.min(bead_unmixed_all))
    #bead_unmixed_single_all = np.append(bead_unmixed_single_all, bead_unmixed_single, axis=0)
        #bead_unmixed_single_all = np.subtract(bead_unmixed_single_all, np.min(bead_unmixed_single_all))
    #data_set_all = np.append(data_set_all, data_set, axis=0)
        #data_set_all = np.subtract(data_set_all, np.min(data_set_all))

    # Plots
    # X Split
    x_s = circles_dim_on_filt[:, 0]
    colors = []
    min_c = np.min(x_s)
    max_c = np.max(x_s)
    for x in x_s:
        colors.append([(x - min_c) / (max_c - min_c), (x - min_c) /
                       (max_c - min_c), (x - min_c) / (max_c - min_c)])

    fig = plt.figure()
    fig.suptitle("Training Set - X-Coloring")
    ax = fig.add_subplot(111)
    ax.plot(data_set_all)
    [ax.lines[i].set_color(color) for i, color in enumerate(colors)]
    plt.draw()
    
    # Plot individual beads data
    fig = plt.figure()
    fig.suptitle("Training Set - All Beads")
    plt.plot(data_set_all)
    plt.draw()

    # Plot unmixed codes data
    fig = plt.figure()
    fig.suptitle("Training Set - Unmixed")
    plt.plot(bead_unmixed_all)
    plt.legend(beads)
    plt.draw()

    # Plot unmixed codes data
    #fig = plt.figure()
    #fig.suptitle("Training Set - Unmixed - Single: %s" % (single_bead + 1))
    #plt.plot(bead_unmixed_single_all)
    #plt.draw()

    plt.show()

    return 0


### Main loop ###
if __name__ == '__main__':
    status = main()
    sys.exit(status)
