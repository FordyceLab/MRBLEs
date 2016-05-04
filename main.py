# !/usr/bin/env python

### Future imports
from __future__ import print_function   # "print" function compatibility between Python 2.x and 3.x
from __future__ import division         # Makes it possible to use / for division in Pyhton 2.x


### File header     | Copy and edit for each file in this project!
# title             : main.py               [filename]
# description       : Main file to call BeadKinetics module
# author            : Bjorn Harink          [Original author/starter of this file]
# date              : 20160308              [Initial date yyyymmdd]
# last update       : 20160407              [Last update yyyymmdd]
# version           : v0.3
# usage             : As main file
# notes             : 
# python_version    : 2.7


### Modules
# General
import sys
# Math
import numpy as np
from scipy import ndimage as ndi
# Plotting
from matplotlib import pyplot as plt
# Project
import bead_analysis as ba
# Machine learning
from sklearn import mixture
from mpl_toolkits.mplot3d import axes3d


### Settings
## Reference image location
REF_FILES = {"CeTb" : "Z:\\Huy\\20150706\\20150709_CeTbSolo_1\\20150709_CeTbSolo_1_MMStack_Pos0.ome.tif", 
            "Dy"    : "Z:\\Huy\\20150706\\20150709_DySolo_1\\20150709_DySolo_1_MMStack_Pos0.ome.tif",
            "Sm"    : "Z:\\Huy\\20150706\\20150709_SmSolo_3\\20150709_SmSolo_3_MMStack_Pos0.ome.tif",
            "Tm"    : "Z:\\Huy\\20150706\\20150709_TmSolo_3\\20150709_TmSolo_3_MMStack_Pos0.ome.tif",
            "Eu"    : "Z:\\Huy\\20150706\\20150709_EuSolo_1\\20150709_EuSolo_1_MMStack_Pos0.ome.tif",
            "Trp"      : "Z:\\Huy\\20160314_WBlank\\20160314_W_blank_4\\20160314_W_blank_4_MMStack.ome.tif"}

## Target file location
TARGET_FILE = "Z:\\Code Sets\\20150714_DySmTm_48codes.csv"
#TARGET_FILE = "D:\\Documents\\ThornLab Server\\Code Sets\\20160226_DySmTm_48Codes.csv"

## Training set location
TRAIN_PATH = "D:\\Documents\\ThornLab Server\\20160328"
# Image set channels
TRAIN_CHANNELS =  { "BF"     :1,
                    "435"    :2, 
                    "474"    :3, 
                    "527"    :4, 
                    "536"    :5, 
                    "546"    :6,
                    "572"    :7,
                    "620"    :8,
                    "630"    :9,
                    "650"    :10,
                    "Cy3"    :11,
                    "Cy5"    :12,
                    "FITC"   :13}

## Image set location
IMAGE_PATH = "Z:\\Huy\\20160225"
# Image set channels
IMAGE_CHANNELS =  { "BF"     :1,
                    "435"    :2, 
                    "474"    :3, 
                    "527"    :4, 
                    "536"    :5, 
                    "546"    :6,
                    "572"    :7,
                    "620"    :8,
                    "630"    :9,
                    "650"    :10,
                    "Cy3"    :11,
                    "Cy5"    :12,
                    "FITC"   :13}

CROP = [100,400,100,400] # [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)

### Main function ###
def main():
    target = np.genfromtxt(TARGET_FILE, delimiter=',')
    
    # Get reference spectra of solo lanthanide beads    
    ref_data = np.empty((9,6), dtype="float64")
    #image_set = ba.ImageSet(REF_FILES["CeTb"])
    #image_data_CeTb = image_set.readSet()[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    #del image_set
    image_set = ba.ImageSet(REF_FILES["Dy"])
    image_data_Dy = image_set.readSet()[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    del image_set
    image_set = ba.ImageSet(REF_FILES["Sm"])
    image_data_Sm = image_set.readSet()[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    del image_set
    image_set = ba.ImageSet(REF_FILES["Tm"])
    image_data_Tm = image_set.readSet()[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    del image_set
    image_set = ba.ImageSet(REF_FILES["Eu"])
    image_data_Eu = image_set.readSet()[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    del image_set
    # Tryptophan spectrum
    image_set = ba.ImageSet(REF_FILES["Trp"])
    image_data_Trp = image_set.readSet()[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    del image_set

    print("Generate reference spectra")
    #ref_data[:,1] = ba.getRef(image_data_CeTb)
    ref_data[:,1] = ba.getRef(image_data_Trp, min_dist = 4, min_r = 2, max_r = 5, param_2 = 11)
    ref_data[:,2] = ba.getRef(image_data_Dy)
    ref_data[:,3] = ba.getRef(image_data_Sm)
    ref_data[:,4] = ba.getRef(image_data_Tm)
    ref_data[:,5] = ba.getRef(image_data_Eu)

    image_files = ba.mmScanPath(IMAGE_PATH)

    # Background spec
    # [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
    BACK = [280,300,230,270]
    image_set = ba.ImageSet("Z:\\Huy\\20160225\\20160226_48codes_1to48Test_1\\20160226_48codes_1to48Test_1_MMStack.ome.tif")
    image_data_set = image_set.readSet()
    ref_data[:,0] = ba.getBack(image_data_set, BACK)
    del image_set

    print("Generate training data")
    # Data all
    raw_data = []        
    data_ratios = []
    data_circles_dim = []
    def run_paths(ipath):
        labels, labels_annulus, circles_dim, median_data, ratio_data = ba.getUnmixedData(ipath, CROP, ref_data, 5)
        raw_data.append(median_data)
        data_ratios.append(ratio_data)
        data_circles_dim.append(circles_dim)
    map(run_paths, image_files)

    plt.figure()
    plt.plot(ref_data)
    plt.draw()

    rflatten_1 = [item for sublist in raw_data for item in sublist]
    data_raw = np.array(rflatten_1)

    rat_flatten_1 = [item for sublist in data_ratios for item in sublist]
    ratio_data = np.array(rat_flatten_1) 

    c_flatten_1 = [item for sublist in data_circles_dim for item in sublist]
    circles_dim = np.array(c_flatten_1)

    background = data_raw[:,0]  # Device background
    reference = data_raw[:,5]   # Internal reference: Eu

    # Filter objects based on background and reference
    data_filter_list = ba.filterObjects(ratio_data, background, reference, circles_dim[:,2])
    data_filtered = ratio_data[data_filter_list,1:4] # Omit CeTb and Trp ratios
    circles_dim_filt = circles_dim[data_filter_list]

    # Iterative closest point
    target = target[:,1:] # Target set to Dy, Sm and Tm (no CeTb)
    data_icp = ba.icp(data_filtered, target, tol = 1e-4, max_iter = 10)
    
    #Gaussian Mixture Modeling
    nclusters = len(target[:,0])
    naxes = len(target[0,:])    
    sigma = np.eye(naxes) * 1e-5
    gmix = mixture.GMM(n_components=nclusters, covariance_type='full', min_covar=1e-7, tol = 1e-7, init_params='', params='wmc')
    gmix.means_ = target
    gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
    gmix.weights_ = np.tile(1/48, (nclusters))
    gmix.fit(data_icp, target)
    predict = gmix.predict(data_icp)

    np.set_printoptions(threshold=1e4)
    print("Number of unique beads found:", len(np.unique(predict)))
    colors = np.multiply(predict, 5)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data_filtered[:,0], data_filtered[:,1], data_filtered[:,2], c=colors, alpha=0.8)
    ax.scatter(target[:,0], target[:,1], target[:,2],alpha=0.8)
    plt.draw() 
  
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data_icp[:,0], data_icp[:,1], data_icp[:,2], c=colors, alpha=0.8)
    ax.scatter(target[:,0], target[:,1], target[:,2],alpha=0.8)
    plt.draw() 

#################################
    
    print("BeadReactor data")
    TARGET_FILE_ADJ = "Z:\Code Sets\\20150714_DySmTm_48codes.csv"
    target = np.genfromtxt(TARGET_FILE_ADJ, delimiter=',')
    bead_no = 48
    ### Calcineurin
    CALC_PATH_ON_1 = "Z:\\Bjorn\\20160331 - Calcineurin\\20160331 - Start Antibody Calc_3\\20160331 - Start Antibody Calc_3_MMStack.ome.tif"
    CALC_PATH_ON_S = "Z:\\Bjorn\\20160331 - Calcineurin\\20160331 - Start Antibody Calc - Seq_1\\20160331 - Start Antibody Calc - Seq_1_MMStack.ome.tif"
    CALC_PATH_OFF_1 = "Z:\\Bjorn\\20160331 - Calcineurin\\20160331 - Start Antibody Calc - Seq End 4hr - Off start_1\\20160331 - Start Antibody Calc - Seq End 4hr - Off start_1_MMStack.ome.tif"
    CALC_PATH_OFF_S = "Z:\\Bjorn\\20160331 - Calcineurin\\20160331 - Start Antibody Calc - Seq End 4hr - Off start_2\\20160331 - Start Antibody Calc - Seq End 4hr - Off start_2_MMStack.ome.tif"
    
    #CALC_PATH_ON_1 = "Z:\\Bjorn\\20160322 - BeadReactor 0.4 - HA Flag Mic\\BR04 - On-rate Cy3 Anti-Flag_5\\BR04 - On-rate Cy3 Anti-Flag_5_MMStack.ome.tif"
    #CALC_PATH_ON_S = "Z:\\Bjorn\\20160322 - BeadReactor 0.4 - HA Flag Mic\\BR04 - On-rate Cy3 Anti-Flag_7\\BR04 - On-rate Cy3 Anti-Flag_7_MMStack.ome.tif"
    #CALC_PATH_OFF_1 = "Z:\\Bjorn\\20160322 - BeadReactor 0.4 - HA Flag Mic\\BR04 - Cy3 Anti-Flag While flowing buffer_2\\BR04 - Cy3 Anti-Flag While flowing buffer_2_MMStack.ome.tif"
    #CALC_PATH_OFF_S = "Z:\\Bjorn\\20160322 - BeadReactor 0.4 - HA Flag Mic\\BR04 - Cy3 Anti-Flag While flowing buffer - Just measure_1\\BR04 - Cy3 Anti-Flag While flowing buffer - Just measure_1_MMStack.ome.tif"

    calc_on = ba.ImageSet(CALC_PATH_ON_1)
        
    calc_on_set = calc_on.readSet()
    calc_on_set_1 = calc_on.readSet(idx = 0)
    calc_on_set = calc_on_set[:,CROP[0]:CROP[1],CROP[2]:CROP[3]]
    calc_on_objects = ba.Objects(calc_on_set_1[CROP[0]:CROP[1],CROP[2]:CROP[3]])

    labels_on, labels_annulus_on, circles_dim_on = calc_on_objects.findObjects()

    # [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
    # Dimensions after or before crop!?
    BACK_BR = [100,400,261,273]
    ba_img_set = ba.ImageSet("Z:/Bjorn/20160321 - BeadReactor 0.4 - Calcineurin/BR04 - BR Background UV-Focussed_1/BR04 - BR Background UV-Focussed_1_MMStack.ome.tif")
    ba_img_read = ba_img_set.readSet()
    ref_data[:,0] = ba.getBack(ba_img_read[0:10], BACK_BR)
    del ba_img_set

    plt.figure()
    plt.plot(ref_data)
    plt.draw()

    # Unmix images by least squares
    unmixed_on =  ba.unmix(ref_data, calc_on_set[1:10,:,:])

    median_data_on = ba.getSpectralMedianIntensities(labels_on, unmixed_on)
    ratio_data_on = ba.getRatios(labels_on, unmixed_on[2:5], unmixed_on[5]) # Omit CeTb and Trp ratios

    background_on = median_data_on[:,0] # Device background
    reference_on = median_data_on[:,5]  # Internal reference: Eu

    # Filter objects based on background and reference
    data_filter_list_on = ba.filterObjects(ratio_data_on, background_on, reference_on, circles_dim_on[:,2])
    data_filtered_on = ratio_data_on[data_filter_list_on] 
    circles_dim_on_filt = circles_dim_on[data_filter_list_on]

    plt.figure()
    plt.imshow(calc_on_objects.overlayImage(circles_dim_on, ring_size=2))
    plt.draw()

    plt.figure()
    plt.imshow(calc_on_objects.overlayImage(circles_dim_on_filt, ring_size=2))
    plt.draw()

    # Iterative closest point
    target = target[:,1:] # Target set to Dy, Sm and Tm (no CeTb)
    d_size = data_filtered_on[:,0].size
    #data_filtered_on = np.append(data_filtered, data_filtered_on, axis = 0)

    # Initial guess matrix
    nLn = len(data_filtered_on[0,:])
    matrix = np.eye(nLn)
    for n in xrange(nLn):
        matrix[n,n] = np.divide( np.mean(target[:,n]), np.mean(data_filtered_on[:,n]) )
    matrix[2,2] = 3

    data_icp_on = ba.icp(data_filtered_on, target, input_matrix = matrix, tol = 1e-4)

    # GMM Setup
    nclusters = len(target[:,0])
    naxes = len(target[0,:])    
    sigma = np.eye(naxes) * 1e-5

    # New GMM
    gmix_on = mixture.GMM(n_components=nclusters, covariance_type='full', min_covar=1e-7, tol = 1e-5, init_params='', params='wmc')
    gmix_on.means_ = target
    #gmix_on.covars_ = np.tile(sigma, (nclusters, 1, 1))
    gmix_on.weights_ = np.tile(1/48, (nclusters))
    #gmix_on.means_ = gmix.means_
    gmix_on.covars_ = gmix.covars_
    gmix_on.fit(data_icp_on, target)
    predict_on = gmix_on.predict(data_icp_on)

    # Re-use trained GMM
    #gmix.weights_ = np.tile(1/48, (nclusters))  # Reset weights
    #gmix.means_ = gmix.means_                   # Reset means
    #gmix_on.covars_ = gmix.covars_
    #gmix.fit(data_icp_on, target)
    #predict_on = gmix.predict(data_icp_on)

    #data_filtered_on = data_filtered_on[-d_size:]
    #data_icp_on = data_icp_on[-d_size:]
    #predict_on = predict_on[-d_size:]

    print("Number of unique beads found:", len(np.unique(predict_on)))
    colors_on = np.multiply(predict_on, 5)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data_filtered_on[:,0], data_filtered_on[:,1], data_filtered_on[:,2], c=colors_on, alpha=0.8)
    ax.scatter(target[:,0], target[:,1], target[:,2],alpha=0.8)
    plt.draw() 
  
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data_icp_on[:,0], data_icp_on[:,1], data_icp_on[:,2], c=colors_on, alpha=0.8)
    ax.scatter(target[:,0], target[:,1], target[:,2],alpha=0.8)
    plt.draw() 

### ON-Rate
    calc_on_s = ba.ImageSet(CALC_PATH_ON_S)
    calc_on_s_set = calc_on_s.readSet()
    calc_on_s_set = calc_on_s_set[:,:,CROP[0]:CROP[1],CROP[2]:CROP[3]]

    t_images = xrange(0,calc_on_s.sizeT)

    data_set = []
    idx=np.arange(1, len( np.unique(labels_on)))
    for i in t_images:
        #data = ndi.labeled_comprehension(calc_on_s_set[i,2,:,:], labels_on, idx, np.median, float, -1)
        data = ndi.mean( calc_on_s_set[i,2,:,:], labels=labels_annulus_on, index=idx )
        data_set.append(data)
        
    data_set = np.array(data_set)
    data_set = data_set[:,data_filter_list_on]

    beads = xrange(0,bead_no-1)

    bead_unmixed = np.empty((calc_on_s.sizeT,bead_no))
    for t in t_images:
        data_array = data_set[t]
        for x in beads:
            where = np.argwhere(predict_on == x)
            bead_unmixed[t,x] = np.mean(data_array[where[:,0]])

    

### OFF-Rate
    bead_unmixed_all = bead_unmixed
    data_set_all = data_set

    calc_off_s = ba.ImageSet(CALC_PATH_OFF_S)
    calc_off_s_set = calc_off_s.readSet()
    calc_off_s_set = calc_off_s_set[:,:,CROP[0]:CROP[1],CROP[2]:CROP[3]]

    t_images = xrange(0,calc_off_s.sizeT)

    data_set = []
    idx=np.arange(1, len( np.unique(labels_on)))
    for i in t_images:
        #data = ndi.labeled_comprehension(calc_off_s_set[i,1,:,:], labels_on, idx, np.median, float, -1)
        data = ndi.mean( calc_off_s_set[i,1,:,:], labels=labels_annulus_on, index=idx )
        data_set.append(data)

    data_set = np.array(data_set)
    data_set = data_set[:,data_filter_list_on]

    beads = xrange(0,bead_no-1)

    bead_unmixed = np.empty((calc_off_s.sizeT,bead_no))
    for t in t_images:
        data_array = data_set[t]
        for x in beads:
            where = np.argwhere(predict_on == x)
            bead_unmixed[t,x] = np.mean(data_array[where[:,0]])

    bead_unmixed_all = np.append(bead_unmixed_all, bead_unmixed, axis = 0)
    bead_unmixed_all = np.subtract(bead_unmixed_all, np.min(bead_unmixed_all))
    data_set_all = np.append(data_set_all, data_set, axis=0)
    data_set_all = np.subtract(data_set_all, np.min(data_set_all))

    ### Plots
    # X Split
    x_s = circles_dim_on_filt[:,0]
    colors = []
    min_c = np.min(x_s)
    max_c = np.max(x_s)
    for x in x_s:
        colors.append([(x-min_c)/(max_c-min_c), (x-min_c)/(max_c-min_c), (x-min_c)/(max_c-min_c)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data_set_all[:98])
    [ax.lines[i].set_color(color) for i, color in enumerate(colors)]
    plt.draw()

    x_filter_1 = np.argwhere(np.logical_and( x_s >= 0, x_s < 100 ))
    x_filter_2 = np.argwhere(np.logical_and( x_s >= 100, x_s < 200 ))
    x_filter_3 = np.argwhere(np.logical_and( x_s >= 200, x_s < 300 ))

    #plt.figure()
    #plt.plot(data_set_all[:98, x_filter_1[:,0]])
    #plt.axis([0, 98, 0, 16000])
    #plt.draw()

    #plt.figure()
    #plt.plot(data_set_all[:98, x_filter_2[:,0]])
    #plt.axis([0, 98, 0, 16000])
    #plt.draw()

    #plt.figure()
    #plt.plot(data_set_all[:98, x_filter_3[:,0]])
    #plt.axis([0, 98, 0, 16000])
    #plt.draw()

    plt.figure()
    plt.plot(data_set_all[:98])
    plt.axis([0, 98, 0, 16000])
    plt.draw()

    plt.figure()
    plt.plot(bead_unmixed_all[:98])
    plt.axis([0, 98, 0, 16000])
    plt.draw()
       
    plt.show()


### Main loop ###
if __name__ == '__main__':
    status = main()
    sys.exit(status)