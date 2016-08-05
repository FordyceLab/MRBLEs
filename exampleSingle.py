# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : exampleSingle.py      [filename]
# description       : Main file to call and use BeadKinetics module
# author            : Bjorn Harink          [Original author(s) of this file]
# credits           : Kurt Thorn, Huy Nguyen[Contributors to this file]
# date              : 20160308              [Initial date yyyymmdd]
# version update    : 20160801              [Last version update yyyymmdd]
# version           : v0.4
# usage             : As startup file
# notes             : This is an example file for the module using a single file set
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General
import sys
sys.path.append('./')
from itertools import cycle
import re
# Math
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.integrate import odeint
# Machine learning
from sklearn import mixture
# Image disply
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import axes3d
# Project
import bead_analysis as ba

# [Settings]
# Reference image location
REF_FILES = {"Trp":"Z:/Huy/20160314_WBlank/20160314_W_blank_4/20160314_W_blank_4_MMStack.ome.tif",
             "Dy" : "Z:/Huy/20160315_Solo/20160315_Solo_Dy_9/20160315_Solo_Dy_9_MMStack.ome.tif",
             "Sm" : "Z:/Huy/20160315_Solo/20160315_Solo_Sm_4/20160315_Solo_Sm_4_MMStack.ome.tif",
             "Tm" : "Z:/Huy/20160315_Solo/20160315_Solo_Tm_4/20160315_Solo_Tm_4_MMStack.ome.tif",
             "Eu" : "Z:/Huy/20160315_Solo/20160315_Solo_Eu_4/20160315_Solo_Eu_4_MMStack.ome.tif"}

# Target file location
TARGET_FILE = "Z:/Code Sets/20160226_DySmTm_48Codes.csv"

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
CROP = [80, 420, 80, 420]

# Bead image set file location
#BEAD_IMAGE_FILE = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/On-Rate-T0.ome.tif"
BEAD_IMAGE_FILE = "Z:/Bjorn/20160706 - Follow-up Peptide test/20160706 - PBST flush_1/20160706 - PBST flush_1_MMStack.ome.tif"

# Background image set file location
BACK_FILE = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/20160615_Empty_BR05_1/20160615_Empty_BR05_1_MMStack.ome.tif"
# [Y1, Y2, X1, X2] Y and X are reversed in array since rows (Y) go first and columns go second (X)
BACK_CROP = [100, 400, 261, 273]



# [Main Script]
print(ba.__copyright__)

# Set target file of code set and choose targets
target = np.genfromtxt(TARGET_FILE, delimiter=',')
target = target[:, 1:]  # Target set to Dy, Sm and Tm (no CeTb)

##################################################
# Get reference spectra of solo lanthanide beads #
##################################################

print("Create reference spectra")
ref_objects = ba.Objects2(min_r=3, max_r=6, min_dist=10, param_1=10, param_2=7)
ref_data_object = ba.data.Spectra()

dark_noise = 451
for name, file in REF_FILES.iteritems():
    print("Spectrum %s:" % name)
    ref_img = ba.ImageSet(file).readSet()[:,CROP[0]:CROP[1], CROP[2]:CROP[3]]
    ref_objects.find(ref_img[0])
    ref_data_tmp = np.array([ndi.median(ch, ref_objects.labeled_mask) for ch in ref_img[1:10]])
    ref_data_tmp = ref_data_tmp - dark_noise   # Dark noise subtract
    sum = ref_data_tmp.sum()
    ref_data_tmp = np.divide(ref_data_tmp, sum)   # Normalize
    ref_data_object.add_channel(name, ref_data_tmp)
    
###################################################
# Get kinetics data clustering and classification #
################################################### 

# Get background spectrum
bkg_img = ba.ImageSet(BACK_FILE).readSet()[:,BACK_CROP[0]:BACK_CROP[1], BACK_CROP[2]:BACK_CROP[3]]
ref_data_tmp = np.array([np.median(ch) for ch in bkg_img[1:10]])
sum = ref_data_tmp.sum()
ref_data_tmp = np.divide(ref_data_tmp, sum)   # Normalize
ref_data_object.add_channel('Bkg', ref_data_tmp )

fig = plt.figure()
fig.suptitle("Background Image")
plt.imshow(bkg_img[0], cmap='Greys_r')
plt.draw()

fig = plt.figure()
fig.suptitle("Reference Spectra")
plt.plot(ref_data_object.all)
plt.draw()

###################################################
# Get objects                                     #
################################################### 

print("Load Bead Images")
bead_image_set = ba.ImageSet(BEAD_IMAGE_FILE).readSet()[:, CROP[0]:CROP[1], CROP[2]:CROP[3]]
bead_objects = ba.Objects2(min_r=3, max_r=6, param_1=20, param_2=6, annulus_width=3, enlarge = 1)
bead_objects.find(bead_image_set[0])
labels = bead_objects.labeled_mask
labels_annulus = bead_objects.labeled_annulus_mask
circles_dim = bead_objects.circles_dim

fig = plt.figure()
fig.suptitle("Overlay Image Pre-filter")
plt.imshow(bead_objects.overlay_image(bead_image_set[0]), cmap='Greys_r')
plt.draw()

# Unmix images by least squares
unmixed = ba.unmix(ref_data_object.all, bead_image_set[1:10, :, :])

#for ch in ref_data_object.channels:
#    chan_no = ref_data_object.channel_no(ch)
#    plt.figure()
#    plt.suptitle("Lanthanide channel: %s" % ch)
#    plt.imshow(unmixed[chan_no], cmap='Greys_r')
#    plt.draw()
#plt.show()

#median_data = ba.getSpectralMedianIntensities(labels, unmixed)
#background_data = median_data[:, ref_data_object.channel_no('Bkg')]
#reference_data = median_data[:, ref_data_object.channel_no('Eu')]
# Omit Bkg and Trp ratios
#ratio_list = [ref_data_object.channel_no('Dy'), ref_data_object.channel_no('Sm'), ref_data_object.channel_no('Tm')]
#ratio_ref = ref_data_object.channel_no('Eu')
#ratio_data = ba.getRatios(labels, unmixed[ratio_list], unmixed[ratio_ref])

background = unmixed[ref_data_object.channel_no('Bkg')]  # Device background
reference = unmixed[ref_data_object.channel_no('Eu')]  # Internal reference: Eu
# Omit CeTb and Trp ratios
ratio_Dy = np.divide(unmixed[ref_data_object.channel_no('Dy'),:,:], reference)
ratio_Sm = np.divide(unmixed[ref_data_object.channel_no('Sm'),:,:], reference)
ratio_Tm = np.divide(unmixed[ref_data_object.channel_no('Tm'),:,:], reference)
# Get ratios
idx = np.arange(1, len(np.unique(labels)))
ratio_data = np.empty((len(idx), 3))
ratio_data[:, 0] = ndi.labeled_comprehension(ratio_Dy, labels, idx, np.median, float, -1)
ratio_data[:, 1] = ndi.labeled_comprehension(ratio_Sm, labels, idx, np.median, float, -1)
ratio_data[:, 2] = ndi.labeled_comprehension(ratio_Tm, labels, idx, np.median, float, -1)

background_data = ndi.labeled_comprehension(background, labels, idx, np.median, float, -1)
reference_data = ndi.labeled_comprehension(reference, labels, idx, np.median, float, -1)

# Filter objects based on background and reference
radius_min = 3
radius_max = 7
reference_std_factor_low = 1
reference_std_factor_high = 3
back_std_factor = 3
mean_back = np.mean(background_data)
std_back = np.std(background_data)
mean_reference = np.mean(reference_data)
std_reference = np.std(reference_data)
# Filter objects based on background and reference
size_filter = np.logical_and(circles_dim[:, 2] >= radius_min, circles_dim[:, 2] <= radius_max)
back_filter = np.logical_and(background_data < (mean_back + back_std_factor * std_back),
                                background_data > (mean_back - back_std_factor * std_back))
ref_filter = np.logical_and(reference_data > (mean_reference - reference_std_factor_low * std_reference),
                                        reference_data < (mean_reference + reference_std_factor_high * std_reference))
data_filter_list = np.argwhere(np.logical_and(size_filter, np.logical_and(back_filter, ref_filter)))[:, 0]
data_filtered = ratio_data[data_filter_list]
circles_dim_filt = circles_dim[data_filter_list]

#data_filtered = ratio_data
#circles_dim_filt = circles_dim

print("Pre filter: %s" % len(ratio_data))
print("Post filter: %s" % len(data_filtered))

fig = plt.figure()
fig.suptitle("Overlay Image Post-filter")
plt.imshow(bead_objects.overlay_image(bead_image_set[0], dim=circles_dim_filt), cmap='Greys_r')
plt.draw()

# Iterative Closest Point
print("Iterative Closest Point")
icp= ba.ICP(matrix_method = 'std', max_iter=100, tol=1e-4)
icp.fit(data_filtered, target)
data_icp = icp.transform(data_filtered)
print("Tranformation matrix: ", icp.matrix)
print("Offset vector: ", icp.offset)

# GMM Setup
nclusters = len(target[:, 0])
naxes = len(target[0, :])
sigma = np.eye(naxes) * 1e-5

# New GMM
gmix = mixture.GMM(n_components=nclusters, covariance_type='full',
                        min_covar=1e-7, tol=1e-5, init_params='', params='wmc')
gmix.means_ = target
gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
gmix._weights_ = np.tile(1 / 48, (nclusters))
gmix.fit(data_icp, target)
predict = gmix.predict(data_icp)

print("Number of unique beads found:", len(np.unique(predict)))
    
colors = np.multiply(predict, 5)
colors_target = np.empty([48,3])
colors_target.fill(0)

fig = plt.figure()
fig.suptitle("Clustering pre-ICP - Kinetics Set")
ax = fig.gca(projection='3d')
ax.scatter(data_filtered[:, 0], data_filtered[:, 1], data_filtered[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

fig = plt.figure()
fig.suptitle("Clustering post-ICP - Training Set")
ax = fig.gca(projection='3d')
ax.scatter(data_icp[:, 0], data_icp[:, 1], data_icp[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

# Show all images at once
plt.show()
