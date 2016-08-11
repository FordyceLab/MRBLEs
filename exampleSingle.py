# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : exampleSingle.py          [filename]
# description       : Main file to call and use BeadKinetics module
# author            : Bjorn Harink              [Original author(s) of this file]
# credits           : Kurt Thorn, Huy Nguyen    [Contributors to this file]
# date              : 20160308                  [Initial date yyyymmdd]
# version update    : 20160808                  [Last version update yyyymmdd]
# version           : v0.4
# usage             : This is an example file for the Bead Analysis module.
# notes             : Single file code set.
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General
import sys
sys.path.append('./')
from itertools import cycle
import re
# Data structures
import numpy as np
import pandas as pd
# Math
from scipy import ndimage as ndi   # Imaging
from sklearn import mixture   # GMM
# Image display
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import axes3d
# Project
import bead_analysis as ba

##################################################
#                    [NOTES]                     # 
##################################################
"""[Notes]:
Notes here...
"""


##################################################
#                   [SETINGS]                    # 
##################################################
"""[Notes SETINGS]:
Notes here...
"""
# Reference image location
REF_FILES = {"Trp":"Z:/Huy/20160314_WBlank/20160314_W_blank_4/20160314_W_blank_4_MMStack.ome.tif",
             "Dy" : "Z:/Huy/20160315_Solo/20160315_Solo_Dy_9/20160315_Solo_Dy_9_MMStack.ome.tif",
             "Sm" : "Z:/Huy/20160315_Solo/20160315_Solo_Sm_4/20160315_Solo_Sm_4_MMStack.ome.tif",
             "Tm" : "Z:/Huy/20160315_Solo/20160315_Solo_Tm_4/20160315_Solo_Tm_4_MMStack.ome.tif",
             "Eu" : "Z:/Huy/20160315_Solo/20160315_Solo_Eu_4/20160315_Solo_Eu_4_MMStack.ome.tif"}

# Target file location
TARGET_FILE = "Z:/Code Sets/20160226_DySmTm_48Codes.csv"

# General Region or interest
# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X). Pandas includes stop element!
CROPy = slice(80, 420)
CROPx = slice(80, 420)

# Bead image set file location
BEAD_IMAGE_FILE = "Z:/Bjorn/20160706 - Follow-up Peptide test/20160706 - PBST flush_1/20160706 - PBST flush_1_MMStack.ome.tif"

# Background image set file location
BACK_FILE = "Z:/Bjorn/20160615 - BR05 - Calc LXVP 48B/20160615_Empty_BR05_1/20160615_Empty_BR05_1_MMStack.ome.tif"
# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X). Pandas includes stop element!
BACK_CROPy = slice(100, 400)
BACK_CROPx = slice(261, 273)


##################################################
#                 [MAIN SCRIPT]                  # 
##################################################


#########################
###   Targets/Codes   ###
"""[NOTES - Target File]
The target file contains the code ratios and are stored 
in a csv file with makeup [CeTb, Dy, Sm, Tm], e.g.:
>>> target
array([[ 0.20061,  0.08155,  0.     ,  0.65606],
       [ 0.     ,  0.08155,  0.10822,  0.     ],
       [ 0.     ,  0.08155,  0.10822,  0.19141],
       ...
"""

target = np.genfromtxt(TARGET_FILE, delimiter=',')
target = target[:, 1:]  # Target set to Dy, Sm and Tm (no CeTb)


#########################
### Reference Spectra ###
"""[NOTES - Reference Spectra]
Get reference spectra of solo lanthanide beads
"""
print("[Creating reference spectra]")

# Find beads objects with search parameters
ref_objects = ba.FindBeads(min_r=3, max_r=6, min_dist=10, param_1=10, param_2=7)
# Reference spectra data object
ref_data_object = ba.data.Spectra()

# Dark noise is subtracted from the lanthanide spectra and then normalized with the sum of the data.
dark_noise = 451   # Dark noise value
for name, file in REF_FILES.iteritems():
    print("Spectrum %s:" % name)
    ref_img_obj = ba.ImageSetRead(file)
    ref_objects.find(ref_img_obj['BF'][CROPy,CROPx])
    ref_data_tmp = np.array([ndi.median(ch, ref_objects.labeled_mask) for ch in ref_img_obj['Ex292-Em435':'Ex292-Em650'][:,CROPy,CROPx]])
    ref_data_tmp -= dark_noise              # Dark noise subtract
    ref_data_tmp /= ref_data_tmp.sum()      # Normalize
    ref_data_object.spec_add(name, data=ref_data_tmp, channels=ref_img_obj.c_names[1:10])
    
# Get background spectrum
print("Spectrum Bkg: %s, %s" % (BACK_CROPy, BACK_CROPy))
bkg_img_obj = ba.ImageSetRead(BACK_FILE)
ref_data_tmp = np.array([np.median(ch) for ch in bkg_img_obj['Ex292-Em435':'Ex292-Em650'][:,BACK_CROPy,BACK_CROPx]])
ref_data_tmp /= ref_data_tmp.sum()   # Normalize
ref_data_object.spec_add('Bkg', ref_data_tmp, channels=bkg_img_obj.c_names[1:10])

# Plot reference spectrum
ref_data_object.plot()


#########################
###    Bead Objects   ###
"""[NOTES - Bead Objects]
"""
print("[Load bead images and find objects]")

bead_image_obj = ba.ImageSetRead(BEAD_IMAGE_FILE)
bead_image_set = bead_image_obj[:][:,CROPy, CROPx]

bead_objects = ba.FindBeads(min_r=3, max_r=6, param_1=20, param_2=6, annulus_width=3, enlarge = 1)
bead_objects.find(bead_image_obj['BF'][CROPy, CROPx])
labels = bead_objects.labeled_mask
labels_annulus = bead_objects.labeled_annulus_mask
circles_dim = bead_objects.circles_dim

bead_no = 0
bead_set = pd.DataFrame(index=['no','img','lbl', 'dim','ratios','bkg','ref','code'])
for lbl in np.arange(1, len(np.unique(labels))):
    bead_set[bead_no] = [bead_no, 0, lbl, circles_dim[lbl-1], None, None, None, None]
    bead_no += 1

fig = plt.figure()
fig.suptitle("Overlay Image Pre-filter")
plt.imshow(bead_objects.overlay_image(bead_image_set[0]), cmap='Greys_r')
plt.draw()


#########################
### Unmix and Ratios  ###
"""[NOTES - Unmix and Ratios]
"""
print("[Unmix and get ratios]")
spec_unmix = ba.SpectralUnmixing(ref_data_object)
spec_unmix.unmix(bead_image_set[1:10])

# Background and reference
background = spec_unmix['Bkg']  # Device background
reference = spec_unmix['Eu']  # Internal reference: Eu
# Ratio images
ratio_Dy = spec_unmix['Dy'] / reference
ratio_Sm = spec_unmix['Sm'] / reference
ratio_Tm = spec_unmix['Tm'] / reference
# Get ratios from images
idx = np.arange(1, len(np.unique(labels)))
ratio_data = np.empty((len(idx), 3))
ratio_data[:, 0] = ndi.labeled_comprehension(ratio_Dy, labels, idx, np.median, float, -1)
ratio_data[:, 1] = ndi.labeled_comprehension(ratio_Sm, labels, idx, np.median, float, -1)
ratio_data[:, 2] = ndi.labeled_comprehension(ratio_Tm, labels, idx, np.median, float, -1)

background_data = ndi.labeled_comprehension(background, labels, idx, np.median, float, -1)
reference_data = ndi.labeled_comprehension(reference, labels, idx, np.median, float, -1)

bead_no = 0
for lbl in np.arange(1, len(np.unique(labels))):
    bead_set[bead_no][['ratios', 'bkg', 'ref']] = [ratio_data[lbl-1], background_data[lbl-1], reference_data[lbl-1]]
    bead_no += 1

#########################
###     Filtering     ###
"""[NOTES - Filtering]
"""
print("[Filtering]")

ratio_data = np.vstack(bead_set.ix['ratios'])
background_data = bead_set.ix['bkg'].values
reference_data = bead_set.ix['ref']
circles_dim = np.vstack(bead_set.ix['dim'])

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

print("Pre filter: %s" % len(ratio_data))
print("Post filter: %s" % len(data_filtered))

fig = plt.figure()
fig.suptitle("Overlay Image Post-filter")
plt.imshow(bead_objects.overlay_image(bead_image_obj['BF'][CROPy,CROPx], dim=circles_dim_filt), cmap='Greys_r')
plt.draw()

#########################
###        ICP        ###
"""[NOTES - ICP]
"""
print("[Iterative Closest Point]")

icp= ba.ICP(matrix_method = 'std', max_iter=100, tol=1e-4, outlier_pct=0.001)
icp.fit(data_filtered, target)
data_icp = icp.transform(data_filtered)
print("Tranformation matrix: ", icp.matrix)
print("Offset vector: ", icp.offset)

#########################
###        GMM        ###
"""[NOTES - GMM]
"""
print("[Gaussian Mixture Modeling]")

# GMM Setup
nclusters = len(target[:, 0])
naxes = len(target[0, :])
sigma = np.eye(naxes) * 1e-5

# GMM
gmix = mixture.GMM(n_components=nclusters, covariance_type='full',
                   min_covar=1e-7, tol=1e-5, init_params='', params='wmc')
gmix.means_ = target
gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
gmix._weights_ = np.tile(1 / 48, (nclusters))
gmix.fit(data_icp, target)
predict = gmix.predict(data_icp)

print("Number of unique beads found:", len(np.unique(predict)))

# Clustering graphs
colors = np.multiply(predict, 5)
colors_target = np.empty([48,3])
colors_target.fill(0)

fig = plt.figure()
fig.suptitle("Clustering pre-ICP & Pre-filter")
ax = fig.gca(projection='3d')
ax.scatter(ratio_data[:, 0], ratio_data[:, 1], ratio_data[:, 2], alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

fig = plt.figure()
fig.suptitle("Clustering pre-ICP")
ax = fig.gca(projection='3d')
ax.scatter(data_filtered[:, 0], data_filtered[:, 1], data_filtered[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

fig = plt.figure()
fig.suptitle("Clustering post-ICP")
ax = fig.gca(projection='3d')
ax.scatter(data_icp[:, 0], data_icp[:, 1], data_icp[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

# Show all images at once
plt.show()
