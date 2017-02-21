# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : exampleMultiSplit.py      [filename]
# description       : Main file to call and use BeadKinetics module
# author            : Bjorn Harink              [Original author(s) of this file]
# credits           : Kurt Thorn, Huy Nguyen    [Contributors to this file]
# date              : 20160308                  [Initial date yyyymmdd]
# version update    : 20160808                  [Last version update yyyymmdd]
# version           : v0.4
# usage             : This is an example file for the Bead Analysis module.
# notes             : Multiple file code set with two Eu levels.
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General Python
import sys
sys.path.append('./')
# Data structures
import numpy as np
import pandas as pd
# Image Processing
from scipy import ndimage as ndi # Imaging
from sklearn.mixture import GMM # Gaussian Mixture Modeling
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
TARGET_FILE = "Z:/Code Sets/20160226_DySmTm_48Codes.csv"   # 48B

# General Region or interest
# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X). Pandas includes stop element!
CROPy = slice(80, 420)
CROPx = slice(80, 420)

# Bead image set file location
BEAD_IMAGE_FOLDER = "Z:/Bjorn/20160810 CN LxVP Mix"
BEAD_IMAGE_PATTERN = "20160810_BH_48A_LxVP_Mix_250nM_*"
#BEAD_IMAGE_PATTERN = "20160810_BH_48A_LxVP_Mix_100nM_*"
#BEAD_IMAGE_PATTERN = "20160810_BH_48A_LxVP_Mix_50nM_*"
#BEAD_IMAGE_PATTERN = "20160810_BH_48A_LxVP_Mix_25nM_*"
#BEAD_IMAGE_PATTERN = "20160810_BH_48A_LxVP_Mix_LM_*"
#BEAD_IMAGE_PATTERN = "20160810_BH_48A_LxVP_Mix_PM_*"

# Background image set file location
BACK_FILE = "Z:/Huy/20160810 CN/20160810_HQN_2_59_CN_PxIxIT_250nMb_1/20160810_HQN_2_59_CN_PxIxIT_250nMb_1_MMStack.ome.tif"
# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X). Pandas includes stop element!
BACK_CROPy = slice(70, 170)
BACK_CROPx = slice(140, 240)


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
ref_objects = ba.FindBeads(min_r=3, max_r=6, min_dist=10, param_1=10, param_2=7, enlarge=1.1)
# Reference spectra data object
ref_data_object = ba.data.Spectra()

# Dark noise is subtracted from the lanthanide spectra and then normalized with the sum of the data.
dark_noise = 451   # Dark noise value
for name, file in REF_FILES.iteritems():
    print("Spectrum %s:" % name)
    ref_img_obj = ba.ImageSetRead(file)
    ref_objects.find(ref_img_obj['BF',CROPy,CROPx])
    ref_data_tmp = np.array([ndi.median(ch, ref_objects.labeled_mask) for ch in ref_img_obj['Ex292-Em435':'Ex292-Em650',CROPy,CROPx]])
    ref_data_tmp -= dark_noise              # Dark noise subtract
    ref_data_tmp /= ref_data_tmp.sum()      # Normalize
    ref_data_object.spec_add(name, data=ref_data_tmp, channels=ref_img_obj.c_names[1:10])
    
# Get background spectrum
print("Spectrum Bkg: %s, %s" % (BACK_CROPy, BACK_CROPy))
bkg_img_obj = ba.ImageSetRead(BACK_FILE)
ref_data_tmp = np.array([np.median(ch) for ch in bkg_img_obj['Ex292-Em435':'Ex292-Em650',BACK_CROPy,BACK_CROPx]])
ref_data_tmp /= ref_data_tmp.sum()   # Normalize
ref_data_object.spec_add('Bkg', ref_data_tmp, channels=bkg_img_obj.c_names[1:10])

# Plot reference spectrum
ref_data_object.plot()


#########################
###    Bead Objects   ###
"""[NOTES - Bead Objects]
"""
print("[Load bead images and find objects]")

bead_image_files = ba.ImageSetRead.scan_path(BEAD_IMAGE_FOLDER, BEAD_IMAGE_PATTERN)
bead_image_obj = ba.ImageSetRead(bead_image_files)
bead_image_set_bf = bead_image_obj[:,'BF',CROPy,CROPx]
bead_image_set_ln = bead_image_obj[:,'Ex292-Em435':'Ex292-Em650',CROPy,CROPx]
bead_image_set_cy5 = bead_image_obj[:,'Cy5',CROPy,CROPx]

bead_objects = ba.FindBeads(min_r=3, max_r=6, param_1=20, param_2=6, annulus_width=3, enlarge = 1)

bead_set = pd.DataFrame(index=['no','img','lbl', 'dim','ratios','bkg','ref','code', 'cy5'])

labels = []
labels_annulus = []
bead_no = 0
for idx in xrange(bead_image_obj.f_size):
    bead_objects.find(bead_image_set_bf[idx])
    labels.append(bead_objects.labeled_mask)
    labels_annulus.append(bead_objects.labeled_annulus_mask)
    circles_dim = np.array(bead_objects.circles_dim)
    for lbl in np.arange(1, len(np.unique(labels[idx]))):
        bead_set[bead_no] = [bead_no, idx, lbl, circles_dim[lbl-1], None, None, None, None, None]
        bead_no += 1

fig = plt.figure()
fig.suptitle("Overlay Image Pre-filter")
plt.imshow(bead_objects.overlay_image(bead_image_set_bf[0], dim=np.vstack(bead_set.ix['dim'][bead_set.ix['img'] == 0])), cmap='Greys_r')
plt.draw()


#########################
### Unmix and Ratios  ###
"""[NOTES - Unmix and Ratios]
"""
print("[Unmix and get ratios]")
spec_unmix = ba.SpectralUnmixing(ref_data_object)
bead_no = 0
# Unmix images by least squares
for lbls_idx, lbls in enumerate(labels):
    #unmixed = ba.unmix(ref_data_object.ndata, bead_image_set_ln[lbls_idx])
    spec_unmix.unmix(bead_image_set_ln[lbls_idx])
    #unmixed = spec_unmix.ndata

    background = spec_unmix['Bkg']  # Device background
    reference = spec_unmix['Eu']  # Internal reference: Eu
    cy5 = bead_image_set_cy5[lbls_idx]  # cy5 channel
    cy5_back = ndi.labeled_comprehension(cy5, lbls, 0, np.median, float, -1)
    cy5 = np.subtract(cy5, cy5_back)
    # Ratio images
    ratio_Dy = spec_unmix['Dy'] / reference
    ratio_Sm = spec_unmix['Sm'] / reference
    ratio_Tm = spec_unmix['Tm'] / reference
    # Get ratios from images
    idx = np.arange(1, len(np.unique(lbls)))
    ratio_data = np.empty((len(idx), 3))
    ratio_data[:, 0] = ndi.labeled_comprehension(ratio_Dy, lbls, idx, np.median, float, -1)
    ratio_data[:, 1] = ndi.labeled_comprehension(ratio_Sm, lbls, idx, np.median, float, -1)
    ratio_data[:, 2] = ndi.labeled_comprehension(ratio_Tm, lbls, idx, np.median, float, -1)

    background_data = ndi.labeled_comprehension(background, lbls, idx, np.median, float, -1)
    reference_data = ndi.labeled_comprehension(reference, lbls, idx, np.median, float, -1)
    cy5_data = ndi.labeled_comprehension(cy5, labels_annulus[lbls_idx], idx, np.median, float, -1)

    for lbl in np.arange(1, len(np.unique(lbls))):
        bead_set[bead_no][['ratios', 'bkg', 'ref', 'cy5']] = [ratio_data[lbl-1], background_data[lbl-1], reference_data[lbl-1], cy5_data[lbl-1]]
        bead_no += 1

ratio_data = np.vstack(bead_set.ix['ratios'])
background_data = np.vstack(bead_set.ix['bkg'])[:,0]
reference_data = np.vstack(bead_set.ix['ref'])[:,0]
circles_dim = np.vstack(bead_set.ix['dim'])
cy5_data = np.vstack(bead_set.ix['cy5'])[:,0]

#########################
###    GMM - Split    ###
print("[Gaussian Mixture Modeling - Split]")
"""[NOTES - GMM]
"""

# GMM
gmix = GMM(n_components=2, covariance_type='full',
                   min_covar=1e-7, tol=1e-5, init_params='wc', params='wmc')
#gmix._means = [500,1200]
gmix.fit(reference_data.reshape(-1, 1))
predict = gmix.predict(reference_data.reshape(-1, 1))

ratio_data_low = ratio_data[predict == 0]
background_data_low =background_data[predict == 0]
reference_data_low = reference_data[predict == 0]
circles_dim_low = circles_dim[predict == 0]
cy5_data_low = cy5_data[predict == 0]

ratio_data_high = ratio_data[predict == 1]
background_data_high = background_data[predict == 1]
reference_data_high = reference_data[predict == 1]
circles_dim_high = circles_dim[predict == 1]
cy5_data_high = cy5_data[predict == 1]


#########################
###  Filtering - Low  ###
"""[NOTES - Filtering - Low]
"""
print("[Filtering - Low]")

# Filter objects based on background and reference
radius_min = 3
radius_max = 7
reference_std_factor_low = 2
reference_std_factor_high = 2
back_std_factor = 3
mean_back = np.mean(background_data_low)
std_back = np.std(background_data_low)
mean_reference = np.mean(reference_data_low)
std_reference = np.std(reference_data_low)
# Filter objects based on background and reference
size_filter = np.logical_and(circles_dim_low[:, 2] >= radius_min, circles_dim_low[:, 2] <= radius_max)
back_filter = np.logical_and(background_data_low < (mean_back + back_std_factor * std_back),
                                background_data_low > (mean_back - back_std_factor * std_back))
ref_filter = np.logical_and(reference_data_low > (mean_reference - reference_std_factor_low * std_reference),
                                        reference_data_low < (mean_reference + reference_std_factor_high * std_reference))
data_filter_list_low = np.argwhere(np.logical_and(size_filter, np.logical_and(back_filter, ref_filter)))[:, 0]

ratio_data_low_filt = ratio_data_low[data_filter_list_low]
background_data_low_filt = background_data_low[data_filter_list_low]
reference_data_low_filt = reference_data_low[data_filter_list_low]
circles_dim_low_filt = circles_dim_low[data_filter_list_low]
cy5_data_low_filt = cy5_data_low[data_filter_list_low]

print("Pre filter: %s" % len(ratio_data_low))
print("Post filter: %s" % len(ratio_data_low_filt))


#########################
###     ICP - Low     ###
"""[NOTES - ICP - Low]
"""
print("[Iterative Closest Point - Low]")

icp=ba.ICP(matrix_method='std', max_iter=100, tol=1e-4, outlier_pct=0.001)
icp.fit(ratio_data_low_filt, target)
data_icp_low = icp.transform(ratio_data_low_filt)
print("Tranformation matrix: ", icp.matrix)
print("Offset vector: ", icp.offset)

#########################
###     GMM - Low     ###
print("[Gaussian Mixture Modeling - Low]")
"""[NOTES - GMM]
"""

# GMM Setup
nclusters = len(target[:, 0])
naxes = len(target[0, :])
sigma = np.eye(naxes) * 1e-5

# GMM
gmix = GMM(n_components=nclusters, covariance_type='full',
                   min_covar=1e-7, tol=1e-5, init_params='', params='wmc')
gmix.means_ = target
gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
gmix._weights_ = np.tile(1 / 48, (nclusters))
gmix.fit(data_icp_low, target)
predict_low = gmix.predict(data_icp_low)

print("Number of unique beads found:", len(np.unique(predict_low)))

# Clustering graphs
colors = np.multiply(predict_low, 5)
colors_target = np.empty([48,3])
colors_target.fill(0)

fig = plt.figure()
fig.suptitle("Clustering pre-ICP & Pre-filter - Low")
ax = fig.gca(projection='3d')
ax.scatter(ratio_data_low[:, 0], ratio_data_low[:, 1], ratio_data_low[:, 2], alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

fig = plt.figure()
fig.suptitle("Clustering pre-ICP - Low")
ax = fig.gca(projection='3d')
ax.scatter(ratio_data_low_filt[:, 0], ratio_data_low_filt[:, 1], ratio_data_low_filt[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

fig = plt.figure()
fig.suptitle("Clustering post-ICP - Low")
ax = fig.gca(projection='3d')
ax.scatter(data_icp_low[:, 0], data_icp_low[:, 1], data_icp_low[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()


#########################
###  Filtering - High ###
"""[NOTES - Filtering - High]
"""
print("[Filtering - High]")

# Filter objects based on background and reference
radius_min = 3
radius_max = 7
reference_std_factor_low = 1.5
reference_std_factor_high = 2
back_std_factor = 5
mean_back = np.mean(background_data)
std_back = np.std(background_data)
mean_reference = np.mean(reference_data)
std_reference = np.std(reference_data)
# Filter objects based on background and reference
size_filter = np.logical_and(circles_dim_high[:, 2] >= radius_min, circles_dim_high[:, 2] <= radius_max)
back_filter = np.logical_and(background_data_high < (mean_back + back_std_factor * std_back),
                             background_data_high > (mean_back - back_std_factor * std_back))
ref_filter = np.logical_and(reference_data_high > (mean_reference - reference_std_factor_low * std_reference),
                            reference_data_high < (mean_reference + reference_std_factor_high * std_reference))
data_filter_list_high = np.argwhere(np.logical_and(size_filter, np.logical_and(back_filter, ref_filter)))[:, 0]

ratio_data_high_filt = ratio_data_high[data_filter_list_high]
background_data_high_filt =background_data_high[data_filter_list_high]
reference_data_high_filt = reference_data_high[data_filter_list_high]
circles_dim_high_filt = circles_dim_high[data_filter_list_high]
cy5_data_high_filt = cy5_data_high[data_filter_list_high]

print("Pre filter: %s" % len(ratio_data_high))
print("Post filter: %s" % len(ratio_data_high_filt))


#########################
###     ICP - High     ###
"""[NOTES - ICP - High]
"""
print("[Iterative Closest Point - High]")

icp=ba.ICP(matrix_method='std', max_iter=100, tol=1e-4, outlier_pct=0.001)
icp.fit(ratio_data_high_filt, target)
data_icp_high = icp.transform(ratio_data_high_filt)
print("Tranformation matrix: ", icp.matrix)
print("Offset vector: ", icp.offset)

#########################
###     GMM - High     ###
print("[Gaussian Mixture Modeling - High]")
"""[NOTES - GMM]
"""

# GMM Setup
nclusters = len(target[:, 0])
naxes = len(target[0, :])
sigma = np.eye(naxes) * 1e-5

# GMM
gmix = GMM(n_components=nclusters, covariance_type='full',
                   min_covar=1e-7, tol=1e-5, init_params='', params='wmc')
gmix.means_ = target
gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
gmix._weights_ = np.tile(1 / 48, (nclusters))
gmix.fit(data_icp_high, target)
predict_high = gmix.predict(data_icp_high)

print("Number of unique beads found:", len(np.unique(predict_high)))

# Clustering graphs
colors = np.multiply(predict_high, 5)
colors_target = np.empty([48,3])
colors_target.fill(0)

fig = plt.figure()
fig.suptitle("Clustering pre-ICP & Pre-filter - High")
ax = fig.gca(projection='3d')
ax.scatter(ratio_data_high[:, 0], ratio_data_high[:, 1], ratio_data_high[:, 2], alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

fig = plt.figure()
fig.suptitle("Clustering pre-ICP - High")
ax = fig.gca(projection='3d')
ax.scatter(ratio_data_high_filt[:, 0], ratio_data_high_filt[:, 1], ratio_data_high_filt[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()

fig = plt.figure()
fig.suptitle("Clustering post-ICP - High")
ax = fig.gca(projection='3d')
ax.scatter(data_icp_high[:, 0], data_icp_high[:, 1], data_icp_high[:, 2], c=colors, alpha=0.7)
ax.scatter(target[:, 0], target[:, 1], target[:, 2], c=colors_target, alpha=0.5, s=100)
ax.set_xlabel('Dy')
ax.set_ylabel('Sm')
ax.set_zlabel('Tm')
plt.draw()


#########################
###     Cy5 - Low     ###
print("[Get Cy5 Data - Low]")
"""[NOTES - Cy5]
"""

# Bead Frequency
beads_low = np.unique(predict_low)
# Unmix data
beads_freq_low = []
for idx, val in np.ndenumerate(beads_low):
    where = np.argwhere(predict_low == val)[:,0]
    beads_freq_low.append(len(where))

fig = plt.figure()
fig.suptitle("Bead Frequency")
plt.xticks(beads_low+1)
plt.bar(beads_low, beads_freq_low, align='center')
plt.draw()

# Unmix data
unmixed_low = []
unmixed_std_low = []
for idx, val in np.ndenumerate(beads_low):
    where = np.argwhere(predict_low == val)[:,0]
    unmixed_low.append(np.mean(cy5_data_low_filt[where]))
    unmixed_std_low.append(np.std(cy5_data_low_filt[where]) / np.sqrt(len(cy5_data_low_filt[where])) )
unmixed_low = np.array((unmixed_low))
unmixed_std_low = np.array((unmixed_std_low))


#########################
###     Cy5 - High    ###
print("[Get Cy5 Data - High]")
"""[NOTES - Cy5]
"""

# Bead Frequency
beads_high = np.unique(predict_high)
# Unmix data
beads_freq_high = []
for idx, val in np.ndenumerate(beads_high):
    where = np.argwhere(predict_high == val)[:,0]
    beads_freq_high.append(len(where))

fig = plt.figure()
fig.suptitle("Bead Frequency")
plt.xticks(beads_high+1)
plt.bar(beads_high, beads_freq_high, align='center')
plt.draw()

# Unmix data
unmixed_high = []
unmixed_std_high = []
for idx, val in np.ndenumerate(beads_high):
    where = np.argwhere(predict_high == val)[:,0]
    unmixed_high.append(np.mean(cy5_data_high_filt[where]))
    unmixed_std_high.append(np.std(cy5_data_high_filt[where]) / np.sqrt(len(cy5_data_high_filt[where])) )
unmixed_high = np.array((unmixed_high))
unmixed_std_high = np.array((unmixed_std_high))


# Show all images at once
plt.show()
