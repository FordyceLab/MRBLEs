# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : exampleMulti.py           [filename]
# description       : Main file to call and use BeadKinetics module
# author            : Bjorn Harink              [Original author(s) of this file]
# credits           : Kurt Thorn, Huy Nguyen    [Contributors to this file]
# date              : 20160308                  [Initial date yyyymmdd]
# version update    : 20160808                  [Last version update yyyymmdd]
# version           : v0.4
# usage             : This is an example file for the Bead Analysis module.
# notes             : Multiple file code set.
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General Python
import sys
sys.path.append('./')
import random
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
# Image display
import matplotlib as mpl
dpi = 300
mpl.rc("savefig", dpi=dpi)


##################################################
#                    [NOTES]                     # 
##################################################
"""[Notes]:
Multiple position image set (grid).
"""


##################################################
#                   [SETINGS]                    # 
##################################################
"""[Notes SETINGS]:
Notes here...
"""
# Reference image location
REF_FILES = {#"Trp":"Z:/Huy/20160314_WBlank/20160314_W_blank_4/20160314_W_blank_4_MMStack.ome.tif",
             "Dy" : "Z:/Huy/20160315_Solo/20160315_Solo_Dy_9/20160315_Solo_Dy_9_MMStack.ome.tif",
             "Sm" : "Z:/Huy/20160315_Solo/20160315_Solo_Sm_4/20160315_Solo_Sm_4_MMStack.ome.tif",
             #"Tm" : "Z:/Huy/20160315_Solo/20160315_Solo_Tm_4/20160315_Solo_Tm_4_MMStack.ome.tif",
             "Eu" : "Z:/Huy/20160315_Solo/20160315_Solo_Eu_4/20160315_Solo_Eu_4_MMStack.ome.tif"}

# Target file location
TARGET_FILE = r"Z:\Code Sets\20161206_DySm_24_codes.csv"

# General Region or interest
# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X). Pandas includes stop element!
CROPy = slice(80, 420)
CROPx = slice(80, 420)

# Bead image set file location
BEAD_IMAGE_FOLDER = r"Z:\Bjorn\20161215 - 24 Code set test"
BEAD_IMAGE_PATTERN = "20161215_24Codes_Bin2_All_3*"

# Background image set file location
BACK_FILE = r"Z:\Bjorn\20161215 - 24 Code set test\20161215_24Codes_Bin2_All_1\20161215_24Codes_Bin2_All_1_MMStack.ome.tif"
# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X). Pandas includes stop element!
BACK_CROPy = slice(100, 400)
BACK_CROPx = slice(150, 250)


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
target = target[:, 1:3]  # Target set to Dy, Sm (no CeTb and Tm)


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
# Show Bkg image region
fig = plt.figure()
fig.suptitle("Bkg Image region:")
plt.imshow(bkg_img_obj['BF', BACK_CROPy, BACK_CROPx], cmap='Greys_r')
plt.draw()

# Plot reference spectrum
ref_data_object.plot()

bead_objects = ba.FindBeads(min_r=5, max_r=7, min_dist=10, param_1=10, param_2=6, annulus_width=2, enlarge = 1)
radius_min = 4.9
radius_max = 7.1
reference_std_factor_low = 1.5
reference_std_factor_high = 1.5
back_std_factor = 3

#########################
###    Bead Objects   ###
"""[NOTES - Bead Objects]
"""
print("[Load bead images and find objects]")

bead_image_files = ba.ImageSetRead.scan_path(BEAD_IMAGE_FOLDER, BEAD_IMAGE_PATTERN)
bead_image_obj = ba.ImageSetRead(bead_image_files, all=True)
bead_image_set_bf = bead_image_obj[:,'BF',CROPy,CROPx]
bead_image_set_ln = bead_image_obj[:,'Ex292-Em435':'Ex292-Em650',CROPy,CROPx]

bead_set = pd.DataFrame(columns=['no','img','lbl', 'dim','ratios','bkg','ref','code'])

labels = []
labels_annulus = []
img_set = []
bead_no = 0
for idx in xrange(len(bead_image_set_bf)):
    bead_objects.find(bead_image_set_bf[idx])
    if bead_objects.labeled_mask is None:
        continue
    img_set.append(idx)
    labels.append(bead_objects.labeled_mask)
    labels_annulus.append(bead_objects.labeled_annulus_mask)
    circles_dim = np.array(bead_objects.circles_dim)
    for lbl in np.arange(1, len(np.unique(labels[idx]))):
        bead_set.loc[bead_no] = [bead_no, idx, lbl, circles_dim[lbl-1], None, None, None, None]
        bead_no += 1

for x in xrange(3):
    idx = random.choice(img_set)
    fig = plt.figure()
    fig.suptitle("Overlay Image Pre-filter image #: %s" % idx)
    plt.imshow(bead_objects.overlay_image(bead_image_set_bf[idx], dim=bead_set.dim[bead_set['img'] == idx], annulus=True))
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
    spec_unmix.unmix(bead_image_set_ln[lbls_idx])

    background = spec_unmix['Bkg']  # Device background
    reference = spec_unmix['Eu']  # Internal reference: Eu
    # Ratio images
    ratio_Dy = spec_unmix['Dy'] / reference
    ratio_Sm = spec_unmix['Sm'] / reference
    #ratio_Tm = spec_unmix['Tm'] / reference
    # Get ratios from images
    idx = np.arange(1, len(np.unique(lbls)))
    ratio_data = np.empty((len(idx), target[0].size))
    ratio_data[:, 0] = ndi.labeled_comprehension(ratio_Dy, lbls, idx, np.median, float, -1)
    ratio_data[:, 1] = ndi.labeled_comprehension(ratio_Sm, lbls, idx, np.median, float, -1)
    #ratio_data[:, 2] = ndi.labeled_comprehension(ratio_Tm, lbls, idx, np.median, float, -1)

    background_data = ndi.labeled_comprehension(background, lbls, idx, np.median, float, -1)
    reference_data = ndi.labeled_comprehension(reference, lbls, idx, np.median, float, -1)

    for lbl in np.arange(1, len(np.unique(lbls))):
        bead_set.loc[bead_no,('ratios', 'bkg', 'ref')] = [ratio_data[lbl-1], background_data[lbl-1], reference_data[lbl-1]]
        bead_no += 1

#########################
###     Filtering     ###
"""[NOTES - Filtering]
"""
print("[Filtering]")

ratio_data = np.vstack(bead_set['ratios'])
background_data = bead_set['bkg'].values
reference_data = bead_set['ref']
circles_dim = np.vstack(bead_set['dim'])

# Filter objects based on background and reference
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

imgs = xrange(3)
for idx in imgs:
    fig = plt.figure()
    fig.suptitle("Overlay Image Post-filter")
    plt.imshow(bead_objects.overlay_image(bead_image_set_bf[idx], dim=bead_set.dim[data_filter_list][bead_set['img'] == idx], annulus=True), cmap='Greys_r')
    plt.draw()

#########################
###        ICP        ###
"""[NOTES - ICP]
"""
print("[Iterative Closest Point]")

matrix = [2, 10]
#matrix = 'max'
#offset = [0, 0.04]
offset = None
icp=ba.ICP(matrix_method=matrix, max_iter=100, offset=offset, tol=1e-4, outlier_pct=.001)
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
gmix = GMM(n_components=nclusters, covariance_type='full',
                   min_covar=1e-7, tol=1e-5, init_params='', params='wmc')
gmix.means_ = target
gmix.covars_ = np.tile(sigma, (nclusters, 1, 1))
gmix._weights_ = np.tile(1 / nclusters, (nclusters))
gmix.fit(data_icp, target)
predict = gmix.predict(data_icp)

print("Number of unique beads found:", len(np.unique(predict)))

ba.inspect.Cluster.scatter(ratio_data, target, title="Clustering pre-ICP & Pre-filter", axes_names=['Dy', 'Sm'])
ba.inspect.Cluster.scatter(data_filtered, target, predict, title="Clustering pre-ICP", axes_names=['Dy', 'Sm'])
ba.inspect.Cluster.scatter(data_icp, target, predict, title="Clustering post-ICP", axes_names=['Dy', 'Sm'])

# Show all images at once
plt.show()