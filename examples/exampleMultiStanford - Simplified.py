# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : exampleMulti.py           [filename]
# description       : Multi file - Stanford Setup
# author            : Bjorn Harink              [Original author(s) of this file]
# credits           : Kurt Thorn, Huy Nguyen    [Contributors to this file]
# date              : 20161126                  [Initial date yyyymmdd]
# version update    : 20161126                  [Last version update yyyymmdd]
# version           : v0.1
# usage             : This is an example file for the Bead Analysis module.
# notes             : Multiple file code set.
# python_version    : 2.7

from photutils import source_properties, properties_table

# [TO-DO]

# [Modules]
# General Python
#import sys
#sys.path.append('./')
import random
# Data structures
import numpy as np
import pandas as pd
# Image Processing
from scipy import ndimage as ndi # Imaging
from sklearn.mixture import GaussianMixture
# Image display
from matplotlib import pyplot as plt
#import matplotlib.animation as manimation
#from mpl_toolkits.mplot3d import axes3d
#import matplotlib as mpl
#dpi = 200
#mpl.rc("savefig", dpi=dpi)
# Project
import bead_analysis as ba


##################################################
#                    [NOTES]                     # 
##################################################
"""[Notes]:
Stanford setup
"""


##################################################
#                   [SETINGS]                    # 
##################################################
"""[Notes SETINGS]:
Notes here...
"""

# Reference image location
REF_FILES = {"Dy" : r"Z:\Bjorn\[Stanford]\Ref Spectra Stanford\Dy_solo_20160915_3\Dy_solo_20160915_3_MMStack_Pos0.ome.tif",
             "Sm" : r"Z:\Bjorn\[Stanford]\Ref Spectra Stanford\Sm_solo_20160915_1\Sm_solo_20160915_1_MMStack_Pos0.ome.tif",
             "Tm" : r"Z:\Bjorn\[Stanford]\Ref Spectra Stanford\Tm_1_2_solo_20160915_4\Tm_1_2_solo_20160915_4_MMStack_Pos0.ome.tif",
             "Eu" : r"Z:\Bjorn\[Stanford]\Ref Spectra Stanford\Eu_solo_20160915_3\Eu_solo_20160915_3_MMStack_Pos0.ome.tif"}

TRP_FILE = r"Z:\Bjorn\[Stanford]\Ref Spectra Stanford\20170124_Blank_W_1\20170124_Blank_W_1_MMStack_Pos0.ome.tif"

# Target file location
TARGET_FILE = r"Z:\Code Sets\20160226_DySmTm_48Codes.csv"

# General Region or interest
# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X). Pandas includes stop element!
CROPx = slice(250, 750)
CROPy = slice(250, 750)
# Reference images ROI
CROPx_ref = slice(600, 1550)
CROPy_ref = slice(600, 1550)

# Bead image set file location
BEAD_IMAGE_FOLDER = r"Z:\Bjorn\[Stanford]\CN Biotin"
BEAD_IMAGE_PATTERN = "20161209_HQN71_CN_250nMc_*"

# Background image set file location
BACK_FILE = r"Z:\Bjorn\[Stanford]\20161118 48B20160929\20161118_48B_20160929_1to48b_2\20161118_48B_20160929_1to48b_2_MMStack_Pos0.ome.tif"


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
target = target[:, 1:4]  # Target set to Dy, Sm, Tm (no CeTb)


#########################
### Reference Spectra ###
"""[NOTES - Reference Spectra]
Get reference spectra of solo lanthanide beads
"""
print("[Creating reference spectra]")

BACK_CROPx = slice(339, 445)
BACK_CROPy = slice(420, 508)
dark_noise = 99
spec_object = ba.simp.ReferenceSpectra(files = REF_FILES, 
                                       object_channel = 'Brightfield', 
                                       channels = ['435','780'], 
                                       find_param = [14, 16, 10, 6], 
                                       dark_noise = dark_noise)
spec_object.crop_x = CROPx_ref
spec_object.crop_y = CROPy_ref
spec_object.set_back(BACK_FILE, ['l-435','l-780'], BACK_CROPx, BACK_CROPy)
ref_data_object = spec_object.output

# Trp
name = "Tp"
print("Spectrum: %s" % name)
CROPx_ref_tp = slice(250, 750)
CROPy_ref_tp = slice(250, 750)
ref_objects_tp = ba.FindBeads(min_r=7, max_r=9, param_1=10, param_2=6)
ref_img_obj = ba.ImageSetRead(TRP_FILE)
ref_objects_tp.find(ref_img_obj['Brightfield',CROPy_ref_tp,CROPx_ref_tp])
channels = ref_img_obj['l-435':'l-780',CROPy_ref_tp,CROPx_ref_tp]
ref_data_tmp = spec_object.get_spectrum(dark_noise, channels, ref_objects_tp.labeled_mask)
ref_data_object.spec_add(name, data=ref_data_tmp)
    
ref_data_object.plot()


#########################
###    Bead Objects   ###
"""[NOTES - Bead Objects]
"""
print("[Load bead images and find objects]")

bead_image_files = ba.ImageSetRead.scan_path(BEAD_IMAGE_FOLDER, BEAD_IMAGE_PATTERN)
bead_image_obj = ba.ImageSetRead(bead_image_files)
bead_image_obj.crop_x = CROPx
bead_image_obj.crop_y = CROPy
bead_image_set_bf = bead_image_obj[:,'Brightfield']
bead_image_set_ln = bead_image_obj[:,'l-435':'l-780']

# Bead search and filter parameters
bead_objects = ba.FindBeads(min_r=5, max_r=8, min_dist=9, param_1=10, param_2=7, annulus_width=3, enlarge = 1)
reference_std_factor_low = 1.5
reference_std_factor_high = 2
back_std_factor = 3

bead_set = pd.DataFrame(columns=['img','lbl', 'dim_x', 'dim_y', 'dim_r','bkg','ref',
                                 'rat_dy','rat_sm','rat_tm'])

labels = []
labels_annulus = []
bead_no = 0
for idx in xrange(bead_image_obj.f_size):
    bead_objects.find(bead_image_set_bf[idx])
    if bead_objects.labeled_mask is None:
        continue
    labels.append(bead_objects.labeled_mask)
    labels_annulus.append(bead_objects.labeled_annulus_mask)
    circles_dim = np.array(bead_objects.circles_dim)
    for lbl in np.arange(1, len(np.unique(labels[idx]))):
        bead_set.loc[bead_no,('img', 'lbl', 'dim_x', 'dim_y', 'dim_r')] = \
            [idx, lbl, circles_dim[lbl-1, 0], circles_dim[lbl-1, 1], circles_dim[lbl-1, 2]]
        bead_no += 1


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
    ratio_Tm = spec_unmix['Tm'] / reference
    # Get ratios from images
    idx = np.arange(1, len(np.unique(lbls)))
    ratio_data = np.empty((len(idx), target[0].size))
    ratio_data[:, 0] = ndi.labeled_comprehension(ratio_Dy, lbls, idx, np.median, float, -1)
    ratio_data[:, 1] = ndi.labeled_comprehension(ratio_Sm, lbls, idx, np.median, float, -1)
    ratio_data[:, 2] = ndi.labeled_comprehension(ratio_Tm, lbls, idx, np.median, float, -1)

    background_data = ndi.labeled_comprehension(background, lbls, idx, np.median, float, -1)
    reference_data = ndi.labeled_comprehension(reference, lbls, idx, np.median, float, -1)

    for lbl in np.arange(1, len(np.unique(lbls))):
        bead_set.loc[bead_no,('rat_dy', 'rat_sm', 'rat_tm', 'bkg', 'ref')] = \
            [ratio_data[lbl-1,0], ratio_data[lbl-1,1], ratio_data[lbl-1,2], background_data[lbl-1], reference_data[lbl-1]]
        bead_no += 1


#########################
###     Filtering     ###
"""[NOTES - Filtering]
"""
print("[Filtering]")

mask_bkg    = ( (bead_set.bkg > (bead_set.bkg.mean() - back_std_factor * bead_set.bkg.std())) &\
                (bead_set.bkg < (bead_set.bkg.mean() + back_std_factor * bead_set.bkg.std())) )
mask_ref    = ( (bead_set.ref > (bead_set.ref.mean() - reference_std_factor_low * bead_set.ref.std())) &\
                (bead_set.ref < (bead_set.ref.mean() + reference_std_factor_high * bead_set.ref.std())) )
filter_all = (mask_bkg & mask_ref)

print("Pre filter: %s" % bead_set.index.size)
print("Post filter: %s" % bead_set[filter_all].index.size)


#########################
###        ICP        ###
"""[NOTES - ICP]
"""
print("[Iterative Closest Point]")

icp=ba.ICP(matrix_method='std', max_iter=100, tol=1e-4, outlier_pct=0.01, train=False)
icp.fit(bead_set.loc[filter_all, ('rat_dy', 'rat_sm', 'rat_tm')], target)
bead_set = bead_set.join(icp.transform())
print("Tranformation matrix: ", icp.matrix)
print("Offset vector: ", icp.offset)


#########################
###        GMM        ###
"""[NOTES - GMM]
"""
print("[Gaussian Mixture Modeling]")

gmix = ba.Classify(target, tol=1e-5, min_covar=1e-7, sigma=1e-5, train=False)
gmix.decode(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')])
bead_set = bead_set.join(gmix.output)
print("Number of unique codes found:", gmix.found)
print("Missing codes:", gmix.missing)


#########################
###      Inspect      ###
"""[NOTES - Inspect]
"""
print("[Inspect]")

# Clustering pre-ICP & Pre-filter
ba.inspect.Cluster.scatter(bead_set.loc[:, ('rat_dy', 'rat_sm', 'rat_tm')].values, 
                           target, 
                           title="Clustering pre-ICP & Pre-filter", 
                           axes_names=['Dy', 'Sm', 'Tm'])
# Clustering pre-ICP & Filtered
ba.inspect.Cluster.scatter(bead_set.loc[filter_all, ('rat_dy', 'rat_sm', 'rat_tm')].values, 
                           target, 
                           bead_set.loc[filter_all, ('code')].values, 
                           title="Clustering pre-ICP & Filtered", 
                           axes_names=['Dy', 'Sm', 'Tm'])
# Clustering post-ICP & Filtered
ba.inspect.Cluster.scatter(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')].values, 
                           target, 
                           bead_set.loc[filter_all, ('code')].values, 
                           title="Clustering post-ICP & Filtered", 
                           axes_names=['Dy', 'Sm', 'Tm'])
# Clustering post-ICP & Filtered & Confidence Filtered
ba.inspect.Cluster.scatter(bead_set.loc[(bead_set.confidence > 0), ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')].values, 
                           target, 
                           bead_set.loc[(bead_set.confidence > 0), ('code')].values, 
                           title="Clustering post-ICP & Filtered & Confidence Filtered", 
                           axes_names=['Dy', 'Sm', 'Tm'])

# Single code chart
code_no = 22 # Starts at 0!
code_data = np.vstack(bead_set.loc[(bead_set.code == code_no), ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')].as_matrix())
ba.inspect.Cluster.scatter(code_data, target, title="Clustering post-ICP", axes_names=['Dy', 'Sm', 'Tm'])

# Beads per code distribution
bead_set.loc[(bead_set.confidence > 0), ('code')].hist(bins=48)

# Show all images at once
plt.show()