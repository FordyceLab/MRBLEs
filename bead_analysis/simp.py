# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division
from __builtin__ import staticmethod, property

# [File header]     | Copy and edit for each file in this project!
# title             : simp.py
# description       : Simplified and condensed function.
# author            : Bjorn Harink
# credits           : 
# date              : 20170219
# version update    : 20170219
# version           : v0.1
# usage             : As module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [Modules]
# General Python
import sys
import os
import types
import warnings
from math import sqrt
# Data Structure
import numpy as np
import pandas as pd
# Image Processing
from scipy import ndimage as ndi
# Graphs
from matplotlib import pyplot as plt
# Project
from bead_analysis.core import *
from bead_analysis.data import *


class ReferenceSpectra(object):
    def __init(self, files, object_image, dark_noise=0, crop_x=None, crop_y=None):
        self.files = files
        self._object_image = object_image
        self._darknoise = dark_noise
        self._crop_x = crop_x
        self._crop_y = crop_y

        self.ref_data_object = Spectra()

    def find(self):
        self.ref_objects = FindBeads(min_r=14, 
                                        max_r=16, 
                                        param_1=10, 
                                        param_2=6)

    def run(self):
        pass

    def _slice(value_1, value_2):
        return slice(value_1, value_2)

    def get_spectra(self):
        for name, file in REF_FILES.iteritems():
            print("Spectrum: %s" % name)
            img_obj = ImageSetRead(file)
    
    @staticmethod
    def get_spectrum(file, object_image, dark_noise, channels, mask):
        data = np.array([ndi.median(ch, mask) for ch in channels])
        data -= dark_noise              # Dark noise subtract
        data /= data.sum()      # Normalize
        return data

    def plot_image(self, name=''):
        fig = plt.figure()
        fig.suptitle("Overlay Image ref %s:" % name)
        plt.imshow(ref_objects.overlay_image(ref_img_obj['Brightfield',CROPy_ref,CROPx_ref], dim = ref_objects.circles_dim), cmap='Greys_r')
        plt.draw()

    def plot_spectra(self):
        self.ref_data_object.plot()

class BeadDecode(object):
    def __init__(self, spectra, target):
        self._spectra = spectra
        self._target = target
        self._result = None

    def icp(self):
        pass

    def gmm(self):
        # GMM Setup
        nclusters = len(self._target[:, 0])
        naxes = len(self.target[0, :])
        sigma = np.eye(naxes) * 1e-5
        weights = np.tile(1 / nclusters, (nclusters))
        covars = np.tile(sigma, (nclusters, 1, 1))
        covars_inv = np.linalg.inv(covars)

        gmix = GaussianMixture(covariance_type='full', tol=1e-5, reg_covar=1e-7, 
                               n_components=nclusters, 
                               means_init=target, 
                               weights_init = weights, 
                               precisions_init=covars_inv)
        gmix.fit(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')].values, target)
        predict = gmix.predict(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')].values)
        predict_pb = gmix.predict_proba(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')].values)

    def assay(self, image, mask):
        idx = np.arange(1, len(np.unique(mask)))
        data = ndi.labeled_comprehension(image, mask, idx, np.median, float, -1)

    def decode(self):
        self.icp()
        self.gmm()

    @property
    def result(self):
        return self._result

def load_dy_sm_tm(BEAD_IMAGE_FOLDER, BEAD_IMAGE_PATTERN, 
                  CROPy, CROPx, 
                  ref_data_object, 
                  bead_objects, 
                  target,
                  radius_min, radius_max,
                  back_std_factor,
                  reference_std_factor_low,
                  reference_std_factor_high,
                  icp, gmix):
    bead_image_files = ImageSetRead.scan_path(BEAD_IMAGE_FOLDER, BEAD_IMAGE_PATTERN)
    bead_image_obj = ImageSetRead(bead_image_files)
    bead_image_set_bf = bead_image_obj[:,'Brightfield',CROPy,CROPx]
    bead_image_set_ln = bead_image_obj[:,'l-435':'l-780',CROPy,CROPx]

    bead_set = pd.DataFrame(columns=['img', 
                                     'lbl', 
                                     'dim_x', 
                                     'dim_y', 
                                     'dim_r',
                                     'bkg',
                                     'ref',
                                     'rat_dy',
                                     'rat_sm',
                                     'rat_tm'])

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

    spec_unmix = SpectralUnmixing(ref_data_object)
    bead_no = 0
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

    # Make filter mask
    mask_size   = ( (bead_set.dim_r >= radius_min) & (bead_set.dim_r <= radius_max) )
    mask_bkg    = ( (bead_set.bkg > (bead_set.bkg.mean() - back_std_factor * bead_set.bkg.std())) &\
                    (bead_set.bkg < (bead_set.bkg.mean() + back_std_factor * bead_set.bkg.std())) )
    mask_ref    = ( (bead_set.ref > (bead_set.ref.mean() - reference_std_factor_low * bead_set.ref.std())) &\
                    (bead_set.ref < (bead_set.ref.mean() + reference_std_factor_high * bead_set.ref.std())) )
    filter_all = (mask_size & mask_bkg & mask_ref)

    print("Pre filter: %s" % bead_set.index.size)
    print("Post filter: %s" % bead_set[filter_all].index.size)

    # ICP
    icp.fit(bead_set.loc[filter_all, ('rat_dy', 'rat_sm', 'rat_tm')], target)
    bead_set = bead_set.join(icp.transform())
    print("Tranformation matrix: ", icp.matrix)
    print("Offset vector: ", icp.offset)

    # GMM
    gmix.decode(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp', 'rat_tm_icp')])
    bead_set = bead_set.join(gmix.output)
    print("Number of unique codes found:", gmix.found)
    print("Missing codes:", gmix.missing)

    return bead_set

def load_dy_sm(BEAD_IMAGE_FOLDER, BEAD_IMAGE_PATTERN, 
               CROPy, CROPx, 
               ref_data_object, 
               bead_objects, 
               target,
               radius_min, radius_max,
               back_std_factor,
               reference_std_factor_low,
               reference_std_factor_high,
               icp, gmix):
    bead_image_files = ImageSetRead.scan_path(BEAD_IMAGE_FOLDER, BEAD_IMAGE_PATTERN)
    bead_image_obj = ImageSetRead(bead_image_files)
    bead_image_set_bf = bead_image_obj[:,'Brightfield',CROPy,CROPx]
    bead_image_set_ln = bead_image_obj[:,'l-435':'l-780',CROPy,CROPx]

    bead_set = pd.DataFrame(columns=['img', 
                                     'lbl', 
                                     'dim_x', 
                                     'dim_y', 
                                     'dim_r',
                                     'bkg',
                                     'ref',
                                     'rat_dy',
                                     'rat_sm'])

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

    spec_unmix = SpectralUnmixing(ref_data_object)
    bead_no = 0
    for lbls_idx, lbls in enumerate(labels):
        spec_unmix.unmix(bead_image_set_ln[lbls_idx])

        background = spec_unmix['Bkg']  # Device background
        reference = spec_unmix['Eu']  # Internal reference: Eu
        # Ratio images
        ratio_Dy = spec_unmix['Dy'] / reference
        ratio_Sm = spec_unmix['Sm'] / reference
        # Get ratios from images
        idx = np.arange(1, len(np.unique(lbls)))
        ratio_data = np.empty((len(idx), target[0].size))
        ratio_data[:, 0] = ndi.labeled_comprehension(ratio_Dy, lbls, idx, np.median, float, -1)
        ratio_data[:, 1] = ndi.labeled_comprehension(ratio_Sm, lbls, idx, np.median, float, -1)

        background_data = ndi.labeled_comprehension(background, lbls, idx, np.median, float, -1)
        reference_data = ndi.labeled_comprehension(reference, lbls, idx, np.median, float, -1)

        for lbl in np.arange(1, len(np.unique(lbls))):
            bead_set.loc[bead_no,('rat_dy', 'rat_sm', 'bkg', 'ref')] = \
                [ratio_data[lbl-1,0], ratio_data[lbl-1,1], background_data[lbl-1], reference_data[lbl-1]]
            bead_no += 1

    # Make filter mask
    mask_size   = ( (bead_set.dim_r >= radius_min) & (bead_set.dim_r <= radius_max) )
    mask_bkg    = ( (bead_set.bkg > (bead_set.bkg.mean() - back_std_factor * bead_set.bkg.std())) &\
                    (bead_set.bkg < (bead_set.bkg.mean() + back_std_factor * bead_set.bkg.std())) )
    mask_ref    = ( (bead_set.ref > (bead_set.ref.mean() - reference_std_factor_low * bead_set.ref.std())) &\
                    (bead_set.ref < (bead_set.ref.mean() + reference_std_factor_low * bead_set.ref.std())) )
    filter_all = (mask_size & mask_bkg & mask_ref)

    print("Pre filter: %s" % bead_set.index.size)
    print("Post filter: %s" % bead_set[filter_all].index.size)

    # ICP
    icp.fit(bead_set.loc[filter_all, ('rat_dy', 'rat_sm')], target)
    bead_set = bead_set.join(icp.transform())
    print("Tranformation matrix: ", icp.matrix)
    print("Offset vector: ", icp.offset)

    # GMM
    gmix.decode(bead_set.loc[filter_all, ('rat_dy_icp', 'rat_sm_icp')])
    bead_set = bead_set.join(gmix.output)
    print("Number of unique codes found:", gmix.found)
    print("Missing codes:", gmix.missing)

    return bead_set