# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quality Control Report Classes and Functions
============================================

This file stores the quality control report classes and functions for the MRBLEs Analysis
module.
"""

# [Future imports]
from __future__ import (absolute_import, division, print_function)
from builtins import (super, range, int, object)

# [File header]     | Copy and edit for each file in this project!
# title             : inpect.py
# description       : MRBLEs - Inspection
# author            : Bjorn Harink
# credits           : Kurt Thorn
# date              : 20161114

# [Modules]
# General Python
import sys
import os
import itertools
import random
import time
import warnings
# import multiprocessing as mp
# Data
import numpy as np
import pandas as pd
import xarray as xr
# Imaging
import cv2
# Image display
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import seaborn as sns
# Intra-Package dependencies
from mrbles.data import TableDataFrame


# Functions


def circle_overlay(image, dims=None, ring=None):
    """Circle Overlay Image.

    Overlay image with circles of labeled mask.
    """
    img = image.copy()
    if dims is not None:
        for dim_idx, dim in enumerate(dims):
            if ring is not None:
                if type(ring) is int:
                    cv2.circle(img, (int(ring[dim_idx][0]), int(ring[dim_idx][1])), int(
                        ceil(ring[dim_idx][2])), (0, 255, 0), 1)
                else:
                    for dim_r in ring:
                        cv2.circle(img, (int(dim_r[0]), int(dim_r[1])), int(
                            ceil(dim_r[2])), (0, 255, 0), 1)
            cv2.circle(img, (int(dim[0]), int(dim[1])),
                       int(ceil(dim[2])), (0, 255, 0), 1)
    plt.imshow(img)
    return img

def image_overlay(image, image_blend, alpha=0.2, cmap_image='Greys_r', cmap_blend='jet'):
    """Overlay 2 Images.
    Overlay of 2 images using alpha blend.
    """
    plt.imshow(image, cmap=cmap_image)
    plt.imshow(image_blend, cmap=cmap_blend, interpolation='none', alpha=alpha)


# Classes


class ClusterCheck(TableDataFrame):
    """Cluster check reporting."""

    def __init__(self, decode_object, *args, **kwargs):
        super(ClusterCheck, self).__init__(*args, **kwargs)
        self._dataframe = decode_object.data
        self._decode = decode_object
        init_notebook_mode(connected=True)

    def _set_min_prob(self, min_prob):
        if min_prob is not None:
            p_data = self._dataframe[self._dataframe.prob >= min_prob]
        else:
            p_data = self._dataframe
        return p_data

    def _ellipses(self, means, covars, confidence):
        data = []
        for i, (mean, covar) in enumerate(zip(means, covars)):
            v, w = np.linalg.eigh(covar)
            sigma = (1 - confidence) / 0.0255102040816327
            v = sigma * 2. * np.sqrt(v)
            angle = np.arctan2(w[1, 0], w[0, 0])
            angle = np.degrees(angle)

            # Plot an ellipse to show the Gaussian component
            a = v[1]
            b = v[0]
            x_origin = mean[0]
            y_origin = mean[1]

            theta = np.radians(np.arange(0.0, 360.0, 1.0))
            x = a * (np.cos(theta))
            y = b * (np.sin(theta))
            rtheta = np.radians(angle + 90)
            R = np.array([
                 [np.cos(rtheta), -np.sin(rtheta)],
                 [np.sin(rtheta), np.cos(rtheta)]
                ])

            x, y = np.dot(R, np.array([x, y]))
            x += x_origin
            y += y_origin

            elle = go.Scatter(x=x, y=y,
                              name='CI %0.2f' % confidence,
                              showlegend=False,
                              hoverinfo='none',
                              line=dict(color='Grey',
                                        width=1,
                                        dash='dot'))
            data.append(elle)
        return data

    # TODO: Add possibility to choose dimensions.
    def plot_3D(self, min_prob=None):
        """Plot ratio clusters in 3D.

        Parameters
        ----------
        min_prob : float
            Set minimal probability level.
        """
        bead_set = self._set_min_prob(min_prob)
        target = self._decode._target.values
        colors = np.multiply(
            bead_set.code.values.astype(int),
            np.ceil(255 / len(target))
        )
        dims_pre_icp = [dim for dim in bead_set.columns if
                        ('_ratio.mask_inside' in dim) and
                        (dim[len(dim) - 18:] == '_ratio.mask_inside')]
        dims_post_icp = [dim for dim in bead_set.columns
                         if '_ratio.mask_inside_icp' in dim]
        dims_names = [dim.replace('.mask_inside', '') for dim in dims_pre_icp]
        if len(dims_names) > 3:
            warnings.warn("Set has more than 3 dimensions, only first 3 dimensions are used for 3D plot.",
                          UserWarning)
        if len(dims_names) < 3:
            ValueError("Set has less than 3 dimensions.")

        bead_ratios_pre = go.Scatter3d(
            name='Bead ratios - Pre-ICP',
            x=bead_set[dims_pre_icp[0]].values,
            y=bead_set[dims_pre_icp[1]].values,
            z=bead_set[dims_pre_icp[2]].values,
            text=bead_set['code'].values + 1,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='Rainbow',
                opacity=0.6,
                symbol="circle-open"
            )
        )

        bead_ratios_post = go.Scatter3d(
            name='Bead ratios - Post-ICP',
            x=bead_set[dims_post_icp[0]].values,
            y=bead_set[dims_post_icp[1]].values,
            z=bead_set[dims_post_icp[2]].values,
            text=bead_set['code'].values + 1,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='Rainbow',
                opacity=0.6
            )
        )

        target_ratios = go.Scatter3d(
            name='Target ratios',
            x=target[:, 0],
            y=target[:, 1],
            z=target[:, 2],
            text=list(range(1, len(target) + 1)),
            mode='markers',
            marker=dict(
                size=4,
                color='black',
                symbol="diamond"
            )
        )

        mean_ratios = go.Scatter3d(
            name='GMM mean ratios',
            x=self._decode.settings.gmm.means[:, 0],
            y=self._decode.settings.gmm.means[:, 1],
            z=self._decode.settings.gmm.means[:, 2],
            text=list(range(1, len(target) + 1)),
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                opacity=0.5,
                symbol="diamond"
            )
        )

        data = [bead_ratios_pre, bead_ratios_post, target_ratios, mean_ratios]
        layout = go.Layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    title=dims_names[0]
                ),
                yaxis=dict(
                    title=dims_names[1]
                ),
                zaxis=dict(
                    title=dims_names[2]
                )
            ),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)

    def plot_2D(self, colors, ci_trace=None, min_prob=None):
        """Plot 2D clusters."""
        mrbles_data = self._set_min_prob(min_prob)
        color_exclude = [color for color in self._decode._target.columns if
                         color not in colors][0]
        mrbles_data = mrbles_data[mrbles_data[('info.%s' % color_exclude)] == 0]
        levels = (self._decode._target[color_exclude] == 0)
        target_ratios = self._decode._target.loc[levels, colors].values
        codes_target = self._decode._target.loc[levels].index.values
        colors_data = np.multiply(mrbles_data.code.values,
                                  (255 / len(target_ratios))).astype(int)
        codes_found = np.unique(mrbles_data.code)
        color_pos = [np.argwhere(
            self._decode._target.columns == colors[0])[0, 0]]
        color_pos.append(np.argwhere(
            self._decode._target.columns == colors[1])[0, 0])
        means = self._decode.settings.gmm.means[levels][:, color_pos]
        covars = self._decode.settings.gmm._gmix.covariances_[levels][:, color_pos][..., color_pos]
        data = []
        if ci_trace is not None:
            data.extend(self._ellipses(means, covars, ci_trace))

        bead_ratios_plot = go.Scatter(
            name='Bead ratios',
            x=mrbles_data[('%s_ratio.mask_inside_icp' % colors[0])].values,
            y=mrbles_data[('%s_ratio.mask_inside_icp' % colors[1])].values,
            text=mrbles_data[('code')].values + 1,
            mode='markers',
            marker=dict(
                size=3,
                color=colors_data,
                colorscale='Rainbow',
                opacity=0.6
            )
        )
        data.append(bead_ratios_plot)

        target_ratios_plot = go.Scatter(
            name='Target ratios',
            x=target_ratios[:, 0],
            y=target_ratios[:, 1],
            text=codes_target + 1,
            mode='markers',
            marker=dict(
                size=4,
                color='black',
                symbol="diamond"
            )
        )
        data.append(target_ratios_plot)

        mean_ratios_plot = go.Scatter(
            name='GMM mean ratios',
            x=means[:, 0],
            y=means[:, 1],
            text=codes_found + 1,
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                opacity=0.5,
                symbol="diamond"
            )
        )
        data.append(mean_ratios_plot)

        layout = go.Layout(
            showlegend=True,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        fig1 = go.Figure(data=data, layout=layout)
        iplot(fig1)

    @staticmethod
    def scatter(data, target, codes=None, title=None, axes_names=None):
        """Plot clusters in scatter plot 2D or 3D.

        Parameters
        ----------

        """
        nclusters = len(target[:, 0])
        naxes = len(target[0, :])
        # Clustering graphs
        if title is None:
            title = "Clustering"
        if axes_names is None:
            axes_names = ['La1', 'La2', 'La3']
        if codes is not None:
            colors = np.multiply(codes, np.ceil(255 / nclusters))
        else:
            colors = None

        if naxes == 2:
            fig = plt.figure()
            fig.suptitle(title)
            ax = fig.add_subplot(111)
            if colors is None:
                ax.scatter(data[:, 0], data[:, 1], alpha=0.7)
            else:
                ax.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
            ax.scatter(target[:, 0], target[:, 1], alpha=0.5, s=100)
            ax.set_xlabel(axes_names[0])
            ax.set_ylabel(axes_names[1])
            for i in range(nclusters):
                ax.annotate(i + 1, (target[i, 0], target[i, 1]))
            plt.draw()
        if naxes == 3:
            fig = plt.figure()
            fig.suptitle(title)
            ax = fig.gca(projection='3d')
            if colors is None:
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.7)
            else:
                ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                           c=colors, alpha=0.7, s=10)
            ax.scatter(target[:, 0], target[:, 1],
                       target[:, 2], alpha=0.3, s=75)
            ax.set_xlabel(axes_names[0])
            ax.set_ylabel(axes_names[1])
            ax.set_zlabel(axes_names[2])
            for i in range(nclusters):
                ax.text(target[i, 0], target[i, 1], target[i, 2], i + 1)
            plt.draw()


class BeadsReport(object):
    """Per-MRBLE images report.

    This method generates the selected images per-MRBLE.

    WARNING!
    --------
    This method can take a lot of time, since it will generate images
    per-MRBLE. It takes about 5 minutes per 1,000 beads, for 12 images
    each, which makes a total of 11,000 images.

    Parameters
    ----------
    data : Pandas DataFrame
        Contains all the dimension, posotional, and intensity data per-MBRLE.
    images : mrbles.ImageDataFrame, Xarray DataArray
        Contains images.
    masks : Xarray DataArray
        Contains masks.
    assay_channel : str
        Assay channel name, e.g. 'Cy5_FF'
    codes : int, list of int
        Integer or list of integers with selected codes.
        Defaults to None.
    files : int, list of int
            Integer or list of integers with selected files.
            Defaults to None.
    sort : boolean
        Sort by code.
        Defaults to True.

    Methods:
    --------
    generate() : method

    Attributes
    ----------
    time_sec : float
        Time required per-image generated in seconds.
        For instance, 300 beads times 12 images is 3600 images.
        Defaults to 0.0275
    parallelize : boolean
        Wether to use parallelization. Can be slowing down on low-power
        computers.
        Defaults to True.
    """

    def __init__(self, data, images, masks, assay_channel,
                 codes=None, files=None, sort=True):
        # Speed settings
        self.parallelize = True
        self.time_sec = 0.0275

        # Default values
        self.ref_channel = "Eu"
        self.ref_mask = "mask_inside"
        self.bkg_channel = "bkg"
        self.bkg_mask = "mask_inside"
        self.assay_mask = "mask_ring"
        self.npl_channels = ['Eu', 'Dy', 'Tm', 'Sm']
        self.npl_mask = "mask_inside"

        self._dataframe = data.fillna(0)
        self.assay_channel = assay_channel
        self._images = images.combine_first(masks)

        if sort is True:
            self._dataframe.sort_values('code', inplace=True)
        # Set max values for image plots
        self.max_assay = self._images.sel(c=self.assay_channel).max()
        self.max_npl = self._images.sel(c=self.npl_channels).max()
        if codes is not None:
            if isinstance(codes, list):
                self._dataframe = \
                    self._dataframe[self._dataframe['code'].isin(codes)]
            else:
                self._dataframe = \
                    self._dataframe[self._dataframe['code'] == codes]
        if files is not None:
            if isinstance(files, list):
                self._dataframe = \
                    self._dataframe[self._dataframe['f'].isin(files)]
            else:
                self._dataframe = \
                    self._dataframe[self._dataframe['f'] == files]
        self._time_estimate()

    def _time_estimate(self):
        self.beads_num = self._dataframe.shape[0]
        self.images_num = self._images.c.shape[0]
        total_img = self.images_num * self.beads_num
        total_time = (total_img * self.time_sec) / 60  # Time in minutes
        print("Total beads: %i" % self.beads_num)
        print("Total images: %i" % total_img)
        print("Total estimated time required: %i minutes"
              % round(total_time))

    def generate(self, filename):
        """Per-MRBLE images report.

        This method generates the selected images per-MRBLE.

        WARNING!
        --------
        This method can take a lot of time, since it will generate images
        per-MRBLE. It takes about 5 minutes per 1,000 beads, for 12 images
        each, which makes a total of 12,000 images.
        """
        time_0 = time.time()
        self._per_pdf(filename)
        time_1 = time.time()
        time_total = (time_1 - time_0)
        time_per_image = time_total / (self.beads_num * self.images_num)
        time_min = time_total / 60
        time_sec = time_total % 60
        print("Total time: %i minutes %02d seconds" % (time_min, time_sec))
        print("Time per-image: %0.5f" % time_per_image)
        self.time_sec = time_per_image

    def _per_bead_plot(self, idx, image, g_idx, dim, bead_num, img_num):
        ax_sub = plt.subplot2grid(
            (bead_num, img_num), (g_idx, idx)
        )
        if g_idx == 0:
            ax_sub.title.set_text(str(image.c.values))
            ax_sub.title.set_size(2.5)
        if idx == 0:
            ax_sub.set_ylabel("B#:%i \n F#:%i" % (dim['index'], dim['f']), size=3)
            ax_sub.set_xlabel("Code#:%i" % (dim['code']), size=3)
            ax_sub.imshow(image.astype(int))
        elif str(image.c.values) == "mask_check":
            ax_sub.imshow(image.astype(int))
        elif "mask" in str(image.c.values):
            ax_sub.set_xlabel(
                "I: %i" % (dim[self.assay_channel + '.' + str(image.c.values)]),
                size=3
            )
            min_mask = image.values[image.values > 0].min()
            max_mask = image.values[image.values > 0].max()
            img = image.astype(np.uint).values
            ax_sub.imshow(img, vmin=min_mask - 1, vmax=max_mask)
        elif str(image.c.values) == self.assay_channel:
            ax_sub.set_xlabel("I: %i" % (dim[self.assay_channel + '.' + self.assay_mask]), size=3)
            ax_sub.imshow(image.astype(int), vmin=0, vmax=self.max_assay)
        elif str(image.c.values) == self.ref_channel:
            ax_sub.set_xlabel("I: %i" % (dim[self.ref_channel + '.' + self.ref_mask]), size=3)
            ax_sub.imshow(image.astype(int), vmin=0, vmax=self.max_npl)
        elif str(image.c.values) == self.bkg_channel:
            ax_sub.set_xlabel("I: %i" % (dim[self.bkg_channel + '.' + self.bkg_mask]), size=3)
            ax_sub.imshow(image.astype(int), vmin=0, vmax=self.max_npl)
        elif any(channel in str(image.c.values) for channel in self.npl_channels):
            ax_sub.set_xlabel(
                "R: %0.3f" % (dim[str(image.c.values) + '_ratio' + '.' + self.npl_mask]),
                # "R: %0.3f" % (dim[str(image.c.values) + '.' + self.npl_mask]),
                size=3
            )
            ax_sub.imshow(image.astype(int), vmin=0, vmax=self.max_npl)
        ax_sub.set_yticklabels([])
        ax_sub.set_xticklabels([])
        ax_sub.tick_params(axis=u'both', which=u'both', length=0)
        ax_sub.xaxis.labelpad = -3
        ax_sub.patch.set_visible(False)

    def _iter_dims(self, index, dim, figs, bead_num, img_num):
        d_x, d_y, d_r, d_f = dim['x_centroid'], dim['y_centroid'], dim['radius'], dim['f']
        x_min, x_max = round(d_x - 2 * d_r), round(d_x + 2 * d_r)
        y_min, y_max = round(d_y - 2 * d_r), round(d_y + 2 * d_r)
        [self._per_bead_plot(idx, image, index, dim, bead_num, img_num)
         for idx, image in enumerate(
             figs[d_f, :, slice(y_min, y_max), slice(x_min, x_max)])]

    def _per_set_pdf(self, dims_per_step, figs, pdf_object):
        dims_per_step.reset_index(inplace=True, drop=True)
        bead_num = dims_per_step.shape[0]
        img_num = figs.c.shape[0]
        plt_fig = plt.figure(figsize=(0.3 * img_num, 0.3 * bead_num), dpi=300)
        plt.axis('off')
        plt_fig.tight_layout(pad=1, w_pad=0.1, h_pad=2)
        [self._iter_dims(idx, dim, figs, bead_num, img_num)
         for idx, dim in dims_per_step.iterrows()]
        pdf_object.savefig(plt_fig)
        plt.close()

    def _per_pdf(self, filename, dim_step=33):
        with PdfPages(filename) as pdf_object:
            [self._per_set_pdf(self._dataframe[x:(x + dim_step)], self._images, pdf_object)
             if(x + dim_step < self.beads_num)
             else self._per_set_pdf(self._dataframe[x:self.beads_num],
                                    self._images, pdf_object)
             for x in range(0, self.beads_num, dim_step)]


class QCReport(object):
    """MRBLE library Quality Control report.

    Parameters
    ----------
    data : Pandas DataFrame
        Per beads data.
    """

    def __init__(self, data):
        self._per_bead_data = data
        self._savefig = None
        self.dpi = 300

        # Defaults
        self.report_folder = 'report/'

    def generate(self, filename, savefig=False):
        """Generate QC report.

        Parameters
        ----------
        filename : str
            Filename to save QC PDF report to.
        savefig : boolean
            Save figures separately to 'report' folder.
            Defaults to False.

        """
        self._savefig = savefig
        if self._savefig is True:
            if not os.path.exists(self.report_folder):
                os.makedirs(self.report_folder)
        with PdfPages(filename) as pdf_object:
            self._add_figure([self.bead_size],
                             pdf_object)
            if 'code' in self._per_bead_data.columns:
                self._add_figure([self.beads_per_code],
                             pdf_object)
                self.npl_plots(pdf_object)
                self.assay_contamination(pdf_object)

    def _add_figure(self, plots, pdf_object):
        for plot in plots:
            plot()
            pdf_object.savefig(dpi=self.dpi)
            plt.close()

    def bead_size(self):
        """Bead size distribution plot."""
        if 'diameter_conv' in self._per_bead_data.columns:
            d_name = 'diameter_conv'
            dim = 'Converted'
        else:
            d_name = 'diameter'
            dim = 'Pixels'
        b_std = self._per_bead_data[d_name].std()
        b_mean = self._per_bead_data[d_name].mean()
        x_left = b_mean - (5 * b_std)
        x_right = b_mean + (5 * b_std)
        self._per_bead_data[d_name].plot(
            kind='hist', bins=100,
            color='lightgray',
            title="Beads Diameter (%s) Mean: %0.2f SD: %0.2f" % (dim, b_mean, b_std)).set_xlim(left=x_left, right=x_right)
        self._per_bead_data[d_name].plot(kind='kde', secondary_y=True, color='black', alpha=0.7).set_ylim(bottom=0)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'bead_size.png'), dpi=self.dpi)

    def beads_per_code(self):
        """Beads per-code distribution plot."""
        b_std = self._per_bead_data.groupby(['set', 'code']).size().std()
        b_mean = self._per_bead_data.groupby(['set', 'code']).size().mean()
        self._per_bead_data.groupby(['set', 'code']).size().plot(
            kind='hist', color='lightgray', title="Beads-per-code (N) Mean: %0.2f SD: %0.2f" % (b_mean, b_std))
        self._per_bead_data.groupby(['set', 'code']).size().plot(kind='kde', secondary_y=True, color='black', alpha=0.7)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'beads_per_code.png'), dpi=self.dpi)

    def npl_plots(self, pdf_object):
        # Pre ICP
        sns.distplot(self._per_bead_data['Dy_ratio.mask_inside'], hist=True, kde=False, bins=1000)
        plt.title("Dy ratios - pre-ICP")
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'Dy_ratios_pre-ICP.png'), dpi=self.dpi)
        plt.close()

        # Before CI filter
        g = sns.FacetGrid(self._per_bead_data, col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - pre-ICP (No CI filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside', 'Tm_ratio.mask_inside', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        plt.tight_layout()
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'Sm_vs_Tm_ratios_pre-ICP.png'), dpi=self.dpi)
        plt.close()

        # After CI filter
        g = sns.FacetGrid(self._per_bead_data.query('prob > 0.95'), col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - pre-ICP (CI > 0.95 filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside', 'Tm_ratio.mask_inside', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        plt.tight_layout()
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'Sm_vs_Tm_ratios_pre-ICP_CI95_filter.png'), dpi=self.dpi)
        plt.close()

        # Post ICP
        sns.distplot(self._per_bead_data['Dy_ratio.mask_inside_icp'], hist=True, kde=False, bins=1000)
        plt.title("Dy ratios - Post-ICP")
        plt.tight_layout()
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'Dy_ratios_post-ICP.png'), dpi=self.dpi)
        plt.close()

        # Before CI filter
        g = sns.FacetGrid(self._per_bead_data, col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - Post-ICP (No CI filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside_icp', 'Tm_ratio.mask_inside_icp', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        plt.tight_layout()
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'Sm_vs_Tm_ratios_post-ICP.png'), dpi=self.dpi)
        plt.close()

        # After CI filter
        g = sns.FacetGrid(self._per_bead_data.query('prob > 0.95'), col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - Post-ICP (CI > 0.95 filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside_icp', 'Tm_ratio.mask_inside_icp', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        plt.tight_layout()
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'Sm_vs_Tm_ratios_post-ICP_CI95_filter.png'), dpi=self.dpi)
        plt.close()

    def assay_contamination(self, pdf_object):
        self._per_bead_data.plot(kind='scatter', figsize=(9, 6),
                                 x='Cy5_FF.mask_ring_min_bkg',
                                 y='Dy_ratio.mask_inside_icp',
                                 title='Dy Ratio vs Cy5 (ring)')
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'bead_size.png'), dpi=self.dpi)
        plt.close()

        self._per_bead_data.plot(kind='scatter', figsize=(9, 6),
                                 x='Cy5_FF.mask_ring_min_bkg',
                                 y='Sm_ratio.mask_inside_icp',
                                 title='Sm Ratio vs Cy5 (ring)')
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'bead_size.png'), dpi=self.dpi)
        plt.close()

        self._per_bead_data.plot(kind='scatter', figsize=(9, 6),
                                 x='Cy5_FF.mask_ring_min_bkg',
                                 y='Tm_ratio.mask_inside_icp',
                                 title='Tm Ratio vs Cy5 (ring)')
        pdf_object.savefig(dpi=self.dpi)
        if self._savefig is True:
            plt.savefig((self.report_folder + 'bead_size.png'), dpi=self.dpi)
        plt.close()

    def npl_covar_plots(self):
        colors = ['green', 'blue', 'red']
        fig, ax = plt.subplots(dpi=100)
        plt.title('Green: Dy-levels, Blue: DyTm-levels, Red: DySm-levels')
        n = 0
        for X1, y1 in zip(Dy_masks_means, Dy_masks_sds):
            regr = linear_model.LinearRegression()
            X =X1.reshape(-1,1)
            y = y1.reshape(-1,1)
            regr.fit(X, y)
            slope = regr.coef_[0]
            intercept = regr.intercept_
            r2 = sp.stats.pearsonr(X, y)[0]
            sns.regplot(x=X, y=y, ci=None, ax=ax, scatter=True,
                        scatter_kws={'alpha': 0.4, 'color': colors[n]},
                        line_kws={'alpha': 0.6, 'color': colors[n]})
            plt.annotate('Slope: %0.3f Intercept: %0.3f PR2: %0.2f' % (slope, intercept, r2), xy=(X[1], max(y)))
            n+=1
        plt.savefig('Dy-MeanSD.png', dpi=300)

        colors = ['green', 'blue', 'red']
        fig, ax = plt.subplots(dpi=100);
        plt.title('Green: Sm-levels, Blue: SmTm-levels, Red: SmDy-levels')
        n = 0
        for X1, y1 in zip(Sm_masks_means, Sm_masks_sds):
            regr = linear_model.LinearRegression()
            X =X1.reshape(-1,1)
            y = y1.reshape(-1,1)
            regr.fit(X, y)
            slope = regr.coef_[0]
            intercept = regr.intercept_
            r2 = sp.stats.pearsonr(X, y)[0]
            sns.regplot(x=X, y=y, ci=None, ax=ax, scatter=True,
                        scatter_kws={'alpha': 0.4, 'color':colors[n]},
                        line_kws={'alpha': 0.6, 'color':colors[n]})
            plt.annotate('Slope: %0.3f Intercept: %0.3f PR2: %0.2f' % (slope, intercept, r2), xy=(X[1], max(y)))
            n+=1
        plt.savefig('Sm-MeanSD.png', dpi=300)

        colors = ['green', 'blue', 'red']
        fig, ax = plt.subplots(dpi=100)
        plt.title('Green: Tm-levels, Blue: TmSm-levels, Red: TmDy-levels')
        n = 0
        for X1, y1 in zip(Tm_masks_means, Tm_masks_sds):
            regr = linear_model.LinearRegression()
            X =X1.reshape(-1,1)
            y = y1.reshape(-1,1)
            regr.fit(X, y)
            slope = regr.coef_[0]
            intercept = regr.intercept_
            r2 = sp.stats.pearsonr(X, y)[0]
            sns.regplot(x=X, y=y, ci=None, ax=ax, scatter=True,
                        scatter_kws={'alpha':0.4, 'color':colors[n]},
                        line_kws={'alpha':0.6, 'color':colors[n]})
            plt.annotate('Slope: %0.3f Intercept: %0.3f PR2: %0.2f'%(slope, intercept, r2), xy=(X[1], max(y)))
            n+=1
        plt.savefig('Tm-MeanSD.png', dpi=300)

    def code_stability(self):
        #g = sns.FacetGrid(data_per_step_all, col="code", col_wrap=4, sharey=False, xlim=(0,128))
        g = sns.FacetGrid(data_per_step_all.sort_values('code'), col="code", col_wrap=6, sharey=False)

        g.map(sns.regplot, 'sample_size', 'Cy5_min_bkg', data=data_per_step_all,
            logx=True, ci=95, n_boot=5000,
            scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})

    def html_report(self):
        """Generate HTML report."""

        html = """
        <H1 align="center">html2fpdf</H1>
        <h2>Basic usage</h2>
        <p>You can now easily print text while mixing different
        styles : <B>bold</B>, <I>italic</I>, <U>underlined</U>, or
        <B><I><U>all at once</U></I></B>!

        <BR>You can also insert hyperlinks
        like this <A HREF="http://www.mousevspython.com">www.mousevspython.comg</A>,
        or include a hyperlink in an image. Just click on the one below.<br>
        <center>
        <A HREF="http://www.mousevspython.com"><img src="tutorial/logo.png" width="150" height="150"></A>
        </center>

        <h3>Sample List</h3>
        <ul><li>option 1</li>
        <ol><li>option 2</li></ol>
        <li>option 3</li></ul>

        <table border="0" align="center" width="50%">
        <thead><tr><th width="30%">Header 1</th><th width="70%">header 2</th></tr></thead>
        <tbody>
        <tr><td>cell 1</td><td>cell 2</td></tr>
        <tr><td>cell 2</td><td>cell 3</td></tr>
        </tbody>
        </table>
        """

        from pyfpdf import FPDF, HTMLMixin

        class MyFPDF(FPDF, HTMLMixin):
            pass

        pdf=MyFPDF()
        #First page
        pdf.add_page()
        pdf.write_html(html)
        pdf.output('html.pdf','F')


class GenerateCodes(object):
    """Generate bead code set.

    Parameters
    ----------
    colors : list of str
        List of coding colors in a list of strings.

    s0s : liat of float
        List of standard deviations (SD) at intensity 0 for each encoding color.

    slopes : float
        List of slopes of the SDs versus intensity for each encoding color.

    nsigma : float
        The number of SD to separate coding levels.

    Examples
    --------
    >>> code_set_gen = GenerateCodes(['Dy', 'Sm', 'Tm'],
                                     [0.0039, 0.0055, 0.0029],
                                     [0.022, 0.016, 0.049], 6.4)
    >>> code_set_gen.result
               Dy        Sm        Tm
    0    0.000000  0.000000  0.000000
    >>> code_set_gen = GenerateCodes(['Dy', 'Sm'],
                                     [0.0039, 0.0055],
                                     [0.022, 0.016], 8.4)
    >>> code_set_gen.generate()
    Number of codes:  24
    >>> code_set_gen.result
              Dy        Sm
    0   0.000000  0.000000
    1   0.000000  0.106747
    2   0.000000  0.246642
    ......................
    >>> code_set_gen.iterate(28)
    ....................
    Number of codes:  26
    Number of codes:  26
    Number of codes:  28
    Final nsigma:  8.09
    Iterations  :  31
    """

    def __init__(self, colors, s0s, slopes, nsigma):
        if not len(colors) == len(slopes) == len(s0s):
            raise ValueError(
                "Length colors, nsigmas and slopes not equal: %s." % sys.exit()[1])
        self._colors = colors
        self._s0s = s0s
        self._slopes = slopes
        self._nsigma = nsigma
        self._result = None

    def __repr__(self):
        """Return levels."""
        return repr([self.levels])

    @property
    def colors(self):
        """Return color names."""
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = value

    @property
    def axis(self):
        """Return number of axis (colors)."""
        return len(self._colors)

    @property
    def levels(self):
        """Return number of levels."""
        return np.array(self._levels()).T

    def _levels(self, nsigma=None):
        if nsigma is None:
            nsigma = self._nsigma
        levels = []
        for idx, _ in enumerate(self._colors):
            levels.append(self.get_levels(self._s0s[idx],
                                          self._slopes[idx],
                                          nsigma))
        return levels

    @staticmethod
    def recursive_looper(iterators, pos=0):
        """Recursive looper.

        Implements the same functionality as nested for loops, but is more
        dynamic. Iterators can either be a list of methods which return
        iterables, a list of iterables, or a combination of both.
        """
        next_loop, v = None, []
        try:
            gen = iter(iterators[pos]())
        except TypeError:
            gen = iter(iterators[pos])
        while True:
            try:
                yield v + next_loop.next()
            except (StopIteration, AttributeError):
                v = [gen.next(), ]
                if pos < len(iterators) - 1:
                    next_loop = recursive_looper(iterators, pos + 1)
                else:
                    yield v

    @property
    def result(self):
        """Return resulting ratios."""
        if self._result is not None:
            return pd.DataFrame(self._result, columns=self._colors)
        else:
            return None

    def to_csv(self, filename):
        """Export to CSV."""
        self.result.to_csv(filename, sep=',', encoding='utf-8')

    # Experimental
    def to_csv_rep(self, filename, repeats,
                   labels=None, pos=True):
        """Export to CSV, with repeated ratios for bead synthesis."""
        if labels is None:
            labels = ['CeTb', 'Dy', 'Sm', 'Tm']
        if pos is True:
            labels.append('pos')
        data = pd.DataFrame(columns=labels)
        position = 1
        no = 0
        for code in range(self.result.count()[0]):
            for r in range(repeats):
                data.loc[no] = [0,
                                self.result.loc[code, 'Dy'],
                                self.result.loc[code, 'Sm'],
                                self.result.loc[code, 'Tm'],
                                code + 1]
                #data.loc[no] = [0,
                #                self.result.loc[code, 'Dy'],
                #                self.result.loc[code, 'Sm'],
                #                0,
                #                code+1]
                no += 1
        data.to_csv(filename, sep=',', encoding='utf-8')

    def generate(self, nsigma=None, depends=None):
        """Generate codes with default nsigma or given nsigma.

        parameters
        ----------
        nsigma : float, optional
            The number of SD to separate coding levels.
            Defaults to initial nsigma.

        depends : any, experimental, optional
            Used for Tm (3rd in array) dependence on Dy (1st array).

        """
        if nsigma is None:
            nsigma = self._nsigma
        if depends is not None:
            levels = self._levels(nsigma)[:-1]
        else:
            levels = self._levels(nsigma)
        codes = []
        for value in itertools.product(*levels):
            if sum(value) <= 1:
                codes.append(value)

        # Experimental for Tm (must be 3rd in array) depence on Dy (must be 1st in array)
        if depends is not None:
            codes_dep = []
            for code in codes:
                # depends = 0.045
                levels = self.get_levels(
                    self._s0s[2] + depends * code[0], self._slopes[2], nsigma)
                print(levels)
                for level in levels:
                    if (code[0] + code[1] + level) <= 1:
                        codes_dep.append([code[0], code[1], level])
            codes = codes_dep
        # END

        print("Number of codes: ", len(codes))
        self._result = codes

    def iterate(self, num, nsimga_start=None, nsigma_step=0.01, max_iter=1000):
        """Iterate nsigma until number of codes are found.

        Does not work with dependence, such as Tm dependence on Dy.

        parameters
        ----------
        num : int
            Number of required codes.

        nsigma_start : float, optional
            Start value of nsigma.
            Defaults to initial nsigma.

        nsigma_step : float, optional
            Iterarion step.
            Defaults to 0.01

        max_iter : int, optional
            Maximum iteration steps.
            Defaults to 1000.

        """
        if nsimga_start is None:
            nsimga_start = self._nsigma
        len_codes = 0
        step = 0
        while len_codes < num and step < max_iter:
            self.generate(nsimga_start)
            len_codes = len(self._result)
            nsimga_start -= nsigma_step
            step += 1
        print("Final nsigma: ", nsimga_start)
        print("Iterations  : ", step)

    @staticmethod
    def get_levels(std0, slope, nsigma):
        """Predict the number of levels of a coding color.

        The coding levels are based on s0, the standard deviation (SD) at
        intensity 0, and the slope between intensity and SD.

        Parameters
        ----------
        std0 : float
            The SD at intensity 0.

        slope : float
            The slope of the SD versus intensity.

        nsigma : float
            The number of SD to separate levels.

        Returns
        -------
        levels : list of float
            Returns list of codes values for a given coding color.

        """
        nslope = nsigma * slope
        levels = [0]
        while levels[-1] <= 1:
            levels.append((levels[-1] * (1 + nslope) + 2 * nsigma * std0) / (1 - nslope))
        levels.pop()
        return levels


class PeptideScramble(object):
    """Randomizes amino acid sequence

    Parameters
    ----------
    seq : string
        Inset amino acid sequence as a string.

    Returns
    -------
    seq : string
        Returns string of shuffled amino acid sequence.

    """

    def __init__(self, seq):
        self.seq = seq

    def random(self, seq=None):
        """Return randomized amino acid sequence."""
        if seq is None:
            seq = self.seq
        seq_list = list(seq)
        random.shuffle(seq_list)
        return ''.join(seq_list)
