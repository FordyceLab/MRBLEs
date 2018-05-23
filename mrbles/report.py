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
import itertools
import random
import time
# import multiprocessing as mp
# Data
import numpy as np
import pandas as pd
# Imaging
import cv2
# Image display
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objs as go
import seaborn as sns


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

def cluster3d_check(bead_set, target, gmix, set_prob=1, channels=None):
    """Clustering plot
    """
    if channels is None:
        channels = ['rat_dy_icp', 'rat_tm_icp', 'rat_sm_icp']
    clusters = len(target)
    colors = np.multiply(bead_set.code.loc[((bead_set.code >= 0) & (
        bead_set.log_prob > set_prob))].values, np.ceil(255 / clusters))

    bead_ratios_all = go.Scatter3d(
        name='Bead ratios - Marked',
        x=bead_set.loc[((bead_set.code.isnull()) | (
            bead_set.log_prob <= set_prob)), (channels[0])].values,
        y=bead_set.loc[((bead_set.code.isnull()) | (
            bead_set.log_prob <= set_prob)), (channels[1])].values,
        z=bead_set.loc[((bead_set.code.isnull()) | (
            bead_set.log_prob <= set_prob)), (channels[2])].values,
        text=bead_set.loc[:, ('lbl')].values,
        mode='markers',
        marker=dict(
            size=3,
            colorscale='grey',
            opacity=0.7,
            symbol='cross'
        )
    )

    bead_ratios = go.Scatter3d(
        name='Bead ratios - Filtered',
        x=bead_set.loc[((bead_set.code >= 0) & (
            bead_set.log_prob > set_prob)), ('rat_dy_icp')].values,
        y=bead_set.loc[((bead_set.code >= 0) & (
            bead_set.log_prob > set_prob)), ('rat_sm_icp')].values,
        z=bead_set.loc[((bead_set.code >= 0) & (
            bead_set.log_prob > set_prob)), ('rat_tm_icp')].values,
        text=bead_set.loc[(bead_set.code >= 0), ('lbl')].values,
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
        text=list(range(1, clusters + 1)),
        mode='markers',
        marker=dict(
            size=4,
            color='black',
            opacity=0.5,
            symbol="diamond"
        )
    )

    mean_ratios = go.Scatter3d(
        name='GMM mean ratios',
        x=gmix.means[:, 0],
        y=gmix.means[:, 1],
        z=gmix.means[:, 2],
        text=list(range(1, clusters + 1)),
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            opacity=0.5,
            symbol="diamond"
        )
    )

    data = [bead_ratios_all, bead_ratios, target_ratios, mean_ratios]
    layout = go.Layout(
        showlegend=True,
        scene=dict(
            xaxis=dict(
                title='Dy/Eu'
            ),
            yaxis=dict(
                title='Sm/Eu'
            ),
            zaxis=dict(
                title='Tm/Eu'
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
    return fig


# Classes
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
    >>> code_set_gen = GenerateCodes(['Dy', 'Sm', 'Tm'], [0.0039, 0.0055, 0.0029], [0.022, 0.016, 0.049], 6.4)
    >>> code_set_gen.result
               Dy        Sm        Tm
    0    0.000000  0.000000  0.000000
    >>> code_set_gen = GenerateCodes(['Dy', 'Sm'], [0.0039, 0.0055], [0.022, 0.016], 8.4)
    >>> code_set_gen.generate()
    Number of codes:  24
    >>> code_set_gen.result
              Dy        Sm
    0   0.000000  0.000000
    1   0.000000  0.106747
    2   0.000000  0.246642
    ......................
    >>> code_set_gen.iterate(27)
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
                "Length colors, nsigmas en slopes not equal: %s." % sys.exit()[1])
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
        for idx, value in enumerate(self._colors):
            levels.append(self.get_levels(self._s0s[idx],
                                          self._slopes[idx],
                                          nsigma))
        return levels

    @staticmethod
    def recursive_looper(iterators, pos=0):
        """Recursive looper.

        Implements the same functionality as nested for loops, but is more
        dynamic. iterators can either be a list of methods which return
        iterables, a list of iterables, or a combination of both.
        """
        nextLoop, v = None, []
        try:
            gen = iter(iterators[pos]())
        except TypeError:
            gen = iter(iterators[pos])
        while True:
            try:
                yield v + nextLoop.next()
            except (StopIteration, AttributeError):
                v = [gen.next(), ]
                if pos < len(iterators) - 1:
                    nextLoop = recursive_looper(iterators, pos + 1)
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
                   labels=['CeTb', 'Dy', 'Sm', 'Tm'], pos=True):
        """Export to CSV, with repeated ratios for bead synthesis."""
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


class Cluster(object):

    def __init__(self, *args, **kwargs):
        return super(Cluster, self).__init__(*args, **kwargs)

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
    sets : string, list of string
        String or list of strings with set names to be combined.
        Defaults to None.
    codes : int, list of int
        Integer o rlist of integers with selected codes.
        Defaults to None.

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

    def __init__(self, data, images, assay_channel,
                 sets=None, codes=None, sort=True):
        """Init."""
        self._dataframe = data.fillna(0)
        if sort is True:
            self._dataframe.sort_values('code', inplace=True)
        if sets is not None:
            if isinstance(sets, list):
                self._dataframe = \
                    self._dataframe[self._dataframe['set'].isin(sets)]
            else:
                self._dataframe = \
                    self._dataframe[self._dataframe['set'] == sets]
        if codes is not None:
            if isinstance(codes, list):
                self._dataframe = \
                    self._dataframe[self._dataframe['code'].isin(codes)]
            else:
                self._dataframe = \
                    self._dataframe[self._dataframe['code'] == codes]
        self._images = images

        # Speed settings
        self.parallelize = True
        self.time_sec = 0.0275

        # Default values
        self.ref_channel = "Eu"
        self.ref_mask = "mask_inside"
        self.bkg_channel = "bkg"
        self.bkg_mask = "mask_inside"
        self.assay_channel = assay_channel
        self.assay_mask = "mask_ring"
        self.npl_channels = ['Eu', 'Dy', 'Tm', 'Sm']
        self.npl_mask = "mask_inside"

        self.max_assay = self._images.sel(c=self.assay_channel).max()
        self.max_npl = self._images.sel(c=self.npl_channels).max()

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
        self._per_pdf(filename, sets, codes)
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
    """MRBLE library Quality Control report."""

    def __init__(self, data):
        self._per_bead_data = data
        self.dpi = 300

        self.npl_channels = ['Eu', 'Dy', 'Tm', 'Sm']

    def generate(self, filename):
        with PdfPages(filename) as pdf_object:
            self._add_figure([self.bead_size,
                              self.beads_per_code],
                             pdf_object)
            self.npl_plots(pdf_object)

    def _add_figure(self, plots, pdf_object):
        for plot in plots:
            plot()
            pdf_object.savefig(dpi=self.dpi)
            plt.close()

    def bead_size(self):
        """Bead size distribution plot."""
        self._per_bead_data.loc[:, 'diameter'] = self._per_bead_data.radius * 2
        b_std = self._per_bead_data.diameter.std()
        b_mean = self._per_bead_data.diameter.mean()
        x_left = b_mean - (5 * b_std)
        x_right = b_mean + (5 * b_std)
        self._per_bead_data.diameter.plot(
            kind='hist', bins=100, color='lightgray', title="Beads Diameter (Pixels) Mean: %0.2f SD: %0.2f" % (b_mean, b_std)).set_xlim(left=x_left, right=x_right)
        self._per_bead_data.diameter.plot(kind='kde', secondary_y=True, color='black', alpha=0.7).set_ylim(bottom=0)

    def beads_per_code(self):
        """Beads per-code distribution plot."""
        b_std = self._per_bead_data.groupby(['set', 'code']).size().std()
        b_mean = self._per_bead_data.groupby(['set', 'code']).size().mean()
        self._per_bead_data.groupby(['set', 'code']).size().plot(
            kind='hist', color='lightgray', title="Beads-per-code (N) Mean: %0.2f SD: %0.2f" % (b_mean, b_std))
        self._per_bead_data.groupby(['set', 'code']).size().plot(kind='kde', secondary_y=True, color='black', alpha=0.7)

    def npl_plots(self, pdf_object):
        # Pre ICP
        sns.distplot(self._per_bead_data['Dy_ratio.mask_inside'], hist=True, kde=False, bins=1000)
        plt.title("Dy ratios - pre-ICP")
        pdf_object.savefig(dpi=self.dpi)
        plt.close()

        # Before CI filter
        g = sns.FacetGrid(self._per_bead_data, col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - pre-ICP (No CI filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside', 'Tm_ratio.mask_inside', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        pdf_object.savefig(dpi=self.dpi)
        plt.close()

        # After CI filter
        g = sns.FacetGrid(self._per_bead_data.query('confidence > 0.95'), col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - pre-ICP (CI > 0.95 filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside', 'Tm_ratio.mask_inside', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        pdf_object.savefig(dpi=self.dpi)
        plt.close()

        # Post ICP
        sns.distplot(self._per_bead_data['Dy_ratio.mask_inside_icp'], hist=True, kde=False, bins=1000)
        plt.title("Dy ratios - Post-ICP")
        pdf_object.savefig(dpi=self.dpi)
        plt.close()

        # Before CI filter
        g = sns.FacetGrid(self._per_bead_data, col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - Post-ICP (No CI filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside_icp', 'Tm_ratio.mask_inside_icp', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        pdf_object.savefig(dpi=self.dpi)
        plt.close()

        # After CI filter
        g = sns.FacetGrid(self._per_bead_data.query('confidence > 0.95'), col="info.Dy", col_wrap=3, sharey=True)
        g.fig.suptitle("Sm vs Tm ratios - Post-ICP (CI > 0.95 filter)")
        g.fig.subplots_adjust(top=10)
        g.map(sns.regplot, 'Sm_ratio.mask_inside_icp', 'Tm_ratio.mask_inside_icp', fit_reg=False,
              scatter=True, scatter_kws={'alpha': 0.3, 'color': 'darkgray'}, line_kws={'color': 'black'})
        pdf_object.savefig(dpi=self.dpi)
        plt.close()

    def assay_contamination(self):
        npl_num = len(self.npl_channels)
        plt.figure(dpi=150)
        self._per_bead_data.plot(kind='scatter', figsize=(9, 6), x='Cy5_min_bkg', y='Dy_ratio.mask_inside_icp',
                         title='Dy Ratio vs Cy5 (ring)')
        #plt.savefig('Dy Ratio vs Cy5 (ring) - BKG.png', dpi=300)

        plt.figure(dpi=150)
        self._per_bead_data.plot(kind='scatter', figsize=(9, 6), x='Cy5_min_bkg', y='Sm_ratio.mask_inside_icp',
                         title='Sm Ratio vs Cy5 (ring)')
        #plt.savefig('Sm Ratio vs Cy5 (ring) - BKG.png',dpi=300)

        plt.figure(dpi=300)
        self._per_bead_data.plot(kind='scatter', figsize=(9, 6), x='Cy5_min_bkg', y='Tm_ratio.mask_inside_icp',
                         title='Tm Ratio vs Cy5 (ring)')
        #plt.savefig('Tm Ratio vs Cy5 (ring) - BKG.png', dpi=300)

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
