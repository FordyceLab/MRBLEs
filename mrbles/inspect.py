# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspection Classes and Functions
================================

This file stores the inspection classes and functions for the MRBLEs Analysis module.
"""

# [Future imports]
from __future__ import print_function, division

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
# Data
import numpy as np
import pandas as pd
# Imaging
import cv2
# Image display
from matplotlib import pyplot as plt
import plotly.graph_objs as go

# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    from future.standard_library import install_aliases
    from __builtin__ import *  # NOQA
    install_aliases()


### Functions


def cirle_overlay(image, dims=None, ring=None):
    """Cirle Overlay Image.

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


## Classes
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
        return repr([self.levels])

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = value

    @property
    def axis(self):
        return len(self._colors)

    @property
    def nsigma(self):
        return self._nsigma

    @nsigma.setter
    def nsigma(self, value):
        self._nsigma = value

    @property
    def levels(self):
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

    def recursive_looper(iterators, pos=0):
        """ Implements the same functionality as nested for loops, but is
            more dynamic. iterators can either be a list of methods which
            return iterables, a list of iterables, or a combination of both.
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
        if self._result is not None:
            return pd.DataFrame(self._result, columns=self._colors)
        else:
            return None

    def to_csv(self, filename):
        self.result.to_csv(filename, sep=',', encoding='utf-8')

    # Experimental
    def to_csv_rep(self, filename, repeats, labels=['CeTb', 'Dy', 'Sm', 'Tm'], pos=True):
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
            Used for Tm (must de 3rd in array) dependence on Dy (must 1st in array).
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
        if seq is None:
            seq = self.seq
        seq_list = list(seq)
        random.shuffle(seq_list)
        return ''.join(seq_list)


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