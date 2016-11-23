# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : inpect.py
# description       : Bead Kinetics module - Inspect
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20161114
# version update    : 20161114
# version           : v0.4
# usage             : As module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [Modules]
# General Python
import sys
import os
import types
import warnings
import itertools
# Data
import numpy as np
import pandas as pd
# Image display
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import axes3d

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
            raise ValueError("Length colors, nsigmas en slopes not equal: %s." % sys.exit()[1])
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

    def recursiveLooper(iterators, pos=0):
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
                v = [gen.next(),]
                
                if pos < len(iterators) - 1:
                    nextLoop = recursiveLooper(iterators, pos + 1)
                else:
                    yield v

    def depends(self):
        pass

    @property
    def result(self):
        if self._result is not None:
            return pd.DataFrame(self._result, columns=self._colors)
        else:
            return None

    def to_csv(self, filename):
        self.result.to_csv(filename, sep=',', encoding='utf-8')


    # Experimental
    def to_csv_rep(self, filename, repeats):
        data = pd.DataFrame(columns = ['CeTb', 'Dy', 'Sm', 'Tm', 'pos'])
        position = 1
        no = 0
        for code in xrange(self.result.count()[0]):
            for r in xrange(repeats):
                data.loc[no] = [0, self.result.loc[code, 'Dy'], self.result.loc[code, 'Sm'], self.result.loc[code, 'Tm'], code+1]
                no += 1
        data.to_csv(filename, sep=',', encoding='utf-8')


        result.to_csv(filename, sep=',', encoding='utf-8')

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

        # Experimental for Tm (3rd in array) depence on Dy (1st in array)
        if depends is not None:
            codes_dep = []
            for code in codes:
                levels = self.get_levels(self._s0s[2]+0.045*code[0], self._slopes[2], nsigma)
                print(levels)
                for level in levels:
                    if (code[0]+code[1]+level) <= 1:
                        codes_dep.append([code[0],code[1],level])
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
    def get_levels(s0, slope, nsigma):
        """Predict the number of levels of a coding color. 
        
        The coding levens are based on s0, the standard deviation (SD) at intensity 0, and the slope between intensity and SD.

        Parameters
        ----------
        s0 : float
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
        nc = nsigma * slope
        levels = [0]
        while levels[-1] <= 1:
            levels.append( ( levels[-1]*(1+nc) + 2*nsigma*s0 ) / (1-nc) )
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
            colors = np.multiply(codes, np.ceil(255/nclusters))
        else:
            colors = None

        if naxes == 2:
            fig = plt.figure()
            fig.suptitle(title)
            ax = fig.add_subplot(111)
            ax.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
            ax.scatter(target[:, 0], target[:, 1], alpha=0.5, s=100)
            ax.set_xlabel(axes_names[0])
            ax.set_ylabel(axes_names[1])
            plt.draw()
        if naxes == 3:
            fig = plt.figure()
            fig.suptitle(title)
            ax = fig.gca(projection='3d')
            ax.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
            ax.scatter(target[:, 0], target[:, 1], alpha=0.5, s=100)
            ax.set_xlabel(axes_names[0])
            ax.set_ylabel(axes_names[1])
            ax.set_zlabel(axes_names[2])
            plt.draw()