# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRBLE-Path Classes and Functions
================================

This file stores the MRBLE-Path classes and functions for the MRBLEs Analysis
module.
"""

# [Future imports]
from __future__ import (absolute_import, division, print_function)
from builtins import (object)

# [File header]     | Copy and edit for each file in this project!
# title             : path.py
# description       : MRBLEs - MRBLE-Path functions
# author            : Bjorn Harink
# credits           : Adam White, Huy Nguyen
# date              : 20180628

# [TO-DO]

# [Modules]
# General Python
from random import randrange
# Data Structure
import numpy as np
from scipy.stats.mstats import zscore
import pandas as pd
# Project
from mrbles.data import TableDataFrame


# Classes


class PathUnmix(TableDataFrame):
    """MRBLE-Path unmixing algorithm.

    Parameters
    ----------
    references : Pandas DataFrame
        Dataframe with reference spectra.

    """

    def __init__(self, references, blast=True):
        super(PathUnmix, self).__init__()
        if blast is True:
            self.references = self.blast_convert(references)
        else:
            self.references = references

    def unmix(self, data, signal, z_score=True):
        """Unmix data."""
        data_conv = pd.DataFrame(
                {'signal': data.groupby(["set", "code"])[signal].median()}
            ).reset_index()
        sets = self.get_set_names(data_conv)
        data_sets = {s_name: self._unmix(data_conv[data_conv.set == s_name], z_score)
                     for s_name in sets}
        dataframe = pd.DataFrame.from_dict(data_sets, orient='index', columns=self.references.columns)
        return dataframe

    def _unmix(self, data, z_score=True):
        if z_score is True:
            data = zscore(data.groupby('code')['signal'].median())
        unmixed = np.linalg.lstsq(self.references, data, rcond=None)[0]
        return unmixed

    @staticmethod
    def blast_convert(data):
        """Convert and invert BLAST E-values to 0-1 reference spectra."""
        refs_log = np.log10(data) * -1
        refs_log[refs_log < 0] = 0
        refs_log /= refs_log.sum()
        return refs_log

    @staticmethod
    def generate_test_refs(channels, spike_channel=None, signal_max=2 ** 13,
                           scale=True):
        """Generate test reference spectra."""
        data = np.zeros((channels))
        for ch in range(channels):
            data[ch] = randrange(0, signal_max)
        if spike_channel is not None:
            for sp_ch in spike_channel:
                data[sp_ch] = data[sp_ch] * randrange(1, 10)
        if scale is True:
            data /= data.sum()
        return pd.DataFrame(data)
