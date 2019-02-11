# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""MRBLE-Path Classes and Functions.

This file stores the MRBLE-Path classes and functions for the MRBLEs Analysis
module.
"""

# [Future imports]
from __future__ import (absolute_import, division, print_function)

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
    blast : bool
        Setting to convert blast E-scores.
        Defaults to True.

    """

    def __init__(self, references, blast=True):
        super(PathUnmix, self).__init__()
        if blast is True:
            self.references = self.blast_convert(references)
        else:
            self.references = references

    def unmix(self, data, signal, z_score=True):
        """Unmix data.

        Parameters
        ----------
        data : Pandas DataFrame
            Data that contains the various sets.
        signal : str
            Column with signal data.
        z_score : bool or list
            Convert to Z-score if set to True, or uses mean and SD values of
            provided list position 0 is mean, position 1 is SD.
            Defaults to True.

        """
        data_conv = pd.DataFrame(
            {'signal': data.groupby(["set", "code"])[signal].median()}
        ).reset_index()
        sets = self.get_set_names(data_conv)
        data_sets = {s_name: self._unmix(data_conv[data_conv.set == s_name],
                                         z_score)
                     for s_name in sets}
        dataframe = pd.DataFrame.from_dict(data_sets,
                                           orient='index',
                                           columns=self.references.columns)
        self._dataframe = dataframe

    def _unmix(self, data, z_score=True):
        data = data.groupby('code')['signal'].median()
        if z_score is True:
            data = zscore(data)
        elif isinstance(z_score, list):
            data = (data - z_score[0]) / z_score[1]
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
    def generate_test_refs(channels, spike_channel=None, signal_max=2**16,
                           scale=True):
        """Generate test reference spectra.

        spike_channel : list
            List of channel numbers to spike.
        signal_max : int
            Maximum value.
            Defaults to 2**16: 65536.
        scale : bool
            Scale to 1.
            Defaults to True.
        """
        data = np.zeros((channels))
        for channel in range(channels):
            data[channel] = randrange(0, signal_max)
        if spike_channel is not None:
            for sp_ch in spike_channel:
                data[sp_ch] = data[sp_ch] * randrange(1, 10)
        if scale is True:
            data /= data.sum()
        return pd.DataFrame(data)
