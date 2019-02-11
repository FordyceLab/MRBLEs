# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Kinetics Classes and Functions.

This file stores the kinetics classes and functions for the MRBLEs Analysis
module.
"""

# [Future imports]
from __future__ import (absolute_import, division, print_function)
from builtins import (range, object)

# [File header]     | Copy and edit for each file in this project!
# title             : kinetics.py
# description       : MRBLEs - Kinetics functions
# author            : Bjorn Harink
# credits           : Kurt Thorn
# date              : 20160511

# [TO-DO]

# [Modules]
# General Python
# import sys
from math import sqrt
# Data Structure
import numpy as np
# Data analysis
import lmfit


# Classes


class KModelSim(object):
    """Kurt's white paper for competive binding.

    Assuming substrate (peptide on bead) or complex (protein) concentration in
    excess.

    Parameters
    ----------
    c_substrate : array
        NumPy array of substrate (peptide) concentrations.

    c_complex : array
        NumPy array of complex (protein) concentrations.

    kd_init : int
        Set range max for Kd's starting from 0.

    tol : float
        Set tolerance error.
        Defaults to 1E-4.

    Attributes
    ----------
    result : array
        Returns NumPy array of fit.

    Functions
    ---------
    fit : function
        Function to fit parameters.

    Examples
        --------
        >>> Mmat = np.logspace(0, 3, 20)  # Matrix of protein concentrations,
            e.g. 20x between 0 to 3 uM.
        >>> Pt = np.array(([10]*10))      # Concentrations of each peptide,
            e.g. 10x 10 uM.
        >>> test_kshow = ba.kin.kshow(Pt, Mmat, 2, 1E-4)
        >>> test_kshow.fit()
        >>> plt.plot(test_kshow.result)
        ...

    """

    def __init__(self, c_substrate, c_complex, kd_init, tol=1E-4):
        self.n_substrate = len(c_substrate)
        self.n_complex = len(c_complex)
        self.c_substrate = c_substrate
        self.c_complex = c_complex
        self.kds = self.kd_init(kd_init, self.n_substrate)
        self.tol = tol

    @property
    def result(self):
        """Return result from fit."""
        return self.MPfinal

    def fit(self):
        """Fit solution and save to results.

        Access result with object.result.
        """
        self.MPfinal = np.zeros((self.n_complex, self.n_substrate))
        MPapproxM = self.comp_excess(
            self.c_complex, self.c_substrate, self.kds)
        MPapproxP = self.sub_excess(self.c_complex, self.c_substrate, self.kds)
        MPnew = np.zeros((self.n_substrate))
        for m in range(self.n_complex):
            Mt = self.c_complex[m]  # total protein concentration
            # initial guess
            if Mt > sum(self.c_substrate):
                MP = MPapproxM[m, :]
            else:
                MP = MPapproxP[m, :]
            err = 1
            while err > self.tol:
                MPsum = sum(MP)
                for p in range(self.n_substrate):
                    MPsump = MPsum - MP[p]
                    b = MPsump - Mt - self.c_substrate[p] - self.kds[p]
                    c = self.c_substrate[p] * (Mt - MPsump)
                    MPnew[p] = (-b - sqrt(b**2 - 4 * c)) / 2
                err = abs(sum((MP - MPnew) / MPnew))
                MP = (0.1 * MPnew + MP) / 1.1
            self.MPfinal[m, :] = MP

    @staticmethod
    def kd_init(kd_init, n_substrate):
        """Initialize Kd values."""
        Kd = np.logspace(
            0, kd_init, n_substrate)  # Kd for each protein-peptide complex
        return Kd

    @staticmethod
    def sub_excess(Mmat, Pt, Kd):
        """Substrate (e.g. peptides on bead) in excess."""
        MPapproxP = np.zeros((len(Mmat), len(Pt)))
        for m in range(len(Mmat)):
            Mt = Mmat[m]  # total protein concentration
            for p in range(len(Pt)):
                MPapproxP[m, p] = (Pt[p] / Kd[p]) * (Mt / (1 + sum(Pt / Kd)))
        return MPapproxP

    @staticmethod
    def comp_excess(Mmat, Pt, Kd):
        """Complex (e.g. added protein concentration) in excess."""
        MPapproxM = np.zeros((len(Mmat), len(Pt)))
        for m in range(len(Mmat)):
            Mt = Mmat[m]  # total protein concentration
            for p in range(len(Pt)):
                MPapproxM[m, p] = Mt * Pt[p] / (Kd[p] + Mt)
        return MPapproxM


class GlobalFit(object):
    """Global non-linear regression.

    This class is based on the lmfit module pipeline.
    For mor information and functionalilty, please visit:
    https://lmfit.github.io/lmfit-py/
    """

    def __init__(self):
        """Initialize values and lmfit."""
        self.fit_params = lmfit.Parameters()
        self.lm_model = None
        self._result = None

    def __repr__(self):
        """Return lmfit Model object."""
        return repr(self.lm_model)

    @property
    def result(self):
        """Return lmfit ModelResult object."""
        return self._result

    @property
    def fit_metrics(self):
        """Return fit metrics."""
        if self._result is None:
            return None
        print('Chi2: ', self._result.chisqr)
        print('Reduced Chi2: ', self._result.redchi)
        print('Degrees of freedom: ', self._result.nfree, '\n')
        return lmfit.report_fit(self._result.params)

    def fit(self, concentrations, fit_all, fit_all_se):
        """Fit data."""
        self._init_params(fit_all)
        self.lm_model = lmfit.Minimizer(self.objective,
                                        self.fit_params,
                                        fcn_args=(concentrations,
                                                  fit_all,
                                                  fit_all_se))
        self._result = self.lm_model.minimize()
        print('Success: ', self._result.success)

    def conf_int(self):
        """Return confidence intervals."""
        result = lmfit.conf_interval(self.lm_model, self._result)
        return result

    def _init_params(self, select_data_median):
        # Setup initial conditions
        Mt_concentrations = np.array([62.5, 125, 250, 500, 1000, 2000])
        Kd_init = 1000
        Kd_initial = np.array([Kd_init] * len(select_data_median))
        # Using a better estimator makes fitting also faster...
        # Kd_initial = np.abs(np.nan_to_num([Kd_init/(np.max(select_data_median[p])/np.max(select_data_median)) for p in range(len(select_data_median))]))

        for p_idx, p_data in enumerate(select_data_median):
            self.fit_params.add('Kd_%i' % (
                p_idx + 1), value=Kd_initial[p_idx], min=0, vary=True)  # Initial values from max
            # fit_params.add( 'Kd_%i' % (p_idx+1), value=KDs[p_idx], min=0, vary=True) # Values initial global fit

            # fit_params.add( 'Rmax_%i' % (p_idx+1), value=np.max(select_data_median), min=0, vary=True)
            self.fit_params.add('Rmax_%i' % (p_idx + 1), value=np.max(
                select_data_median), min=0.5 * np.max(select_data_median), vary=True)
            # fit_params.add( 'Rmax_%i' % (p_idx+1), value=Rmaxs[p_idx], min=0.5*Rmaxs[p_idx], vary=True) # Values initial global fit

        # Pegging parameters
        for iy in range(2, len(select_data_median) + 1):
            self.fit_params['Rmax_%i' % iy].expr = 'Rmax_1'  # Rmax is shared.

    @classmethod
    def model_dataset(cls, params, i, Mt):
        """Model dataset function.

        This extracts the data from the Parameters, used by lmfit.
        """
        Kd = params['Kd_%i' % (i + 1)].value
        Rmax = params['Rmax_%i' % (i + 1)].value
        return cls.model_bind(Mt, Kd, Rmax)

    @classmethod
    def objective(cls, params, Mt, data, sigma=None):
        """Objective function of the Langmuir Isotherm model."""
        # If sigma not used, it used an array with ones, same size as data.
        if sigma is None:
            sigma = np.ones_like(data)
        ndata, nx = data.shape
        resid = np.zeros_like(data)
        # make residual per data set
        for i in range(ndata):
            resid[i, :] = cls.objective(
                data[i, :], cls.model_dataset(params, i, Mt), sigma[i, :])
        # Now flatten this to a 1D array, as minimize() requires.
        return resid.flatten()

    # Protein [Mt] excess
    @staticmethod
    def model_bind(Mt, Kd, Rmax):
        r"""Langmuir Isothem model.

        Assuming protein [M]total in excess model: [M] ≈ [M]total = [MP]+[M].

        Model -> [MP] := ([M]total * Rmax) / (Kd + [M]total)

                [M]total * Rmax
        [MP] := -------------------
                Kd + [M]total

        All concentrations must be in the same units!

        Parameters
        ----------
        Mt : float
            Total protein [M] concentration: [M]total = [M]free+[MP].
        Rmax : float
            Maximum response level in abitrary units.

        """
        model = ((Mt * Rmax) / (Kd + Mt))
        return model
