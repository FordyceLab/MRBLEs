# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : kinetics.py
# description       : Bead Kinetics module
# author            : Bjorn Harink
# credits           : 
# date              : 20160511
# version update    : 20160511
# version           : v0.1
# usage             : As module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General
import numpy as np
from math import sqrt
# Project
import bead_analysis as ba

class kfit(object):
    """Kurt's white paper fit algorithm
    """
    def __init__(self, c_substrate, c_complex, kd_init):
        self.n_substrate = len(c_substrate)
        self.n_complex = len(c_complex)
        self.c_substrate = c_substrate
        self.c_complex = c_complex
        self.kds = self.kd_init(kd_init, self.n_substrate)
        
    @property
    def result(self):
        """Return result from fit.
        """
        return self.MPfinal
        
    def fit(self):
        """Fit solution and save to results. Access result with object.result
        """
        self.MPfinal = np.zeros((self.n_complex, self.n_substrate))
        MPnew = np.zeros((self.n_substrate))
        for m in xrange(len(Mmat)):
            Mt = Mmat[m]  # total protein concentration
            # initial guess
            if Mt > sum(Pt):
                MP = MPapproxM[m,:]
            else:
                MP = MPapproxP[m,:]
            err = 1
            while err > 0.0001:
                MPsum = sum(MP)
                for p in xrange(self.n_substrate):
                    MPsump = MPsum - MP[p]
                    b = MPsump - Mt - Pt[p] - Kd[p]
                    c = Pt[p] * (Mt - MPsump)
                    MPnew[p] = (-b - sqrt(b**2 - 4*c)) / 2
                err = abs(sum((MP - MPnew)/MPnew))
                MP = (0.1*MPnew + MP) / 1.1
            self.MPfinal[m,:] = MP
    
    @staticmethod
    def kd_init(kd_init, n_substrate):
        """Initial Kd values
        """
        Kd = np.logspace(0, kd_init, n_substrate)  # Kd for each protein-peptide complex
        return Kd
    
    @staticmethod
    def sub_excess(Mt, Pt, Kd):
        """Substrate (e.g. peptides on bead) in excess
        """
        MPapproxP = np.zeros((len(Mmat), len(Pt)))
        for m in xrange(len(Mmat)):
            Mt = Mmat[m]  # total protein concentration
            for p in xrange(len(Pt)):
                MPapproxP[m,p] = (Pt[p]/Kd[p]) * (Mt / (1 + sum(Pt/Kd)))
        return MPapproxP
    
    @staticmethod
    def comp_excess(Mt, Pt, Kd):
        """Complex (e.g. added protein concentration) in excess
        """
        MPapproxM = np.zeros((len(Mmat), len(Pt)))
        for m in xrange(len(Mmat)):
            Mt = Mmat[m]  # total protein concentration
            for p in xrange(len(Pt)):
                MPapproxM[m,p] = Mt * Pt[p] / (Kd[p] + Mt)
        return MPapproxM