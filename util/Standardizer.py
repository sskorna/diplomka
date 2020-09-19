#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GacrhEstimator.py

Purpose:
    Class to demean and/or rescale variable
    
    standardized variable has mean = 0 and std = 0
    mean and std are stored as dictionary so that when applied on
    pandas dataframe all means and standard deviation are saved for
    later use

Date:
    2020/09/19

@author: SimonSkorna
"""

class Standardize :

    def __init__(self, demean=True, rescale=True, skipna=True) :
        self.demean = demean
        self.rescale = rescale
        self.skipna = skipna
        self.means = {}
        self.stds = {}
    
    def perform_calc(self, data):
        
        data_standard = data
        
        if self.demean == True:
            mean = data.mean(skipna=self.skipna)
            data_standard = data_standard - mean
            
            if data.name not in self.means:
                self.means[data.name] = mean
            
        if self.rescale == True:
            std = data.std(skipna=self.skipna)
            data_standard = data_standard / std
            
            if data.name not in self.stds:
                self.stds[data.name] = std
    
        return data_standard
    
    def get_selfs(self):
        print(self.demean)
    
    def get_means(self):
        return self.means