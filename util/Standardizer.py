#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standardizer.py

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

class Standardizer :

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
            # store mean indictionary
            if data.name not in self.means:
                self.means[data.name] = mean
            
        if self.rescale == True:
            std = data.std(skipna=self.skipna)
            data_standard = data_standard / std
            # store std in dictionary
            if data.name not in self.stds:
                self.stds[data.name] = std
    
        return data_standard
    
    def get_stds(self):
        return self.stds
    
    def get_means(self):
        return self.means
    
    def predict(self, data) : 
        # standardize new data if stored means and stds
        data_standard = data
        
        if self.demean == True:
            mean = self.means[data.name]
            data_standard = data_standard - mean
            
        if self.rescale == True:
            std = self.stds[data.name]
            data_standard = data_standard / std
    
        return data_standard