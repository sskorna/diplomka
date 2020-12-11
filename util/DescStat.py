#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DescStat.py

Purpose:
    Object containing stock return data, calculating descriptive statistics 
    over specified window

Date:
    2020/08/27

@author: SimonSkorna
"""
###########################################################
### Imports
import pandas as pd
import numpy as np
###########################################################
### DescStat object

class DescStat: 
    def __init__(self, data): 
        
        self.data = data
        
    def calc_mean(self, window, fill_initial=None):
        
        mean_ = self.data.rolling(window).mean()
        # drop last and include one more Nan at begining to shift according 
        # to explained Variable
        # in order to not have target leak (not to see future in explanatory variables)
        mean_ = pd.concat([pd.Series(np.nan), mean_[:-1]]).reset_index(drop=True)
        if fill_initial is not None:
            
            mean_ = self.fill_initial_nan(
                data=mean_, 
                window=window,
                fill_initial=fill_initial
            )
        
        return mean_
        
    def calc_var(self, window, fill_initial=None): 
        
        var_ = self.data.rolling(window).var()
        # drop last and include one more Nan at begining to shift according 
        # to explained Variable
        # in order to not have target leak (not to see future in explanatory variables)
        var_ = pd.concat([pd.Series(np.nan), var_[:-1]]).reset_index(drop=True)
        if fill_initial is not None:
            var_ = self.fill_initial_nan(
                data=var_, 
                window=window,
                fill_initial=fill_initial
            )
        
        return var_
    
    def fill_initial_nan(self, data, window, fill_initial):
        
        if fill_initial == 'first_constant' : 
            data_filled = data
            data_filled[0:window] = data[window]
            
        elif fill_initial == 'zero' :
            data_filled = data
            data_filled[0:window] = 0
            
        return data_filled