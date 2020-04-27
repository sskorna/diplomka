#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DescStat.py

Purpose:
    Object containing stock return data, calculating descriptive statistics 
    over specified window

Date:
    2020/04/27

@author: SimonSkorna
"""
###########################################################
### Imports
import pandas as pd


###########################################################
### DescStat object

class DescStat: 
    def __init__(self, data): 
        self.data = data
        
    def calc_mean(self, window):
        return self.data.rolling(window).mean()
        
    def calc_var(self, window): 
        return self.data.rolling(window).mean()