#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainTestSplitter.py

Purpose:
    Split dataset into Train/Test parts based on oercentage of Train test parts
    Outputing: Train, Test, datatime point of split

Date:
    2020/09/20

@author: SimonSkorna
"""
###########################################################
### Imports
import pandas as pd

###########################################################
### TrainTestSplitter

class Splitter:

    def __init__(self, train_part):
    
        self.train_part = train_part
    
    def split(self, dataset):
        # sort date indes, just in case it is not
        dataset = dataset.sort_index()
        # place of split
        nrow = dataset.shape[0]
        split = int(round(nrow*self.train_part, 0))
        # dataset split
        train_set = dataset.iloc[:split, :]
        test_set = dataset.iloc[split:, :]
        # split date
        self.split_point = train_set.iloc[split-1].name
        
        return [train_set, test_set]