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
        if len(dataset.shape) == 1:
            train_set = dataset.iloc[:split, ].to_frame()
            test_set = dataset.iloc[split:, ].to_frame()
        else:
            train_set = dataset.iloc[:split, :]
            test_set = dataset.iloc[split:, :]
        # split date
        self.last_split_point = train_set.iloc[split-1].name
        self.first_split_point = test_set.iloc[0].name
        print(f'''Last observation in trainset: {self.last_split_point} , 
            first observation in testset: {self.first_split_point}''')
        
        return [train_set, test_set]
    def get_last_split(self):
        return self.last_split_point
        
    def get_first_split(self):
        return self.first_split_point