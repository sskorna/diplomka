#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArchModelWrapper.py

Purpose:
    class object taking arch package estiamting arch models, 
    predicting and evaluating llik

Date:
    2020/10/11

@author: SimonSkorna
"""
###########################################################
### Imports
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm, t
from arch import arch_model

###########################################################
###
class ArchModelWrapper():
    
    def __init__(self, train_data, eval_data_garch):
        
        self.train_data = train_data
        self.eval_data_garch = eval_data_garch
        self.predictions = {}
        
    def estimate_predict(self, model, p, o, q, dist):
        
        if model == 'GAS':
            mod = pf.GAS(ar=p, sc=q, data=self.train_data, family=pf.Normal())
        elif model == 'EGARCH':
            mod = arch_model(
            self.train_data, vol=model, p=p, o=o, q=q,
            dist = dist, mean='Zero')
        elif model == 'GARCH':
            mod = arch_model(
                self.train_data, vol=model, p=p, o=0, q=q,
                dist = dist, mean='Zero')
            
        forecasts = pd.DataFrame()
        for last_date in self.eval_data_garch.index:

            res = mod.fit(last_obs=last_date , disp='off')
            temp = res.forecast(horizon=1)
            temp_var = temp.variance
            temp_mean = temp.mean

            # ugly way to get one day before - 
            # difficult since no data on weekends and holidays(market's closed)
            day_before = self.train_data.iloc[
                np.append(
                    (self.train_data.index == last_date)[
                        1:len(self.train_data.index == last_date)
                    ],
                False)
            ].index[0]

            fcast = {
                'mean_pred' : temp_mean[temp_mean.index == day_before]['h.1'][0],
                'var_pred': temp_var[temp_var.index == day_before]['h.1'][0]
            }
            
            forecasts = forecasts.append(pd.DataFrame(fcast, index = [last_date]))
            
        self.predictions[f'{model}-{dist}-{p}-{q}'] = forecasts
        
        if dist == 'Normal': 

            llik = (
                np.log(
                    norm.pdf(
                        x=self.eval_data_garch['return100'],
                        loc=forecasts['mean_pred'],
                        scale=forecasts['var_pred'])
                ).sum()
            )
        elif dist == 'StudentsT':
            llik = (
                np.log(
                    t.pdf(
                        x=self.eval_data_garch['return100'],
                        df=forecasts.shape[0]-1,
                        loc=forecasts['mean_pred'],
                        scale=forecasts['var_pred'])
                ).sum()
            )

        return llik

    def get_predictions(self):
        return self.predictions