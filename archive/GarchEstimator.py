#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GacrhEstimator.py

Purpose:
    class object of estimating and fitting various GARCH models

Date:
    2020/04/26

@author: SimonSkorna
"""
###########################################################
### Imports
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
###########################################################
###
class GarchEstimator:
    
    def __init__(self, theta_init, theta=None):
        self.theta_init = theta_init
        if (theta is None) : 
            self.theta = theta_init
        else :
            self.theta = theta
        
    def garch_filter(self, time_series, theta):
        
        T = len(time_series)
        sigma_sqr = np.zeros(T)
        # initiate first value
        # sigma_sqr[0:2] = np.var(time_series)
        sigma_sqr[0:3] = np.var(time_series)
        # loop through all values of sigma
        for i in range(3, T):
            sigma_sqr[i] = theta[1] + theta[2] * time_series[i-1] ** 2 + theta[3] * time_series[i-2] ** 2 + theta[4] * sigma_sqr[i-1] + theta[5] * sigma_sqr[i-2] + theta[6] * sigma_sqr[i-3]
        
        return sigma_sqr
        
    def calc_llik(self, theta, time_series, method, optim = False):
    
        if (method == 'GARCH') :
            sigma_sqr = self.garch_filter(time_series=time_series, theta=theta)
            sigma = np.sqrt(sigma_sqr)
        
        llik = -(1/2) * np.log(2 * np.pi) - (1/2) * np.log(sigma**2) - (1/2) * (((time_series - theta[0]) ** 2) / sigma ** 2)
        
        if optim == True :
            return -sum(llik)
        else: 
            return sum(llik)
        
    def fit_model(self, time_series, method):
        bounds = Bounds([-10, 0.000001, 0, 0, 0, 0, 0],[10, 100, 100, 100, 100, 100, 100])
        x0 = list(self.theta_init.values())
        optim = True

        res = minimize(
            fun = self.calc_llik,  
            x0 = x0, 
            args = (time_series, method, optim),
            method = 'trust-constr',
            options = {'gtol' : 1e-2, 'xtol' : 1e-2, 'maxiter' : 100, 'disp' : True},
            bounds = bounds
        )
        self.theta = {
            'mu' : res.x[0],
            'alpha' : res.x[1],
            'beta1' : res.x[2],
            'beta2' : res.x[3],
            'omega1' : res.x[4],
            'omega2' : res.x[5],
            'omega3' : res.x[6],
        }
        print('Model recalculated!')
        print(res)
        
    def get_estimates(self, coef=True, llik=False):
        
        if (coef == True) & (llik == True) : 
            return self.theta, self.llik
            
        elif(coef == True) & (llik == False) :
            return self.theta
            
        elif(coef == False) & (llik == True) :
            return self.llik
            
    def fit_data(self, method, data_estimate, data_fit):
        # if all coefficients are equal to initialization, than model has not been fitted yet
        if all([self.theta[k] == self.theta_init[k] for k in self.theta]):
            self.fit_model(time_series=data_fit, method=method)
            
        if method == 'GARCH' : 
        
            sigma_sqr = self.garch_filter(time_series=data_fit, theta=list(self.theta.values()))
            sigma = np.sqrt(sigma_sqr)
            
            self.llik = self.calc_llik(theta=list(self.theta.values()), time_series=data_fit, method=method)
            
        return sigma_sqr
        
    def get_AIC(self): 
        
        try: 
            self.llik
        
        except AttributeError as error:
            raise AttributeError('Log-likelihood missing (model needs to be fitted)')
        
        return -2 * self.llik + 2*self.theta.length