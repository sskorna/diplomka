#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_load.py

Purpose:
    Download prices of specified stock Yahoo Finance
    and claculated returns as ratio of price and lagged price by one time interval

Date:
    2020/07/30

@author: SimonSkorna
"""
###########################################################
### Imports
import pandas as pd
import datetime
import yfinance as yf

###########################################################
### returns loading function
def returns_load(ticker, start_dt, end_dt, interval = '1d'):
    data = yf.download(
        tickers = ticker,
        interval = interval,
        start=start_dt,
        end=end_dt
        )
    close_col_index = list(data.columns).index('Close')
    data_len = data.shape[0]
    ret_data = pd.DataFrame({
        'dt': data.index[1:data_len],
        'close': data.iloc[1:data_len, close_col_index].values,
        'close_lag': data.iloc[0:data_len-1, close_col_index].values,
        'return': (data.iloc[1:data_len, close_col_index].values / 
        data.iloc[0:data_len-1, close_col_index].values) - 1 ,
        'return100': ((data.iloc[1:data_len, close_col_index].values / 
        data.iloc[0:data_len-1, close_col_index].values) - 1) * 100
        })
    return ret_data