#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 12:38:36 2022

@author: adnene33
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import signal
import re

def colname2attribs(colname: str) -> list:
    # extract metadata
    colname = re.sub(' ', '', colname)
    colname = re.sub('[(),_]', ' ', colname)
    colname = re.sub('  ', ' ', colname)
    subst, c, date, power, it, run = colname.split(' ')
    # cast string attributes to float
    c = float(c)
    power = float(power[:-2]) 
    it = float(it[:-1])
    attribs = [subst, c, date, power, it, run]
    return attribs

def extract_info(colnames: list) -> pd.DataFrame:
    metadata = list(map(colname2attribs, colnames))
    df_info = pd.DataFrame(data=metadata, columns=['subst', 'c', 'date', 'power', 'it', 'run'])
    return df_info

def wavelength2ramanshift(df: pd.DataFrame, x='x', ref: float = 522):
    assert x in df.columns
    wavelength = df.filter(regex=x).values.ravel()
    assert wavelength.size == df.shape[0]
    df[x] = 1e7*(1/ref - 1/wavelength)
    return df

def trim_data(df: pd.DataFrame, x='x', low: float = np.inf, high:float = np.inf):
    if low == np.inf and high == np.inf:
        return df
    return df[(df[x] > low) & (df[x] < high)]

def read_data(fname, lb=-np.inf, ub=np.inf):
    df = pd.read_csv(fname, delimiter=",")
    df = df[(df.x >= lb) & (df.x < ub)]
    return df

def smooth_data(df, window_length=21, polyorder=5):
    """
    Apply a Savitzky-Golay filter spectrograms contained in DataFrame.
    Column 'x' remains unchanged"""
    x = df.x
    smoothed = signal.savgol_filter(df, window_length=window_length, polyorder=polyorder, axis=0)
    df_smoothed = pd.DataFrame(smoothed, columns=df.columns)
    df_smoothed.x = x.values
    return df_smoothed

def drop_columns(df, pattern=".*H2O"):
    """Drop all columns related to H2O using pattern matching"""
    cols_pattern = list(filter(re.compile(pattern).match, list(df.columns)))
    if cols_pattern:
        return df.drop(columns=cols_pattern, inplace=False)
    return df

def correct_data(df, i=1):
    """Perform baseline correction by substracting the H2O spectrum
    from all other Spectra in the dataframe. 
    i ~ column index of h2o"""
    df_copy = df.copy()
    arr_h2o = df_copy.iloc[:,i].values.reshape(-1,1)
    df_copy.iloc[:,i+1:] = df_copy.iloc[:,i+1:].values - arr_h2o
    return df_copy