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

#############################################################################

import matplotlib.pyplot as plt
# from itertools import combinations
from matplotlib.lines import Line2D
# import mpld3
from matplotlib.widgets import Slider

def plotBothDates(data, title, variant = 2, averaged = True, figSize = (13,7)):
    # variant == "1": Plot Water dec, Water Jan and Pentan Dec and Pentan Jan seperately
    # variant == "1": Plot each concentration separetly, Dates combined
    # averaged : plots average if set to True
    plt.figure(figsize= figSize)#
    x = data.filter(regex='x')
    wasserDec = data.filter(regex='Wasser').filter(regex='17.12')
    wasserJan = data.filter(regex='Wasser').filter(regex='05.01')
    pentanDec = data.filter(regex='Pentan').filter(regex='17.12')
    pentanJan = data.filter(regex='Pentan').filter(regex='05.01')
    
    pentan20 = data.filter(regex='20,|19.8').filter(regex='Pentan')
    pentan50 = data.filter(regex='48.3|49.5|49.7|52.3').filter(regex='Pentan')
    pentan100 = data.filter(regex='101.5|100.8').filter(regex='Pentan')
    
    avg_wasserDec = wasserDec.mean(axis = 1)
    avg_wasserJan = wasserJan.mean(axis = 1)
    avg_pentanDec = pentanDec.mean(axis = 1)
    avg_pentanJan = pentanJan.mean(axis = 1)
    avg_pentan20 = pentan20.mean(axis = 1)
    avg_pentan50 = pentan50.mean(axis = 1)
    avg_pentan100 = pentan100.mean(axis = 1)
        
    legends = []

    ################################################################################################
    # Plot Water dec, Water Jan and Pentan Dec and Pentan Jan seperately 
    if variant == "1":
        nWasserDec = wasserDec.shape[1]
        cmap = plt.get_cmap('Blues')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nWasserDec)]
        legends.append(Line2D([0], [0], color='b', ls="--", lw=1, label="Wasser Dec"))
        
        if averaged == True:
            plt.plot(x, avg_wasserDec, ls="--", linewidth=1, c="b")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, wasserDec[wasserDec.columns[i]], ls="--",linewidth=1, c=color, )

        nWasserJan = wasserJan.shape[1]
        cmap = plt.get_cmap('Greens')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nWasserJan)]
        legends.append(Line2D([0], [0], color='g', ls="--", lw=1, label="Wasser_Jan"))
        
        if averaged == True:
            plt.plot(x, avg_wasserJan, ls="--", linewidth=1, c="g")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, wasserJan[wasserJan.columns[i]], ls="--",linewidth=1, c=color, )

        nPentanDec = pentanDec.shape[1]
        cmap = plt.get_cmap('Reds')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentanDec)]
        legends.append(Line2D([0], [0], color='r', ls="--", lw=1, label="Pentan_Dec"))
        
        if averaged == True:
            plt.plot(x, avg_pentanDec, ls="--", linewidth=1, c="r")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentanDec[pentanDec.columns[i]], ls="--",linewidth=1, c=color, )

        nPentanJan = pentanJan.shape[1]
        cmap = plt.get_cmap('Oranges')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentanJan)]
        legends.append(Line2D([0], [0], color='orange', ls="--", lw=1, label="Pentan_Jan"))
        if averaged == True:
            plt.plot(x, avg_pentanJan, ls="--", linewidth=1, c="orange")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentanJan[pentanJan.columns[i]], ls="--",linewidth=0.3, c=color, )
            
#         print(wasserDec.columns)
#         print(wasserJan.columns)
#         print(pentanDec.columns)
#         print(pentanJan.columns)

    ################################################################################################
    # Plot each concentration separetly, Dates combined

    elif variant == "2":
        nPentan20 = pentan20.shape[1]
        cmap = plt.get_cmap('Greens')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan20)]
        legends.append(Line2D([0], [0], color='g', ls="--", lw=1, label="Pentanediol ~20 mg/ml"))
        
        if averaged == True:
            plt.plot(x, avg_pentan20, ls="--", linewidth=1, c="g")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan20[pentan20.columns[i]], ls="--",linewidth=1, c=color, )
            
        nPentan50 = pentan50.shape[1]
        cmap = plt.get_cmap('Blues')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan50)]
        legends.append(Line2D([0], [0], color='b', ls="--", lw=1, label="Pentanediol ~50 mg/ml"))
        
        if averaged == True:
            plt.plot(x, avg_pentan50, ls="--", linewidth=1, c="b")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan50[pentan50.columns[i]], ls="--",linewidth=0.6, c=color, )
            
        nPentan100 = pentan100.shape[1]
        cmap = plt.get_cmap('Reds')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan100)]
        legends.append(Line2D([0], [0], color='r', ls="--", lw=1, label="Pentanediol ~100 mg/ml"))
        
        if averaged == True:
            plt.plot(x, avg_pentan100, ls="--", linewidth=1, c="r")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan100[pentan100.columns[i]], ls="--",linewidth=0.3, c=color, )
        
#         print(pentan20.columns)
#         print(pentan50.columns)
#         print(pentan100.columns)

    ################################################################################################
    
    plt.title(title, fontsize=18, y=1.02)
    plt.xlabel('Ramanshift $[cm^{-1}]$', fontsize=12)
    plt.ylabel('Intensität [mW]', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    legend_elements = legends
    plt.legend(handles=legend_elements, fontsize=12, loc = 2)
    plt.show()

def plotSameConcentration(data, title, variant, averaged=False, figSize = (13,7)):
    # Plots one concentration with separated Dates
    # variant: "1" --> 20
    #          "2" --> 50
    #          "3" --> 100
    # averaged: plots average if set to True
    
    plt.figure(figsize=figSize)#
    x = data.filter(regex='x')
    
    wasserDec = data.filter(regex='Wasser').filter(regex='17.12')
    wasserJan = data.filter(regex='Wasser').filter(regex='05.01')
    pentanDec = data.filter(regex='Pentan').filter(regex='17.12')
    pentanJan = data.filter(regex='Pentan').filter(regex='05.01')
    
    pentan20 = data.filter(regex='Pentan').filter(regex='20,|19.8')
    pentan20_1 = pentan20.filter(regex='17.12')
    pentan20_2 = pentan20.filter(regex='05.01')
    
    pentan50 = data.filter(regex='Pentan').filter(regex='48.3|49.5|49.7|52.3')
    pentan50_1 = pentan50.filter(regex='17.12')
    pentan50_2 = pentan50.filter(regex='05.01')
    
    pentan100 = data.filter(regex='Pentan').filter(regex='101.5|100.8')
    pentan100_1 = pentan100.filter(regex='17.12')
    pentan100_2 = pentan100.filter(regex='05.01')
    
    avg_pentan20_1 = pentan20_1.mean(axis = 1)
    avg_pentan50_1 = pentan50_1.mean(axis = 1)
    avg_pentan100_1 = pentan100_1.mean(axis = 1)
    avg_pentan20_2 = pentan20_2.mean(axis = 1)
    avg_pentan50_2 = pentan50_2.mean(axis = 1)
    avg_pentan100_2 = pentan100_2.mean(axis = 1)
    
    legends = []

    ################################################################################################
    # Plot Dates separetly, concentration 20
    if variant == "1":
        nPentan20_1 = pentan20_1.shape[1]
        cmap = plt.get_cmap('Blues')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan20_1)]
        legends.append(Line2D([0], [0], color='b', ls="--", lw=1, label="{0}".format(pentan20_1.columns[0][:-2])))
        
        if averaged == True:
            plt.plot(x, avg_pentan20_1, ls="--", linewidth=1, c="b")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan20_1[pentan20_1.columns[i]], ls="--",linewidth=1, c=color, )

        nPentan20_2 = pentan20_2.shape[1]
        cmap = plt.get_cmap('Reds')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan20_2)]
        legends.append(Line2D([0], [0], color='r', ls="--", lw=1, label="{0}".format(pentan20_2.columns[0][:-2])))
        
        if averaged == True:
            plt.plot(x, avg_pentan20_2, ls="--", linewidth=1, c="r")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan20_2[pentan20_2.columns[i]],lw=0.3, ls="--", c=color, )
            
#         print(pentan20_1.columns)
#         print(pentan20_2.columns)


    ################################################################################################
    # Plot Dates separetly, concentration 50
    elif variant == "2":
        nPentan50_1 = pentan50_1.shape[1]
        cmap = plt.get_cmap('Blues')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan50_1)]
        legends.append(Line2D([0], [0], color='b', ls="--", lw=1, label="{0}".format(pentan50_1.columns[0][:-2])))
                        
        if averaged == True:
            plt.plot(x, avg_pentan50_1, ls="--", linewidth=1, c="b")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan50_1[pentan50_1.columns[i]], ls="--",linewidth=1, c=color)

        nPentan50_2 = pentan50_2.shape[1]
        cmap = plt.get_cmap('Reds')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan50_2)]
        legends.append(Line2D([0], [0], color='r', ls="--", lw=1, label="{0}".format(pentan50_2.columns[0][:-2])))
        if averaged == True:
            plt.plot(x, avg_pentan50_2, ls="--", linewidth=1, c="r")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan50_2[pentan50_2.columns[i]],lw=0.3, ls="--", c=color)
        
#         print(pentan50_1.columns)
#         print(pentan50_2.columns)

    ################################################################################################
    # Plot Dates separetly, concentration 20
    elif variant == "3":
        nPentan100_1 = pentan100_1.shape[1]
        cmap = plt.get_cmap('Blues')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan100_1)]
        legends.append(Line2D([0], [0], color='b', ls="--", lw=1, label="{0}".format(pentan100_1.columns[0][:-2])))
                        
        if averaged == True:
            plt.plot(x, avg_pentan100_1, ls="--", linewidth=1, c="b")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan100_1[pentan100_1.columns[i]], ls="--",linewidth=1, c=color)

        nPentan100_2 = pentan100_2.shape[1]
        cmap = plt.get_cmap('Reds')
        colors = [cmap(i) for i in np.linspace(0.5, 1, nPentan100_2)]
        legends.append(Line2D([0], [0], color='r', ls="--", lw=1, label="{0}".format(pentan100_2.columns[0][:-2])))
        if averaged == True:
            plt.plot(x, avg_pentan100_2, ls="--", linewidth=1, c="r")
        else:
            for i, color in enumerate(colors, start=0):
                plt.plot(x, pentan100_2[pentan100_2.columns[i]],lw=0.3, ls="--", c=color)
            
#         print(pentan100_1.columns)
#         print(pentan100_2.columns)
            
    ################################################################################################
    
    plt.title(title, fontsize=18, y=1.02)
    plt.xlabel('Ramanshift $[cm^{-1}]$', fontsize=12)
    plt.ylabel('Intensität [mW]', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    legend_elements = legends
    plt.legend(handles=legend_elements, fontsize=12, loc = 2)
    plt.show()