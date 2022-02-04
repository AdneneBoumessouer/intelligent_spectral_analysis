#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:11:57 2022

@author: aboumessouer
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# import feature matrices
X_train = pd.read_csv(Path('data/preprocessed/X_train.csv'))
X_test = pd.read_csv(Path('data/preprocessed/X_test.csv'))
y_train = pd.read_csv(Path('data/preprocessed/y_train.csv'))
y_test = pd.read_csv(Path('data/preprocessed/y_test.csv'))


from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# linear models
# https://scikit-learn.org/stable/modules/linear_model.html
ridge = Ridge(alpha=1, solver="cholesky")
lasso = Lasso(alpha=0.1)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Random Forest Regressor
# https://scikit-learn.org/stable/modules/ensemble.html
random_forest = RandomForestRegressor(random_state=1)

# MLP
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
mlp = MLPRegressor(hidden_layer_sizes=(10, 10),
                       solver='lbfgs', # small dataset
                       alpha = 1e-4, # L2 regularization
                       learning_rate_init=1e-3,
                       learning_rate='adaptive',
                       random_state=1,
                       batch_size=20)


# training experiment
ridge.fit(X_train, y_train)
some_data = X_train.iloc[:5]
some_labels = y_train.iloc[:5]
print("Predictions:", ridge.predict(some_data))
print("Labels:", some_labels)



# cross validation
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

from sklearn.model_selection import cross_val_score

def cross_validate_regressors(regressors, names, X_train, y_train, cv=10):
    scores = []
    for regressor in regressors:
        scores.append(-cross_val_score(regressor, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv))
    scores = np.array(scores).T
    df_scores = pd.DataFrame(data=scores, columns=names)
    mean, std = df_scores.mean(axis=0), df_scores.std(axis=0)
    df_scores.loc['mean'] = mean
    df_scores.loc['std'] = std
    return df_scores

regressors = [ridge, lasso, elastic_net]
names = ['ridge', 'lasso', 'elastic_net']
result = cross_validate_regressors(regressors, names, X_train, y_train, cv=10)


# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
