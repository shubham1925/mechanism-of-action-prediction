# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:28:30 2020

@author: shubh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA


data_train = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_features.csv')
'''train data size: 23814 rows, 876 cols'''

data_test = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\test_features.csv')
'''test data size: 3982 rows, 876 cols'''

data_target_nonscored = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_targets_nonscored.csv')
'''train target nonscored data size: 23814 rows, 403 col'''

data_target_scored = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_targets_scored.csv')
'''train target scored data size: 23814 rows, 207 col'''



ind_tr = data_train[data_train['cp_type'] == 'ctl_vehicle'].index
ind_te = data_test[data_test['cp_type'] == 'ctl_vehicle'].index

def preprocess(data):
    transformer = QuantileTransformer(n_quantiles = 100, random_state = 42, output_distribution = "normal")
    '''encode cp_time and cp_dose into discrete integers'''
    data['cp_time'] = data['cp_time'].map({24:1, 48:2, 72:3})
    data['cp_dose'] = data['cp_dose'].map({'D1':0, 'D2':1})
    gene_features = [cols for cols in data.columns if cols.startswith('g-')]
    cell_features = [cols for cols in data.columns if cols.startswith('c-')]
    for col in gene_features:
        vec_len = len(data[col].values)
        raw_vec = data[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        data[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    for col in cell_features:
        vec_len = len(data[col].values)
        raw_vec = data[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        data[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    return data

X_test = preprocess(data_test)
X = preprocess(data_train)

