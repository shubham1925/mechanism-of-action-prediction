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
    data['cp_time'] = data['cp_time'].map({24:0, 48:1, 72:2})
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

gene_features = [cols for cols in X.columns if cols.startswith('g-')]
cell_features = [cols for cols in X.columns if cols.startswith('c-')]

PCA_transform = PCA(0.95, random_state = 42)
data = pd.concat([pd.DataFrame(X[gene_features]), pd.DataFrame(X_test[gene_features])])
data2 = PCA_transform.fit_transform(data[gene_features])
train2 = data2[:X.shape[0]]
test2 = data2[-X_test.shape[0]:]

train2 = pd.DataFrame(train2, columns = [f'pca_g-{i}' for i in range(data2.shape[1])])
test2 = pd.DataFrame(test2, columns = [f'pca_g-{i}' for i in range(data2.shape[1])])

X = pd.concat((X, train2), axis = 1)
X_test = pd.concat((X_test, test2), axis = 1)

data3 = pd.concat([pd.DataFrame(X[cell_features]), pd.DataFrame(X_test[cell_features])])
data2 = PCA_transform.fit_transform(data3[cell_features])
train2 = data2[:X.shape[0]]
test2 = data2[-X_test.shape[0]:]

train2 = pd.DataFrame(train2, columns = [f'pca_c-{i}' for i in range(data2.shape[1])])
test2 = pd.DataFrame(test2, columns = [f'pca_c-{i}' for i in range(data2.shape[1])])

X = pd.concat((X, train2), axis = 1)
X_test = pd.concat((X_test, test2), axis = 1)

from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(0.8)
data = X.append(X_test)
data_transformed = vt.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[:X.shape[0]]
test_features_transformed = data_transformed[-X_test.shape[0]:]

X = pd.DataFrame(X[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4) , columns = ['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
X_test = pd.DataFrame(X_test[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4) , columns = ['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

X = pd.concat([X, pd.DataFrame(train_features_transformed)], axis = 1)
X_test = pd.concat([X_test, pd.DataFrame(test_features_transformed)], axis = 1)
