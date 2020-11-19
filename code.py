# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:28:30 2020

@author: shubh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
import tensorflow as tf


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

from sklearn.cluster import KMeans

#data_train = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_features.csv')
#data_test = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\test_features.csv')

def fe_clusters(train, test, n_clusters_g = 35, n_clusters_c = 5, seed = 200):
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    def create_clusters(train, test, features, kind, n_clusters):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters, random_state = seed).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_clusters(train, test, features_g, 'g', n_clusters_g)
    train, test = create_clusters(train, test, features_c, 'c', n_clusters_c)
    return train, test

print("here")
X, X_test = fe_clusters(X, X_test)
print(X.shape)
print(' ')
print(X_test.shape)
X.drop(['cp_type'], axis = 1, inplace = True)
X_test.drop(['cp_type'], axis = 1, inplace = True)

import tensorflow.keras.backend as K

def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0.001, 0.999)
    return -K.mean(y_true*K.log(y_pred) + (1 - y_true)*K.log(1 - y_pred))

def create_model(hp):
    num_cols = X.shape[1]
    inp = tf.keras.layers.Input(shape = (num_cols, ))
    x = tf.keras.layers.BatchNormalization()(inp)
    num_dense = hp.Int('num_dense', min_value = 0, max_value = 3, step = 1)
    for i in range(num_dense):
        hp_units = hp.Int('units_{i}'.format(i=i), min_value = 128, max_value = 4096, step = 128)
        hp_drop_rate = hp.Choice('dp_{i}'.format(i=i), values = [0.25, 0.3, 0.35, 0.40, 0.45, 0.60, 0.65, 0.70])
        hp_activation = hp.Choice('dense_activation_{i}'.format(i=i), values = ['relu', 'selu', 'elu', 'swish'])
        x = tf.keras.layers.Dense(units = hp_units, activation = hp_activation)(x)
        x = tf.keras.layers.Dropout(hp_drop_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(206, activation = 'sigmoid')(x)
    model = tf.keras.Model(inp, outputs)
    learning_rate = 1e-3
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.001), metrics = logloss)
    return model  
