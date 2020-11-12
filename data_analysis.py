# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:19:57 2020

@author: shubh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_features.csv')
data_test = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\test_features.csv')
data_target_nonscored = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_targets_nonscored.csv')
data_target_scored = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_targets_scored.csv')

print('Train data size: ', data_train.shape)
print('Test data size: ', data_test.shape)
print('target_scored size: ', data_target_scored.shape)
print('target_nonscored size: ', data_target_nonscored.shape)

'''visualize gene expression features'''
gene_feat = [cols for cols in data_train.columns if cols.startswith('g-')]
fig1 = plt.figure(1)
for i in range(0, 6):
    plt.subplot(3, 6, i+1)
    sns.kdeplot(data_train.loc[:, gene_feat[i]], color = 'red', shade = True)
    plt.xlabel(gene_feat[i])
#plt.show()

'''visulaize cell viability features'''
cell_feat = [cols for cols in data_train.columns if cols.startswith('c-')]
fig2 = plt.figure(2)
for i in range(0, 10):
    plt.subplot(3, 6, i+1)
    sns.kdeplot(data_train.loc[:, cell_feat[i]], color = 'blue', shade = True)
    plt.xlabel(cell_feat[i])
plt.show()

'''relationship between random feature and random target'''
train_copy = data_train.copy()
train_copy['target_71'] = data_target_scored.iloc[:, 72]
fig3 = plt.figure(3)
sns.stripplot(data = train_copy, x = 'cp_time', y = 'g-1', color = 'blue', hue = 'target_71')
plt.show()
fig4 = plt.figure(4)
sns.stripplot(data = train_copy, x = 'cp_dose', y = 'c-1', color = 'red', hue = 'target_71')
plt.show()

train_copy['g_mean'] = train_copy.loc[:, gene_feat].mean(axis = 1)
train_copy['c_mean'] = train_copy.loc[:, cell_feat].mean(axis = 1)
fig4 = plt.figure(5)
ax1 = fig4.add_subplot(121)
sns.stripplot(data = train_copy, x = 'cp_time', y = 'g_mean', color = 'blue', hue = 'target_71', ax = ax1)
ax2 = fig4.add_subplot(122)
sns.stripplot(data = train_copy, x = 'cp_dose', y = 'g_mean', color = 'blue', hue = 'target_71', ax = ax2)
plt.show()
