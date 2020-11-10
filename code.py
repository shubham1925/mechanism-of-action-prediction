# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:28:30 2020

@author: shubh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\train_features.csv')
'''train data size: 23814 rows, 876 cols'''

data_test = pd.read_csv(r'C:\Users\shubh\Desktop\PMRO\SEM3\ENPM808A - Intro to ML\Final Project\lish-moa\test_features.csv')