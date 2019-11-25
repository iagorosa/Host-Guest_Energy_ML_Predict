#%%
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution as de
import glob as gl
import pylab as pl
import os

path='./pkl/'
os.system('mkdir  '+path)

basename='host_guest_ml___'

datasets=[]
xls=gl.glob('./data/*.csv')

#%%

for f in xls:
    X=pd.read_csv(f)
       
    cols_to_remove  = ['BindingDB ITC_result_a_b_ab_id']
    cols_target     = ['Delta_G0 (kJ/mol)']

    X.drop(cols_to_remove,  axis=1, inplace=True)
    y_train = X[cols_target]
    X.drop(cols_target,  axis=1, inplace=True)
    X_train = X
    
    y_train.columns=['delta_g0']
    
    dataset = {}
    dataset['var_names'], dataset['target_names'] = X_train.columns, y_train.columns
    dataset['name'] = f.split('.xlsx')[0].split('/')[-1]
    dataset['X_train'], dataset['y_train'], = X_train.values, [y_train.values]
    dataset['n_samples'], dataset['n_features'] = X_train.shape
    dataset['task'] = 'regression'
    datasets.append(dataset)
     
#%%