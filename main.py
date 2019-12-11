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

# pop_size = 50
# max_iter=50
# n_splits = 5

run0 = 0
n_runs = 1

for run in range(run0, n_runs):
    random_seed=run*10+100

    for dataset in datasets:#[:1]:   

        target, y_              = dataset['target_names'], dataset['y_train']
        dataset_name, X_        = dataset['name'], dataset['X_train']
        n_samples, n_features   = dataset['n_samples'], dataset['n_features']
        task                    = dataset['task']

        # print(dataset_name, target, n_samples, n_features,)
        np.random.seed(random_seed)

        list_results=[]
        print('='*80+'\n'+dataset_name+': '+target+'\n'+'='*80+'\n')
        
        if task=='classification':
            le = preprocessing.LabelEncoder()
            #le=preprocessing.LabelBinarizer()
            le.fit(y_)
            y=le.transform(y_)
        else:
            y=y_.copy()
        
        X=X_.copy()
        ##scale_X = MinMaxScaler(feature_range=(0.15,0.85)).fit(X_)
        #scale_X = MinMaxScaler(feature_range=(0,1)).fit(X_)
        #X= scale_X.transform(X_)
        ##   scale_y = MinMaxScaler(feature_range=(0.15,0.85)).fit(y_)    
        ##   X,y = scale_X.transform(X_), scale_y.transform(y_)

        args = (X, y, 'eval', n_splits, random_seed)
    
        optimizers=[                
            #('EN'   , get_parameters('EN'), fun_en_fs, args, random_seed,),
            ('XGB'   , get_parameters('XGB'), args, random_seed,),
            #('DT'   , get_parameters('DT'), args, random_seed,),
            #('VC'   , get_parameters('VC'), args, random_seed,),
            #('BAG'  , get_parameters('BAG'), args, random_seed,),
            #('KNN'  , get_parameters('KNN'), args, random_seed,),
            ('ANN'  , get_parameters('ANN'), args, random_seed,),
            ('ELM'  , get_parameters('ELM'), args, random_seed,),
            ('SVR'  , get_parameters('SVR'), args, random_seed,),
            #('MLP'  , get_parameters('MLP'), args, random_seed,),
            #('GB'   , get_parameters('GB'), args, random_seed,),      
            #('KRR'  , get_parameters('KRR'), args, random_seed,),
            #('CAT'  , get_parameters('CAT'), args, random_seed,),
            ]


#%%

def get_parameters(opt):

    if opt == 'ANN':
        lb = [0, 0, 1e-6, 1,   1, 1, 1, 1, 1,] #+ [0.0]*n_features     
        ub = [3, 2, 1e-2, 5,  50,50,50,50,50,] #+ [1.0]*n_features

    elif opt == 'EN':
        lb = [0, 0, 0, ] #+ [0.0]*n_features          
        ub = [2, 1, 1,] #+ [1.0]*n_features

    elif opt == 'MPL':
        lb =[0, 0,     1,  1,  1,  1,  1,  1,] #+ [0.0]*n_features
        ub =[1, 1,     5, 50, 50, 50, 50, 50,] #+ [1.0]*n_features, "rb"))

    elif opt == 'GB':
        lb  = [0.001,  100,  10,  5,  5,   0, 0.1, ] #+ [0.0]*n_features
        ub  = [  0.8,  900, 100, 50, 50, 0.5, 1.0, ] #+ [1.0]*n_features
    
    elif opt == 'SVR':
        lb = [0, 0, -0.1, 1e-6, 1e-6, 1e-6, ]#+ [0.0]*n_features, "rb"))
        ub = [5, 5,    2, 1e+4, 1e+4,    4,]#+ [1.0]*n_features

    elif opt == 'KNN':
        lb = [ 1,  0, 1] #+ [0.0]*n_features
        ub = [50,  1, 3] #+ [1.0]*n_features
    
    elif opt == 'ELM':
        lb = [1e-0,   0,   0,   0.01, ] #+ [0.0]*n_features
        ub = [5e+2,   5,   1,  10.00, ] #+ [1.0]*n_features
    
    elif opt == 'VC':
        lb  = [ 0]*2 + [ 0,   0,]#+ [0.0]*n_features
        ub  = [ 1]*2 + [ 9, 300,]#+ [1.0]*n_features

    elif opt == 'BAG':
        lb  = [0,  10, ]#+ [0.0]*n_features
        ub  = [1, 900, ]#+ [1.0]*n_features

    elif opt == 'DT':
        lb  = [0,  2, ]#+ [0.0]*n_features
        ub  = [1, 20, ]#+ [1.0]*n_features
    
    elif opt == 'KRR':
        lb = [0., 0,  0.0, 1,   0,  1e-6]#+ [0.0]*n_features
        ub = [1., 5, 10.0, 5, 1e3,  1e+4]#+ [1.0]*n_features

    elif opt == 'XGB':
        lb = [0.0,  10,  1, 0.0,  1, 0.0]#+ [0.0]*n_features
        ub = [1.0, 900, 30, 1.0, 10, 1.0]#+ [1.0]*n_features

    elif opt == 'CAT':
        lb = [0.0,  10,  1,    0.,  1., 0.0]#+ [0.0]*n_features
        ub = [1.0, 900, 16, 1000., 50., 1.0]#+ [1.0]*n_features

    return lb, ub

# %%
