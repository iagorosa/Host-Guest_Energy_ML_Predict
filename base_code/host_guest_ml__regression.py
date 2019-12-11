# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
import pygmo as pg
import pickle

from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold, cross_val_predict,train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics.classification import accuracy_score, f1_score, precision_score
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from MLP import MLPRegressor as MLPR
#from sklearn.neural_network import MLPRegressor
from ELM import  ELMRegressor, ELMRegressor
from xgboost import  XGBRegressor
#from pyswarm import pso
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import Pool, CatBoostRegressor

#%%----------------------------------------------------------------------------
pd.options.display.float_format = '{:20,.3f}'.format
import warnings
warnings.filterwarnings('ignore')

import sys, getopt
program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

print ("This is the name of the script: ", program_name)
print ("Number of arguments: ", len(arguments))
print ("The arguments are: " , arguments)

if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0+1
else:
    run0, n_runs = 0,1
    
#%%----------------------------------------------------------------------------
#from read_data import *
#datasets = [
#                read_data_cergy(),
#                read_data_efficiency(),
#                read_data_bogas(),
#                read_data_dutos_csv(),host_guest_ml__regression.py
#                read_data_yeh(),
#                read_data_lim(),
#                read_data_siddique(),
#                read_data_pala(),
#                read_data_bituminous_marshall(),
#                read_data_slump(),
#                read_data_shamiri(),
#                read_data_nguyen_01(),
#                read_data_nguyen_02(),
#                read_data_tahiri(),
#                read_data_diego(),
#            ]

#%%----------------------------------------------------------------------------   
def fun_en_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  clf = ElasticNet(random_state=random_seed,)
  p={
     'alpha': x[0],
     'l1_ratio': x[1],
     'positive': x[2]<0.5
    }
  clf.set_params(**p)
  
  if len(x)<=3:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)

  #x[4::] = [1 if k>0.5 else 0 for k in x[4::]]
  #ft = np.array([1 if k>0.5 else 0 for k in x[4::]])
  #ft = np.where(ft>0.5)
  try:
    #cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    #cv=KFold(n=n_samples, n_folds=5, shuffle=True, random_state=int(random_seed))
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    r = -r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
  except:
    y_p=[None]
    r=1e12
    
  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
      clf.fit(X[:,ft].squeeze(), y)
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'DT',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}

#%%----------------------------------------------------------------------------   
def fun_dtc_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  clf = DecisionTreeRegressor(random_state=random_seed,)
  #clf = RandomForestRegressor(random_state=random_seed, n_estimators=100)
  p={
     'criterion': 'gini' if x[0] < 0.5 else 'entropy',
     'min_samples_split': int(x[1]),
    }
  clf.set_params(**p)
  
  if len(x)<=2:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)

  #x[4::] = [1 if k>0.5 else 0 for k in x[4::]]
  #ft = np.array([1 if k>0.5 else 0 for k in x[4::]])
  #ft = np.where(ft>0.5)
  try:
    #cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    #cv=KFold(n=n_samples, n_folds=5, shuffle=True, random_state=int(random_seed))
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y.ravel(), cv=cv, n_jobs=1)
    #r = r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    r = RMSE(y_p,y)
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
  except:
    y_p=[None]
    r=1e12
    
  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
    clf.fit(X[:,ft].squeeze(), y)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'DT',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}


def fun_bag_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  _estimator=[None,None]
  base_estimator=_estimator[int(round(x[0]))]
  n_estimators=int(round(x[1]))
  clf = BaggingRegressor(random_state=random_seed,)
  p={
     'base_estimator':base_estimator, 
     'n_estimators':n_estimators,
    }
  clf.set_params(**p)
  
  if len(x)<=2:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)

  #x[4::] = [1 if k>0.5 else 0 for k in x[4::]]
  #ft = np.array([1 if k>0.5 else 0 for k in x[4::]])
  #ft = np.where(ft>0.5)
  try:
    #cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    #cv=KFold(n=n_samples, n_folds=5, shuffle=True, random_state=int(random_seed))
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    #r = r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
    r = RMSE(y_p,y)
  except:
    y_p=[None]
    r=1e12
    
  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
      clf.fit(X[:,ft].squeeze(), y)
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'BAG',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}


def fun_ann_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  #n_hidden = int(round(x[1]))
  #hidden_layer_sizes = tuple( int(round(x[2+i])) for i in range(n_hidden))
  n_hidden = int(round(x[3]))
  hidden_layer_sizes = tuple( int(round(x[4+i])) for i in range(n_hidden))
  
  af = {
          0 :'logistic', 
          1 :'identity', 
          2 :'relu', 
          3 :'tanh',
      }  
  
  s = {
        0: 'lbfgs',
        1: 'sgd',
        2: 'adam',
      }

  p={
     'activation': af[int(round(x[0]))],
     'hidden_layer_sizes':hidden_layer_sizes,
     #'alpha':1e-5, 'solver':'lbfgs',
     'solver': s[int(round(x[1]))],'alpha': x[2],
     }
  
  clf = MLPRegressor(random_state=int(random_seed), warm_start=False)
  clf.set_params(**p)
  
  if len(x)<=9:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[4::]])
      ft = np.where(ft>0.5)

  #x[4::] = [1 if k>0.5 else 0 for k in x[4::]]
  #ft = np.array([1 if k>0.5 else 0 for k in x[4::]])
  #ft = np.where(ft>0.5)
  try:
    #cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    #cv=KFold(n=n_samples, n_folds=5, shuffle=True, ranwarm_start=Falsedom_state=int(random_seed))
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    #r = -r2_score(y_p,y)
    r = RMSE(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  MAPE(y,y_p)
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
  except:
    y_p=[None]
    r=1e12

  #print(r,'\t',p,)#'\t',ft)  
  if flag=='eval':
      return r
  else:
      clf.fit(X[:,ft].squeeze(), y)
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'ANN',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}


def fun_mlp_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  n_hidden = int(round(x[2]))
  hidden_layer_sizes = [ int(round(x[3+i])) for i in range(n_hidden)]
  #hidden_layer_sizes = [ int(round(x[2])) for i in range(n_hidden)]
  con = {0: 'mlgraph', 1:'tmlgraph',}
  p={
     'connectivity': con[int(round(x[0]))],
     'bias':bool(round(x[1])),
     #'renormalize':bool(round(x[2])),
     'n_hidden':hidden_layer_sizes, 
     #'algorithm':['tnc', 'l-bfgs', 'sgd', 'rprop', 'genetic'],
     }
  clf = MLPR(algorithm = 'tnc',max_iter=1000, renormalize=True)
  clf.set_params(**p)
  
  if len(x)<=8:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)
  
  try:
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1, verbose=0)
    #r = r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    r =  MAPE(y,y_p)
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y.argmax(axis=1),y_p.argmax(axis=1),average='weighted')
  except:
    y_p=[None]
    r=1e12
    
  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
        clf.fit(X[:,ft].squeeze(), y)
        return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'MLP',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}
    
  
def fun_elm_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  clf = ELMRegressor(random_state=int(random_seed))
  n_samples, n_var = X.shape
  af = {
      #0 :'tribas', 
      0 :'identity', 
      1 :'relu', 
      2 :'swish',
      #4 :'inv_tribase', 
      #5 :'hardlim', 
      #6 :'softlim', 
      3 :'gaussian', 
      4 :'multiquadric', 
      5 :'inv_multiquadric',
  }

  regressor = None #if x[4]<1e-8 else Ridge(alpha=x[4],random_state=int(random_seed))
  p={'n_hidden':int(x[0]), #'alpha':1, 'rbf_width':1,
     'activation_func': af[int(x[1]+0.5)], #'alpha':0.5, 
     'alpha':x[2], 
     'rbf_width':x[3],
     #'regressor':regressor,
     }
  clf.set_params(**p)
  #x[2::] = [1 if k>0.5 else 0 for k in x[4::]]
  if len(x)<=4:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)
      
  #ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
  #ft = np.where(ft>0.5)
  try:
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    #r = -r2_score(y_p,y)
    r = RMSE(y_p,y)
    #r = median_absolute_error(y_p,y)    
    #r = MAPE(y_p,y)
    #r = RMSE(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
    #print(r,'\t',p)  
  except:
    y_p=[None]
    r=1e12
    
  if flag=='eval':
      return r
  else:
    clf.fit(X[:,ft].squeeze(), y)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'ELM',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}
    

def fun_cat_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  clf = CatBoostRegressor(random_state=int(random_seed),verbose=0)
  
  n_samples, n_var = X.shape

#  cr ={
#        0:'reg:linear',
#        1:'reg:logistic',
#        2:'binary:logistic',
#       }
       
  #x=[0.1, 200, 5, 2.5, 10.0, 0.8, ]
  p={
     'learning_rate': x[0],
     'n_estimators':int(round(x[1])), 
     'depth':int(round(x[2])),
     'loss_function':'RMSE',
     'l2_leaf_reg':x[3],
     'bagging_temperature':x[4],
     #'boosting_type':'Pĺain',
     #'colsample_bytree':x[3],
     #'min_child_weight':int(round(x[4])),
     #'bootstrap_type':'Bernoulli',
     #'subsample':int(x[5]*1000)/1000,
     ##'alpha':x[6],
     #'objective':cr[0],
     ##'presort':ps[0],
     }
    
  clf.set_params(**p)
  #x[2::] = [1 if k>0.5 else 0 for k in x[4::]]
  if len(x)<=6:
      ft = np.array([1 for i in range(n_var)])
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])

  ft = np.where(ft>0.5)
  
  try:
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X, y.ravel(),cv=cv,n_jobs=1)
    r = r2_score(y_p,y)
    #r = MAPE(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)    
    #r =  -f1_score(y,y_p,average='weighted')
    #r =  -precision_score(y,y_p)  
    #print(r,p)
  except:
    y_p=[None]
    r=1e12


  print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
         clf.fit(X[:,ft].squeeze(), y)
         return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 
              'EST_NAME':'CAT','ESTIMATOR':clf, 'ACTIVE_VAR':ft, 
              'DATA':X, 'SEED':random_seed}


def fun_xgb_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  clf = XGBRegressor(random_state=int(random_seed))
  n_samples, n_var = X.shape

  cr ={
        0:'reg:squarederror',
        1:'reg:logistic',
        2:'binary:logistic',
       }
       
  #x=[0.1, 200, 5, 0.3, 2, 0.8, ]
  p={
     'learning_rate': x[0],
     'n_estimators':int(round(x[1])), 
     'max_depth':int(round(x[2])),
     'colsample_bytree':x[3],
     'min_child_weight':int(round(x[4])),
     'subsample':int(x[5]*1000)/1000,
     #'alpha':x[6],
     'objective':cr[0],
     #'presort':ps[0],
     #'max_iter':1000,
     }
    
  
  clf.set_params(**p)
  #x[2::] = [1 if k>0.5 else 0 for k in x[4::]]
  if len(x)<=6:
      ft = np.array([1 for i in range(n_var)])
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])

  ft = np.where(ft>0.5)
      
  
  try:
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X, np.ravel(y),cv=cv,n_jobs=1)
    #r = -r2_score(y_p,y)
    r = RMSE(y_p,y)
    #r = MAPE(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)    
    #r =  -f1_score(y,y_p,average='weighted')
    #r =  -precision_score(y,y_p)  
    #print(r,p)
  except:
    y_p=[None]
    r=1e12


  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
         clf.fit(X[:,ft].squeeze(), y)
         return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'XGB',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}


def fun_gb_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  clf = GradientBoostingRegressor(random_state=int(random_seed))
  n_samples, n_var = X.shape

  cr ={
        0:'friedman_mse',
        1:'mse',
        2:'mae',
       }
       
  ps = { 
          0:'auto',
          1:'bool',
      }
    
  #p={'n_hidden':int(x[0]), 'alpha':x[1], 'rbf_width':x[2], 
  #   'activation_func': af[int(x[3]+0.5)]}
  p={
     'learning_rate': x[0],
     'min_samples_leaf':int(round(x[4])),
     'max_depth':int(round(x[2])),
     'n_estimators':int(round(x[1])), 
     'min_samples_split':int(round(x[3])),
     'min_weight_fraction_leaf':x[5],
     'subsample':x[6],
     'criterion':cr[0],
     'presort':ps[0],
     }
    
  
  clf.set_params(**p)
  #x[2::] = [1 if k>0.5 else 0 for k in x[4::]]
  if len(x)<=7:
      ft = np.array([1 for i in range(n_var)])
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])

  ft = np.where(ft>0.5)
      
  try:
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X, y,cv=cv,n_jobs=1)
    #r = r2_score(y_p,y)
    r = RMSE(y_p,y)
    #r = MAPE(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)    
    #r =  -f1_score(y,y_p,average='weighted')
    #r =  -precision_score(y,y_p)  
    #print(r,p)
  except:
    y_p=[None]
    r=1e12


  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
          clf.fit(X[:,ft].squeeze(), y)
          return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'GB',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}
    
    
  
def fun_svm_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  clf = SVR(kernel='rbf',)
  n_samples, n_var = X.shape
  
  kernel = {2:'linear', 3:'poly', 0:'rbf', 1:'sigmoid', 4:'laplacian', 5:'chi2'}  
  
  #p={'C':x[0], 'kernel':kernel[int(round(x[2]))], 'gamma':x[1]}

  p={
     'kernel':kernel[int(round(x[0]))], 
     'degree':int(round(x[1])),
     'gamma': 'scale' if x[2]<0 else x[2],
     'coef0':x[3],
     'C':x[4],
     'epsilon':x[5],
     'max_iter':4000,
  }
  
  clf.set_params(**p)
  n_param=len(p)
  if len(x)<=n_param:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)

  n_splits=n_splits
  try:
    #cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    #cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))    
    y_p  = cross_val_predict(clf,X, y,cv=cv,n_jobs=1)
    #r = -r2_score(y_p,y)
    r = RMSE(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)    
    #r =  -precision_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')  
    #r =  MAPE(y,y_p)
    #r = RMSE(y_p,y)
  except:
    y_p=[None]
    r=1e12
  
  #print (r,'\t',p,'\t',ft)  
  #print (r)
  if flag=='eval':
      return r
  else:
         clf.fit(X[:,ft].squeeze(), y)
         return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'SVR',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed,              }

  
def fun_krr_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  clf = KernelRidge(kernel='rbf',)
  n_samples, n_var = X.shape
  
  kernel = {2:'linear', 3:'poly', 0:'rbf', 1:'sigmoid', 4:'laplacian', 5:'chi2'}  
  p={
     'alpha':x[0],
     'kernel':kernel[int(round(x[1]))], 
     'gamma':x[2],
     'degree':int(round(x[3])),
     'coef0':x[4],
     'kernel_params':{'C':x[5],},
     }
  clf.set_params(**p)
  n_param=len(p)
  if len(x)<=n_param:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)

  n_splits=n_splits
  try:
    #cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    #cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))    
    y_p  = cross_val_predict(clf,X, y,cv=cv,n_jobs=1)
    #r = r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    r = RMSE(y_p,y)
    #r =  -accuracy_score(y,y_p)    
    #r =  -precision_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')  
    #r =  MAPE(y,y_p)
  except:
    y_p=[None]
    r=1e12
  
  print (r,'\t',p,'\t',ft)  
  if flag=='eval':
      return r
  else:
       clf.fit(X[:,ft].squeeze(), y)
       return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'KRR',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed,              }
  

def fun_knn_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  w = {0 :'uniform', 1 :'distance', }   
  
  p={
      'p': int(round(x[2])), 
      'n_neighbors': int(round(x[0])),
      'weights':w[int(round(x[1]))],
  }
     
  clf = KNeighborsRegressor()
  clf.set_params(**p)
  
  if len(x)<=3:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)

  #x[4::] = [1 if k>0.5 else 0 for k in x[4::]]
  #ft = np.array([1 if k>0.5 else 0 for k in x[4::]])
  #ft = np.where(ft>0.5)
  n_splits=n_splits
  try:
    #cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    #cv=KFold(n=n_samples, n_folds=5, shuffle=True, random_state=int(random_seed))
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    #r = -r2_score(y_p,y)
    r = RMSE(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
  except:
    y_p=[None]
    r=1e12
    
  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
      clf.fit(X[:,ft].squeeze(), y)
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'KNN',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed}


def lhsu(xmin,xmax,nsample):
   nvar=len(xmin); ran=np.random.rand(nsample,nvar); s=np.zeros((nsample,nvar));
   for j in range(nvar):
       idx=np.random.permutation(nsample)
       P =(idx.T-ran[:,j])/nsample
       s[:,j] = xmin[j] + P*(xmax[j]-xmin[j]);
       
   return s


def RMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    error = y -  y_pred    
    return np.sqrt(np.mean(np.power(error, 2)))

def RRMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    return RMSE(y, y_pred)*100/np.mean(y)

def MAPE(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100
  #return RMSE(y, y_pred)

pd.options.display.float_format = '{:.3f}'.format

def fmt(x): 
    if (type(x) == str or type(x) == tuple or type(x) == list):
        return str(x)
    else:
      if (abs(x)>0.001 and abs(x)<1e4):
        return '%1.3f' % x   
      else:
        return '%1.4g' % x
    
#%%----------------------------------------------------------------------------                
import pygmo as pg
class evoML:
    def __init__(self, args, fun, lb, ub):
         self.args = args
         self.obj = fun
         self.lb, self.ub= lb, ub
         
    def fitness(self, x):     
        self.res=self.obj(x,*self.args)
        return [self.res]
    
    def get_bounds(self):
         return (self.lb, self.ub)  
     
    def get_name(self):
         return "evoML"
        
#%%----------------------------------------------------------------------------   
from scipy.optimize import differential_evolution as de
import glob as gl
import pylab as pl
import os

path='./pkl/'
os.system('mkdir  '+path)

basename='host_guest_ml___'


datasets=[]
xls=gl.glob('./data/*.csv')
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
     
#%%----------------------------------------------------------------------------   
pop_size = 50
max_iter=50
n_splits = 5
for run in range(run0, n_runs):
    random_seed=run*10+100
    
    for dataset in datasets:#[:1]:        
        for (target,y_) in zip(dataset['target_names'], dataset['y_train']): 
            dataset_name, X_        = dataset['name'], dataset['X_train']
            n_samples, n_features   = dataset['n_samples'], dataset['n_features']
            task                    = dataset['task']

            print(dataset_name, target, n_samples, n_features,)
            np.random.seed(random_seed)
    
            lb_ann = [0, 0, 1e-6, 1,   1, 1, 1, 1, 1,] #+ [0.0]*n_features          
            ub_ann = [3, 2, 1e-2, 5,  50,50,50,50,50,] #+ [1.0]*n_features

            lb_en = [0, 0, 0, ] #+ [0.0]*n_features          
            ub_en = [2, 1, 1,] #+ [1.0]*n_features

            lb_mlp=[0, 0,     1,  1,  1,  1,  1,  1,] #+ [0.0]*n_features
            ub_mlp=[1, 1,     5, 50, 50, 50, 50, 50,] #+ [1.0]*n_features, "rb"))
    
            lb_gb  = [0.001,  100,  10,  5,  5,   0, 0.1, ] #+ [0.0]*n_features
            ub_gb  = [  0.8,  900, 100, 50, 50, 0.5, 1.0, ] #+ [1.0]*n_features
            
            #lb_svm = [ 1e-0,  1e-5, 0] #+ [0.0]*n_features
            #ub_svm = [ 1e+5,  1e+3, 2] #+ [1.0]*n_features
            lb_svm=[0, 0, -0.1, 1e-6, 1e-6, 1e-6, ]#+ [0.0]*n_features, "rb"))
            ub_svm=[5, 5,    2, 1e+4, 1e+4,    4,]#+ [1.0]*n_features

            lb_knn = [ 1,  0, 1] #+ [0.0]*n_features
            ub_knn = [50,  1, 3] #+ [1.0]*n_features
            
            lb_elm = [1e-0,   0,   0,   0.01, ] #+ [0.0]*n_features
            ub_elm = [5e+2,   5,   1,  10.00, ] #+ [1.0]*n_features
            
            lb_vc  = [ 0]*2 + [ 0,   0,]#+ [0.0]*n_features
            ub_vc  = [ 1]*2 + [ 9, 300,]#+ [1.0]*n_features
      
            lb_bag  = [0,  10, ]#+ [0.0]*n_features
            ub_bag  = [1, 900, ]#+ [1.0]*n_features
      
            lb_dtc  = [0,  2, ]#+ [0.0]*n_features
            ub_dtc  = [1, 20, ]#+ [1.0]*n_features
            
            lb_krr=[0., 0,  0.0, 1,   0,  1e-6]#+ [0.0]*n_features
            ub_krr=[1., 5, 10.0, 5, 1e3,  1e+4]#+ [1.0]*n_features
      
            lb_xgb = [0.0,  10,  1, 0.0,  1, 0.0]#+ [0.0]*n_features
            ub_xgb = [1.0, 900, 30, 1.0, 10, 1.0]#+ [1.0]*n_features

            lb_cat = [0.0,  10,  1,    0.,  1., 0.0]#+ [0.0]*n_features
            ub_cat = [1.0, 900, 16, 1000., 50., 1.0]#+ [1.0]*n_features

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
                #('EN'   , lb_en, ub_en, fun_en_fs, args, random_seed,),
                ('XGB'  , lb_xgb, ub_xgb, fun_xgb_fs, args, random_seed,),
                #('DT'   , lb_dtc, ub_dtc, fun_dtc_fs, args, random_seed,),
                #('VC'   , lb_vc , ub_vc , fun_vc_fs , args, random_seed,),
                #('BAG'  , lb_bag, ub_bag, fun_bag_fs, args, random_seed,),
                #('KNN'  , lb_knn, ub_knn, fun_knn_fs, args, random_seed,),
                ('ANN'  , lb_ann, ub_ann, fun_ann_fs, args, random_seed,),
                ('ELM'  , lb_elm, ub_elm, fun_elm_fs, args, random_seed,),
                ('SVR'  , lb_svm, ub_svm, fun_svm_fs, args, random_seed,),
                #('MLP'  , lb_mlp, ub_mlp, fun_mlp_fs, args, random_seed,),
                #('GB'   , lb_gb , ub_gb , fun_gb_fs , args, random_seed,),            
                #('KRR'  , lb_krr, ub_krr, fun_krr_fs, args, random_seed,),
                #('CAT'  , lb_cat, ub_cat, fun_cat_fs, args, random_seed,),
                ]
    
            for (clf_name, lb, ub, fun, args, random_seed) in optimizers:
                print(clf_name, fun, random_seed)
                np.random.seed(random_seed)

#                init=lhsu(lb,ub,pop_size) # latin hypercube sampling strategy
#                res = de(func=fun, bounds=tuple(zip(lb,ub)), args=args, maxiter=max_iter,
#                         init=init, seed=run, disp=True, polish=False, 
#                         strategy='best1bin', tol=1e-6)
#                
#                xopt, fopt = res['x'], res['fun']
#                sim = fun(xopt, *(X,y,'run',n_splits,random_seed))
#                sim['ALGO'] = 'DE'
                
                #algo = pg.algorithm(pg.de1220(gen = max_iter, seed=random_seed))
                algo = pg.algorithm(pg.de(gen = max_iter, variant = 1, seed=random_seed))
                #algo = pg.algorithm(pg.pso(gen = max_iter, seed=random_seed))
                #algo = pg.algorithm(pg.ihs(gen = max_iter*pop_size, seed=random_seed))
                #algo = pg.algorithm(pg.gwo(gen = max_iter, seed=random_seed))
                #algo = pg.algorithm(pg.sea(gen = max_iter, seed=random_seed))
                #algo = pg.algorithm(pg.sade(gen = max_iter, seed=random_seed))
                #algo = pg.algorithm(pg.sga(gen = max_iter, m=0.10, crossover = "sbx", mutation = "gaussian", seed=random_seed))
                #algo = pg.algorithm(pg.cmaes(gen = max_iter, force_bounds = True, seed=random_seed))
                #algo = pg.algorithm(pg.xnes(gen = max_iter, memory=False, force_bounds = True, seed=random_seed))
                #algo = pg.algorithm(pg.simulated_annealing(Ts=100., Tf=1e-5, n_T_adj = 100, seed=random_seed))

                algo.set_verbosity(1)
                prob = pg.problem(evoML(args, fun, lb, ub))
                pop = pg.population(prob,pop_size, seed=random_seed)
                pop = algo.evolve(pop)
                xopt = pop.champion_x
                sim = fun(xopt, *(X,y,'run',n_splits,random_seed))
                sim['ALGO'] = algo.get_name()

                sim['ACTIVE_VAR_NAMES']=dataset['var_names'][sim['ACTIVE_VAR']]
                if task=='classification':
                    sim['Y_TRAIN_TRUE'] = le.inverse_transform(sim['Y_TRUE'])
                    sim['Y_TRAIN_PRED'] = le.inverse_transform(sim['Y_PRED'])
                else:
                    sim['Y_TRAIN_TRUE'] = sim['Y_TRUE']
                    sim['Y_TRAIN_PRED'] = sim['Y_PRED']

                #pl.figure()#(random_seed+0)
                ##pl.plot(sim['Y_TRAIN_TRUE'].ravel(), 'r-', sim['Y_TRAIN_PRED'].ravel(), 'b-' )
                #pl.axis('equal')
                #pl.plot(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel(), 'r.' )
                #pl.plot(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_TRUE'].ravel(), 'k-' )
                #r2=r2_score(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())
                #mape=MAPE(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())
                #pl.ylabel(damodeltaset_name)
                ##pl.title('(Training) R$^2$='+str(r2)+'\n'+', '.join(sim['ACTIVE_VAR_NAMES']))
                #pl.title('(Training) R$^2$='+fmt(r2)+'\nMAPE='+fmt(mape)+'\n')
                #pl.show()
                ##aux = fun(xopt, *(dataset['X_test'],dataset['y_test'][0].ravel(),'run',n_splits,random_seed))
                ##sim['Y_TEST_TRUE'] = aux['Y_TRUE']
                ##sim['Y_TEST_PRED'] = aux['Y_PRED']

                #pl.figure()#(random_seed+1)
                #pl.plot(sim['Y_TEST_TRUE'].ravel(), 'r-', sim['Y_TEST_PRED'].ravel(), 'b-' )
                #r2=r2_score(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                #pl.ylabel(dataset_name)
                #pl.title('(Test) R$^2$='+str(r2)+'\n'+', '.join(sim['ACTIVE_VAR_NAMES']))
                #pl.show()
                
                sim['RUN']=run; #sim['Y_NAME']=yc
                sim['DATASET_NAME']=dataset_name; 
                list_results.append(sim)
        
                data = pd.DataFrame(list_results)
                pk=(path+'__'+basename+
                            '_run_'+str("{:02d}".format(run))+'_'+dataset_name+'_'+
                            os.uname()[1]+'__'+ str.lower(sim['EST_NAME'])+'__'+
                            target+'__'+
                            #time.strftime("%Y_%m_%d_") + time.strftime("_%Hh_%Mm_%S")+
                            #'_loo'+
                            '.pkl') 
                pk=pk.replace(' ','_').replace("'","").lower()
                data.to_pickle(pk)
                
                pm = pk.replace('.pkl', '.dat')
                pickle.dump(sim['ESTIMATOR'], open(pm, "wb"))
                
#%%----------------------------------------------------------------------------
