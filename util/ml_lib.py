#%%
# -*- coding: utf-8 -*- 

## IMPORTES BÁSICOS   
import numpy as np
import pygmo as pg
import pandas as pd

## PRE-PROCESSAMENTO, VALIDAÇÃO E DEFINIÇÃO DE HIPERPARÂMETROS
from sklearn import preprocessing
from scipy.optimize import differential_evolution as de
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split, StratifiedKFold

## IMPORTES DE SISTEMA
import os
import pickle

## LISTA DE REGRESSOERES UTILIZADOS
from ELM import  ELMRegressor, ELMRegressor
from xgboost import  XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import Pool, CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time 

#%%

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


#%%
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



#%%
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
   # print('Começando KFold', flag)
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    #print('Terminando KFold', flag)
    y_p  = cross_val_predict(clf,X, np.ravel(y),cv=cv,n_jobs=1)

    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
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
        return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'XGB', 'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}


# %%

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
    y_p  = cross_val_predict(clf,X, np.ravel(y), cv=cv, n_jobs=1)

    r = RMSE(y_p,y)
    #r = median_absolute_error(y_p,y)    
    r2 = MAPE(y_p,y)
    r3 = RMSE(y_p,y)
    r4 = -r2_score(y_p,y)
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
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

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
    y_p  = cross_val_predict(clf,X, np.ravel(y), cv=cv, n_jobs=1)
    
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
  except:
    # print("entrou except")
    y_p=[None]
    r=1e12

  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
      clf.fit(X[:,ft].squeeze(), y)
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'KNN',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

def fun_en_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  clf = ElasticNet(random_state=random_seed, max_iter=10000) #TODO: max_iter alterado
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
    
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
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
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'EN',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%----------------------------------------------------------------------------   
def fun_dtc_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  n_samples, n_var = X.shape
  clf = DecisionTreeRegressor(random_state=random_seed,)
  #clf = RandomForestRegressor(random_state=random_seed, n_estimators=100)
  p={
    #  'criterion': 'gini' if x[0] < 0.5 else 'entropy', #TODO: gini e entropy nao sao opcoes de criterion, segundo a documentacao
    'criterion': 'mse',
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
    #r =  mean_squared_error(y,y_p)**0.5
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
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
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'DTC',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

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
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y.ravel(), cv=cv, n_jobs=1)
    
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
  except:
    y_p=[None]
    r=1e12
    
  #print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
      clf.fit(X[:,ft].squeeze(), y.ravel())
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'BAG',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%
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
    y_p  = cross_val_predict(clf,X, np.ravel(y), cv=cv, n_jobs=1)
    
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  MAPE(y,y_p)
    #r =  -accuracy_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')
  except:
    # print("except")
    y_p=[None]
    r=1e12

  #print(r,'\t',p,)#'\t',ft)  
  if flag=='eval':
      return r
  else:
      clf.fit(X[:,ft].squeeze(), y)
      return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'ANN',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

#TODO: nao sei qual eh o regressor usado aqui
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
    
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
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
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

def fun_cat_fs(x,*args):
  X, y, flag, n_splits, random_seed = args 
  clf = CatBoostRegressor(random_state=int(random_seed),verbose=0)
  
  n_samples, n_var = X.shape
  #cr ={
  #     0:'reg:linear',
  #     1:'reg:logistic',
  #     2:'binary:logistic',
  #    }
       
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
    
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)    
    #r =  -f1_score(y,y_p,average='weighted')
    #r =  -precision_score(y,y_p)  
    #print(r,p)
  except:
    y_p=[None]
    r=1e12


#   print(r,'\t',p)  
  if flag=='eval':
      return r
  else:
         clf.fit(X[:,ft].squeeze(), y)
         return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 
              'EST_NAME':'CAT','ESTIMATOR':clf, 'ACTIVE_VAR':ft, 
              'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

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
   
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
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
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

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
    
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
    #r =  mean_squared_error(y,y_p)**0.5
    #r =  -accuracy_score(y,y_p)    
    #r =  -precision_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')  

  except:
    y_p=[None]
    r=1e12
  
  #print (r,'\t',p,'\t',ft)  
  #print (r)
  if flag=='eval':
      return r
  else:
         clf.fit(X[:,ft].squeeze(), y)
         return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'SVM',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}

#%%

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
    
    #r =  mean_squared_error(y,y_p)**0.5
    r = RMSE(y_p, y)
    r2 = MAPE(y_p, y)
    r3 = RRMSE(y_p, y)
    r4 = -r2_score(y_p,y)
    #r =  -accuracy_score(y,y_p)    
    #r =  -precision_score(y,y_p)
    #r =  -f1_score(y,y_p,average='weighted')  
    
  except:
    y_p=[None]
    r=1e12
  
#   print (r,'\t',p,'\t',ft)  
  if flag=='eval':
      return r
  else:
       clf.fit(X[:,ft].squeeze(), y)
       return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'KRR',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'ERROR_TRAIN': {'RMSE':r, 'MAPE': r2, 'RRMSE': r3, 'R2_SCORE': r4}}
  

# %%

def get_parameters(opt, flb=[], fub=[]):

    if opt == 'ANN':
        lb = [0, 0, 1e-6, 1,   1, 1, 1, 1, 1,] #+ [0.0]*n_features  
        ub = [3, 2, 1e-2, 5,  50,50,50,50,50,] #+ [1.0]*n_features
        fun = fun_ann_fs

    elif opt == 'EN':
        lb = [0, 0, 0, ] #+ [0.0]*n_features          
        ub = [2, 1, 1,] #+ [1.0]*n_features
        fun = fun_en_fs

    elif opt == 'MLP':
        lb =[0, 0,     1,  1,  1,  1,  1,  1,] #+ [0.0]*n_features
        ub =[1, 1,     5, 50, 50, 50, 50, 50,] #+ [1.0]*n_features, "rb"))
        fun = fun_mlp_fs

    elif opt == 'GB':
        lb  = [0.001,  100,  10,  5,  5,   0, 0.1, ] #+ [0.0]*n_features
        ub  = [  0.8,  900, 100, 50, 50, 0.5, 1.0, ] #+ [1.0]*n_features
        fun = fun_gb_fs
    
    elif opt == 'SVM':
        lb = [0, 0, -0.1, 1e-6, 1e-6, 1e-6, ]#+ [0.0]*n_features, "rb"))
        ub = [5, 5,    2, 1e+4, 1e+4,    4,]#+ [1.0]*n_features
        fun = fun_svm_fs

    elif opt == 'KNN':
        lb = [ 1,  0, 1] #+ [0.0]*n_features
        ub = [50,  1, 3] #+ [1.0]*n_features
        fun = fun_knn_fs
    
    elif opt == 'ELM':
        lb = [1e-0,   0,   0,   0.01, ] #+ [0.0]*n_features
        ub = [5e+2,   5,   1,  10.00, ] #+ [1.0]*n_features
        fun = fun_elm_fs
    
    elif opt == 'VC':
        lb  = [ 0]*2 + [ 0,   0,]#+ [0.0]*n_features
        ub  = [ 1]*2 + [ 9, 300,]#+ [1.0]*n_features
        fun = fun_vc_fs

    elif opt == 'BAG':
        lb  = [0,  10, ]#+ [0.0]*n_features
        ub  = [1, 900, ]#+ [1.0]*n_features
        fun = fun_bag_fs

    elif opt == 'DTC':
        lb  = [0,  2, ]#+ [0.0]*n_features
        ub  = [1, 20, ]#+ [1.0]*n_features
        fun = fun_dtc_fs
    
    elif opt == 'KRR':
        lb = [0., 0,  0.0, 1,   0,  1e-6]#+ [0.0]*n_features
        ub = [1., 5, 10.0, 5, 1e3,  1e+4]#+ [1.0]*n_features
        fun = fun_krr_fs

    elif opt == 'XGB':
        lb = [0.0,  10,  1, 0.0,  1, 0.0]#+ [0.0]*n_features
        ub = [1.0, 900, 30, 1.0, 10, 1.0]#+ [1.0]*n_features
        fun = fun_xgb_fs

    elif opt == 'CAT':
        lb = [0.0,  10,  1,    0.,  1., 0.0]#+ [0.0]*n_features
        ub = [1.0, 900, 16, 1000., 50., 1.0]#+ [1.0]*n_features
        fun = fun_cat_fs

    #return lb, ub

    if flb != []:
        lb = flb

    if fub != []:
        ub = fub

    return lb, ub, fun


#%%

def run_DE_optmization_train_ml_methods(datasets, name_opt, \
                                        de_run0 = 0, de_runf = 1, de_pop_size=50, de_max_iter=50, \
                                        kf_n_splits=5, \
                                            save_basename='host_guest_ml___', save_test_size = '', save_file_erro_train = True):
    '''
    # lista com todos os possiveis algoritmos otmizadores para o DE
    list_opt_name = ['EN', 'XGB', 'DTC', 'VC', 'BAG', 'KNN', 'ANN', 'ELM', 'SVM', 'MLP', 'GB', 'KRR', 'CAT']
    
    de_pop_size: tamanho da populacao de individuos
    de_max_iter: quantidade maxima de iteracoes do DE 
    kf_n_splits: usado no K-fold 

    de_run0:
    de_runf:
    '''

    # save_path = './RESULTADOS/MACHINE_LEARNING/PKL/'

    try:
        os.mkdir('./RESULTADOS/MACHINE_LEARNING/')
    except:
        pass

    # try:
    #     os.mkdir(save_path)
    # except:
    #     pass

    for run in range(de_run0, de_runf):
        
        random_seed=run*10+100
        np.random.seed(random_seed)

        for dataset in datasets:#[:1]:

            # Definicao das variaveis associadas aos datasets
            target, y_              = dataset['target_names'], dataset['y_train']
            dataset_name, X_        = dataset['name'], dataset['X_train']
            n_samples, n_features   = dataset['n_samples'], dataset['n_features']
            task                    = dataset['task']

            list_results_all = []
            
            print('='*80+'\n'+dataset_name+': '+target+'\n'+'='*80+'\n')
            
            # defindo o target y conforme a task associada
            if task=='classification':
                le = preprocessing.LabelEncoder()
                #le=preprocessing.LabelBinarizer()
                le.fit(y_)
                y=le.transform(y_)
            else:
                y=y_.copy() #TODO: precisei pegar o indice 0 para funcionar
                
            X=X_.copy()
            ##scale_X = MinMaxScaler(feature_range=(0.15,0.85)).fit(X_)
            #scale_X = MinMaxScaler(feature_range=(0,1)).fit(X_)
            #X= scale_X.transform(X_)
            ##   scale_y = MinMaxScaler(feature_range=(0.15,0.85)).fit(y_)    
            ##   X,y = scale_X.transform(X_), scale_y.transform(y_)

            args = (X, y, 'eval', kf_n_splits, random_seed)

            # lista das opcoes de algoritmos selecionados da lista acima
            
            optimizers=[      
                (name, *get_parameters(name), args, random_seed) for name in name_opt 
                ]

            for (clf_name, lb, ub, fun, args, random_seed) in optimizers:
                    
                    print()
                    print(clf_name, '%test_size:', save_test_size)
                    print()              
                    
                    PATH = './RESULTADOS/MACHINE_LEARNING/'+str.upper(clf_name)+'/'

                    try:
                        os.mkdir(PATH)
                    except:
                        pass
                    
                    try:
                        os.mkdir(PATH+'CSV_ERROR_TRAIN/')
                    except:
                        pass

                    try:
                        os.mkdir(PATH+'PKL/')
                    except:
                        pass

                    list_results = []
                    
                    #print(clf_name, random_seed)
                    #print(clf_name, fun, random_seed)
                    np.random.seed(random_seed)
                    
                    t0 = time.time()

                    algo = pg.algorithm(pg.de(gen = de_max_iter, variant = 1, seed=random_seed))

                    algo.set_verbosity(1)
                    prob = pg.problem(evoML(args, fun, lb, ub))
                    pop = pg.population(prob, de_pop_size, seed=random_seed)
                    pop = algo.evolve(pop)
                    
                    xopt = pop.champion_x
                    sim = fun(xopt, *(X.to_numpy(),y,'run',kf_n_splits,random_seed)) #TODO: verificar inserção do to_numpy()

                    t1 = time.time()
                    sim['ALGO'] = algo.get_name()

                    sim['ACTIVE_VAR_NAMES']=dataset['var_names'][sim['ACTIVE_VAR']]
                    if task=='classification':
                        sim['Y_TRAIN_TRUE'] = le.inverse_transform(sim['Y_TRUE'])
                        sim['Y_TRAIN_PRED'] = le.inverse_transform(sim['Y_PRED'])
                    else: # TODO: pq isso mds?
                        sim['Y_TRAIN_TRUE'] = sim['Y_TRUE']
                        sim['Y_TRAIN_PRED'] = sim['Y_PRED']

                    sim['RUN']=run #sim['Y_NAME']=yc
                    sim['DATASET_NAME']=dataset_name

                    sim['time'] = t1 - t0

                    pd.Series(sim['ERROR_TRAIN']).to_csv(PATH+"CSV_ERROR_TRAIN/error_train_"+clf_name+"_%test_size_"+save_test_size+".csv", header=False)

                    # Erros no teste #TODO: precisei converter pra numpy pra dar certo. Conferir
                    # erros no teste no evaluate?
                    # mach = sim['ESTIMATOR'] 
                    # y_p = mach.predict(dataset['X_test'].to_numpy())
                    # y_t = dataset['y_test']

                    # r  = RMSE(y_t, y_p)
                    # r2 = MAPE(y_t, y_p)
                    # r3 = RRMSE(y_t, y_p)

                    # sim['ERROR_TEST'] = {'RMSE': r, 'MAPE': r2, 'RRMSE': r3}
    

                    pk=(PATH+'PKL/'+save_basename+
                                '_run_'+str("{:02d}".format(run))+'_'+dataset_name+'_'+
                                os.uname()[1]+'__'+ str.lower(sim['EST_NAME'])+'__'+
                                target+'__%test_size_'+save_test_size+
                                #time.strftime("%Y_%m_%d_") + time.strftime("_%Hh_%Mm_%S")+
                                #'_loo'+
                                '.pkl') 

                    pk=pk[0].replace(' ','_').replace("'","")
                    # pk=pk[0].replace(' ','_').replace("'","").lower() #TODO: precisei pegar o indice 0 para funcionar
                    
                    sim['name_pickle'] = pk
                    
                    
                    list_results.append(sim)
                    list_results_all.append(sim)
            
                    # data = pd.DataFrame(list_results)
                    # data.to_pickle(pk)

                    
                    pm = pk.replace('.pkl', '.dat')
                    pickle.dump(sim['ESTIMATOR'], open(pm, "wb"))
                    

    return list_results_all


#%%

def evaluate(estimator, name_estimator, X_test, y_test, metrics = ['RMSE', 'MAPE', 'RRMSE', 'score'], save_test_size='', save_file_error = True):

    try:
        os.mkdir('./RESULTADOS/MACHINE_LEARNING')
    except:
        pass

    PATH = './RESULTADOS/MACHINE_LEARNING/'+str.upper(name_estimator)+'/CSV_ERROR_TEST/'

    try:
        os.mkdir(PATH)
    except:
        pass

    y_pred = estimator.predict(X_test)
    error_dict = {}
        
    if 'RMSE' in metrics:
        error_dict['RMSE'] = RMSE(y_test, y_pred)
    if 'MAPE' in metrics: 
        error_dict['MAPE'] = MAPE(y_test, y_pred)
    if 'RRMSE' in metrics: 
        error_dict['RRMSE'] = RRMSE(y_test, y_pred) 
    if 'score' in metrics:
        error_dict['score'] = estimator.score(X_test, y_test)
    if 'R2_SCORE' in metrics:
        error_dict['R2_SCORE'] = -r2_score(y_test,y_pred)

    if save_file_error:
        edd = pd.Series(error_dict)
        edd.to_csv(PATH+"error_test_"+name_estimator+"_%test_size_"+save_test_size+".csv", header=False)


    return error_dict


# %%
