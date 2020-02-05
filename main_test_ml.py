#%%

## IMPORTES BÁSICOS   
import numpy as np
import pygmo as pg
import pandas as pd
import glob as gl

## PRE-PROCESSAMENTO, VALIDAÇÃO E DEFINIÇÃO DE HIPERPARÂMETROS
from sklearn import preprocessing
from scipy.optimize import differential_evolution as de
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split, StratifiedKFold

## IMPORTES DE SISTEMA
import os
import sys
sys.path.append(os.getcwd()+'/util')
import pickle

## LISTA DE REGRESSOERES UTILIZADOS
from ELM import  ELMRegressor, ELMRegressor
from xgboost import  XGBRegressor

import   exp_dt_lib    as edl
import   clust_lib     as cll
import   ml_lib        as mll

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

### LEITURA DOS DATASETS

## IDENTIFICAÇÃO DE ARQUIVOS PARA LEITURA
datasets = []
xls      = gl.glob('./data/*.csv')                                             # Encontra todos os arquivos .csv na pasta

for f in xls:
    
    X               = pd.read_csv(f)                                           # Leitura de cada arquivo .csv em xls   
    
    cols_to_remove  = ['BindingDB ITC_result_a_b_ab_id']                       # colunas para remover da base de dados
    cols_target     = ['Delta_G0 (kJ/mol)']                                    # colunas com o target

    X.drop(cols_to_remove,  axis=1, inplace=True)                              # remove as colunas selecionadas anteriormente
    y_train         = X[cols_target]                                           # seleciona a coluna de target para y_train   
    X.drop(cols_target,  axis=1, inplace=True)                                 # remove a coluna de target de X 
    X_train         = X
    
    y_train.columns=['delta_g0']                                               # renomeia coluna em y_train
    
    
    ## CRIAÇÃO DE DICIONÁRIO - dataset - COM AS INFORMAÇÃOES RETIRADAS DE CADA ARQUIVO - X
    dataset                                       = {} 
    
    dataset['var_names'], dataset['target_names'] = X_train.columns, y_train.columns
    dataset['name']                               = f.split('.xlsx')[0].split('/')[-1]
    dataset['X_train'], dataset['y_train'],       = X_train.values, [y_train.values]
    dataset['n_samples'], dataset['n_features']   = X_train.shape
    dataset['task']                               = 'regression'
    
    datasets.append(dataset)

dataset = datasets[0]

#%%

de_pop_size=50
de_max_iter=50
kf_n_splits=5 
save_path='./pkl/'
save_basename='host_guest_ml___' 
name_opt  = ['XGB']
run = 1

random_seed=run*10+100
np.random.seed(random_seed)

# Definicao das variaveis associadas aos datasets
target, y_              = dataset['target_names'], dataset['y_train']
dataset_name, X_        = dataset['name'], dataset['X_train']
n_samples, n_features   = dataset['n_samples'], dataset['n_features']
task                    = dataset['task']

list_results = []

print('='*80+'\n'+dataset_name+': '+target+'\n'+'='*80+'\n')

# defindo o target y conforme a task associada
if task=='classification':
    le = preprocessing.LabelEncoder()
    #le=preprocessing.LabelBinarizer()
    le.fit(y_)
    y=le.transform(y_)
else:
    y=y_.copy()[0] #TODO: precisei pegar o indice 0 para funcionar

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
        #print(clf_name, random_seed)
        #print(clf_name, fun, random_seed)
        np.random.seed(random_seed)

        algo = pg.algorithm(pg.de(gen = de_max_iter, variant = 1, seed=random_seed))

        algo.set_verbosity(1)
        prob = pg.problem(evoML(args, fun, lb, ub))
        pop = pg.population(prob, de_pop_size, seed=random_seed)
        pop = algo.evolve(pop)
        
        xopt = pop.champion_x
        sim = fun(xopt, *(X,y,'run',kf_n_splits,random_seed))
        sim['ALGO'] = algo.get_name()

        sim['ACTIVE_VAR_NAMES']=dataset['var_names'][sim['ACTIVE_VAR']]
        if task=='classification':
            sim['Y_TRAIN_TRUE'] = le.inverse_transform(sim['Y_TRUE'])
            sim['Y_TRAIN_PRED'] = le.inverse_transform(sim['Y_PRED'])
        else:
            sim['Y_TRAIN_TRUE'] = sim['Y_TRUE']
            sim['Y_TRAIN_PRED'] = sim['Y_PRED']

        sim['RUN']=run; #sim['Y_NAME']=yc
        sim['DATASET_NAME']=dataset_name; 
        list_results.append(sim)

        data = pd.DataFrame(list_results)
        pk=(save_path+'__'+save_basename+
                    '_run_'+str("{:02d}".format(run))+'_'+dataset_name+'_'+
                    os.uname()[1]+'__'+ str.lower(sim['EST_NAME'])+'__'+
                    target+'__'+
                    #time.strftime("%Y_%m_%d_") + time.strftime("_%Hh_%Mm_%S")+
                    #'_loo'+
                    '.pkl') 

        pk=pk[0].replace(' ','_').replace("'","").lower() #TODO: precisei pegar o indice 0 para funcionar
        data.to_pickle(pk)
        
        pm = pk.replace('.pkl', '.dat')
        pickle.dump(sim['ESTIMATOR'], open(pm, "wb"))

# %%
