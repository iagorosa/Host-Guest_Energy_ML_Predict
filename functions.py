#%%
# -*- coding: utf-8 -*-    
import numpy as np
from xgboost import XGBRegressor
from ELM import ELMRegressor, ELMRegressor

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


# %%