#%%
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution as de
import glob as gl
import pylab as pl
import os
from sklearn import preprocessing


path='./pkl/'
os.system('mkdir  '+path)

basename='host_guest_ml___'

datasets=[]
xls=gl.glob('./data/*.csv') # Encontra todos os arquivos .csv na pasta

#%%

for f in xls:
    X=pd.read_csv(f) # Leitura de cada arquivo .csv em xls
       
    cols_to_remove  = ['BindingDB ITC_result_a_b_ab_id'] # colunas para remover da base de dados
    cols_target     = ['Delta_G0 (kJ/mol)'] # colunas com o target

    X.drop(cols_to_remove,  axis=1, inplace=True) # remove as colunas selecionadas anteriormente
    y_train = X[cols_target] # seleciona a coluna de target para y_train   
    X.drop(cols_target,  axis=1, inplace=True) # remove a coluna de target de X 
    X_train = X
    
    y_train.columns=['delta_g0'] # renomeia coluna em y_train
    
    # dataset é um dicionario com as informações retiradas de cada arquivo em X. 
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
n_splits = 5

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
        
        # colocando um valor para cada label existente
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
            ('XGB'   , *get_parameters('XGB'), args, random_seed,),
            #('DTC'   , get_parameters('DTC'), args, random_seed,),
            #('VC'   , get_parameters('VC'), args, random_seed,),
            #('BAG'  , get_parameters('BAG'), args, random_seed,),
            #('KNN'  , get_parameters('KNN'), args, random_seed,),
            #('ANN'  , get_parameters('ANN'), args, random_seed,),
            #('ELM'  , get_parameters('ELM'), args, random_seed,),
            #('SVM'  , get_parameters('SVM'), args, random_seed,),
            #('MLP'  , get_parameters('MLP'), args, random_seed,),
            #('GB'   , get_parameters('GB'), args, random_seed,),      
            #('KRR'  , get_parameters('KRR'), args, random_seed,),
            #('CAT'  , get_parameters('CAT'), args, random_seed,),
            ]

        #for (clf_name, lb, ub, fun, args, random_seed) in optimizers:
        for (clf_name, lb, ub, args, random_seed) in optimizers:
                print(clf_name, random_seed)
                #print(clf_name, fun, random_seed)
                np.random.seed(random_seed)

                algo = pg.algorithm(pg.de(gen = max_iter, variant = 1, seed=random_seed))

                algo.set_verbosity(1)
                '''
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
                '''


#%%

def get_parameters(opt):

    if opt == 'ANN':
        lb = [0, 0, 1e-6, 1,   1, 1, 1, 1, 1,] #+ [0.0]*n_features  
        ub = [3, 2, 1e-2, 5,  50,50,50,50,50,] #+ [1.0]*n_features
        fun = fun_ann_fs

    elif opt == 'EN':
        lb = [0, 0, 0, ] #+ [0.0]*n_features          
        ub = [2, 1, 1,] #+ [1.0]*n_features
        fun = fun_en_fs

    elif opt == 'MPL':
        lb =[0, 0,     1,  1,  1,  1,  1,  1,] #+ [0.0]*n_features
        ub =[1, 1,     5, 50, 50, 50, 50, 50,] #+ [1.0]*n_features, "rb"))
        fun = fun_en_fs

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
        #fun = fun_xgb_fs

    elif opt == 'CAT':
        lb = [0.0,  10,  1,    0.,  1., 0.0]#+ [0.0]*n_features
        ub = [1.0, 900, 16, 1000., 50., 1.0]#+ [1.0]*n_features
        fun = fun_cat_fs

    return lb, ub
    #return lb, ub, fun

# %%
