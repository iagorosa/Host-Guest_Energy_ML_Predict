#%%

import sys
#sys.path.append('/home/medina/Documentos/UFJF/PGMC/Ciencia_de_Dados/Host-Guest_Energy_ML_Predict')

# -*- coding: utf-8 -*-    

from scipy.optimize import differential_evolution as de
import glob as gl


from ml_lib import *
from exp_data import *

import time

from clust_lib import *

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

col_env = ['pH', 'Temp (C)'] # colunas do meio 

for dataset in datasets:
    
    X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])

    col_lig = []
    col_host = [ i for i in dataset['var_names'] if "host" in str.lower(i)] # colunas do host: colunas que contem 'host' no nome 
    
    # colunas do ligante: colunas que sobraram do meio e do host
    col_lig = dataset['var_names'].drop(col_host) 
    col_lig = col_lig.drop(col_env)

#TODO: resolver sobre nome dos dataset na hora de salvar uma imaem/arquivo. Alem disso, X_ esta associado a somente um dataset por vez,

#%%
# Colocar no loop depois 

# Informacoes do Host

# correlacoes(X_, col_host, "host", matrix = True, grafic = True, ext = 'png', save = True, show = True)
boxplots(X_, col_host, "host", complete = False)
boxplots(X_, col_host, "host")


# Informacoes do Ligante

# correlacoes(X_, col_lig, "ligante", matrix = True, grafic = True, ext = 'png', save = True, show = True)   

boxplots(X_, col_lig, "ligante")
boxplots(X_, col_lig, "ligante", complete = False)


pl.close('all')
#%%

pop_size = 50 # tamanho da populacao de individuos
max_iter=50   # quantidade maxima de iteracoes do DE 
n_splits = 5  # 

run0 = 0
n_runs = 1

name_opt = ['XGB', 'ELM']

lr = regressions(datasets, name_opt)


#%%

out_host = outlier_identifier(X_, col_host)
out_lig  = outlier_identifier(X_, col_lig)
out = outlier_identifier(X_, list(col_lig)+col_host)

df_out, qtd = df_outliers(out)


# %%

# identifcacao da quantidade de outliers do host
print('OUTLIERS:\nHOST')
print(out_host.sum())
print()
print(out_host.sum(axis=1))
print()
qtd=out_host.sum(axis=1)[out_host.sum(axis=1) == 8]
print('Qtd instancia com todos os valores outliers: ', len(qtd))
print()
print(X_.T[qtd.index].T)

print()
print()

print('\nLIGANTE')
print(out_lig.sum())
print()
print(out_lig.sum(axis=1))
print()
qtd=out_lig.sum(axis=1)[out_lig.sum(axis=1) == 8]
print('Qtd instancia com todos os valores outliers: ', len(qtd))
print()
print(X_.T[qtd.index].T)


#%%

'''
col_out = np.zeros(len(out))
shape = out.shape
col_arr = np.array(out.columns)

for index in out.shape[0]:
    # aux = 
    col in out.shape[1]:
        if index[col] == True:
'''         

# %%


df_pf_lig = polynomial_features(X_, col_lig, 2)
df_pf_host = polynomial_features(X_, col_host, 2)

correlacoes(df_pf_lig, df_pf_lig.columns, atr_type='ligante', matrix=True, grafic=False, show=False, save=True, ext= 'pdf', extra_name = '_polynomial_features', scale=0.7)

correlacoes(df_pf_host, df_pf_host.columns, atr_type='host', matrix=True, grafic=False, show=False, save=True, ext= 'pdf', extra_name = '_polynomial_features',scale=0.7)
     
#%%

red_x, _, _ = run_pca(X, col_lig, 'lig', newDim=2)

#%%

run_clust(red_x, clustering_names=['DBSCAN', 'KMeans', 'Ward'])

#%%