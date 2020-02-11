#%%
# -*- coding: utf-8 -*-    

## IMPORT DE BIBLIOTECAS PARA FUNÇÕES GLOBAIS
import glob            as gl
import pandas          as pd
import time            as tm

## ADICIONAR PASTA DE BIBLIOTECAS CRIADAS NA VARIÁVEL DE CAMINHOS RECONHECIDOS NA EXECUÇÃO
import sys
#sys.path.append('/home/medina/Documentos/UFJF/PGMC/Ciencia_de_Dados/Host-Guest_Energy_ML_Predict')
#sys.path.append('/home/medina/Documentos/UFJF/PGMC/Ciencia_de_Dados/Host-Guest_Energy_ML_Predict/util')

import os
sys.path.append(os.getcwd()+'/util')

## IMPORT DE BIBLIOTECAS CRIADAS
import   exp_dt_lib    as edl
import   clust_lib     as cll
import   ml_lib        as mll


#%%

## CRIAR PASTA DE IMAGENS
try:
    os.mkdir('./imgs')
except:
    pass

try:
    path='./pkl/'
    os.mkdir(path)
except:
    pass

#%%

# Escolha das celulas que rodarao:

run_options = ['mach_learn']

# Possibilidades:
# exp: analise exploratoria
# out_an: analise de outliers (necessario que a analise exploratoria ocorra)
# clust: clusteriazacao
# mach_learn: machine learn

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

#%%

### ANÁLISE EXPLORATÓRIA
    ##  DEFINIÇÃO DE CLASSES DE ATRIBUTOS - ATRIBUTOS RELATIVOS AO MEIO | LIGANTE | HOSPEDEIRO
    ##  ANÁLISE DE CORRELAÇÃO ENTRE CLASSES DE ATRIBUTOS
    ##  GRÁFICOS BOXPLOT DOS DADOS
    
## ATRIBUTOS DO MEIO - ENVIRONMENT     
col_env = ['pH', 'Temp (C)']

if 'exp' in run_options:

    for dataset in datasets:
        
        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = datasets[0]['name'].split('.')[0]
        
        ## ATRIBUTOS DO HOSPEDEIRO - HOST 
        col_lig = []
        col_host = [ i for i in dataset['var_names'] if "host" in str.lower(i)]    # colunas do host: colunas que contem 'host' no nome 
        
        
        ## ATRIBUTOS DO LIGANTE - LIGANT
        col_lig = dataset['var_names'].drop(col_host)                              # colunas do ligante: colunas que sobraram do meio e do host 
        col_lig = col_lig.drop(col_env)


        ## ANALISE DE ATRIBUTOS DO HOSPEDEIRO
        
        edl.correlacoes(X_, col_host, "host", matrix = True, grafic = True, ext = 'png', save = True, show = True, file_name = dataset_name)
        edl.boxplots(X_, col_host, "host", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_host, "host", file_name = dataset_name)


        ## ANALISE DE ATRIBUTOS DO LIGANTE
        
        edl.correlacoes(X_, col_lig, "ligante", matrix = True, grafic = True, ext = 'png', save = True, show = True, file_name = dataset_name)   
        edl.boxplots(X_, col_lig, "ligante", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_lig, "ligante", file_name = dataset_name)
        
        
        ## ANALISE DE OUTLIERS
        
        out_host    = edl.outlier_identifier(X_, col_host)
        out_lig     = edl.outlier_identifier(X_, col_lig)
        out         = edl.outlier_identifier(X_, list(col_lig)+col_host)
        
        df_out, qtd = edl.df_outliers(X_, out, folder_name = dataset_name)

        ## ANÁLISE DE DISTRIBUIÇÃO DE ATRIBUTOS E HISTOGRAMAS
    
        edl.histogramas(X_, ini=True, trat=False, folder_name=dataset_name)
        edl.histogramas(X_, ini=True, trat=True, folder_name=dataset_name)
        
        
        ## ANALISE DE CORRELAÇÃO CRUZADA E NÃO LINEAR
        
        
        df_pf_lig  = edl.polynomial_features(X_, col_lig, 2)
        df_pf_host = edl.polynomial_features(X_, col_host, 2)
        
        edl.correlacoes(df_pf_lig, df_pf_lig.columns, atr_type='ligante', matrix=True, \
                        grafic=False, show=False, save=True, ext= 'pdf', extra_name = '_polynomial_features', \
                        scale=0.7, file_name = dataset_name)
        
        edl.correlacoes(df_pf_host, df_pf_host.columns, atr_type='host', matrix=True, \
                        grafic=False, show=False, save=True, ext= 'pdf', extra_name = '_polynomial_features', \
                        scale=0.7, file_name = dataset_name)
            



#%%

### RESULTADOS DA ANÁLISE DE OUTLIERS 

if 'exp' in run_options and 'out_an' in run_options:
    
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

min_dim = 1
max_dim = 3

if 'clust' in run_options:

    for dataset in datasets:

        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = datasets[0]['name'].split('.')[0]
        
        ## ATRIBUTOS DO MEIO - ENVIRONMENT     
        col_env = ['pH', 'Temp (C)']
        
        ## ATRIBUTOS DO HOSPEDEIRO - HOST 
        col_host = [ i for i in dataset['var_names'] if "host" in str.lower(i)]    # colunas do host: colunas que contem 'host' no nome 
        
        ## ATRIBUTOS DO LIGANTE - LIGANT
        col_lig = []
        col_lig = dataset['var_names'].drop(col_host)                              # colunas do ligante: colunas que sobraram do meio e do host 
        col_lig = list(col_lig.drop(col_env))
        
        opt_sel_col = {'col_env': col_env,
                'col_host': col_host,
                'col_lig': col_lig,
                'all_atr': col_env + col_lig + col_host}
        
        for atrs in opt_sel_col:
        
            for d in range(min_dim, min(max_dim+1, len(opt_sel_col[atrs]))):
                
                file_name = dataset_name+'_dim_'+str(d)
                
                red_x, results, covm = cll.run_pca(X_, opt_sel_col[atrs], str(atrs), newDim=d, save_txt=True, file_name=file_name, folder_name=dataset_name)
                
                if d==2:
                    cll.run_clust(red_x, clustering_names=['DBSCAN', 'KMeans', 'Ward'], file_name=file_name+'_'+atrs, folder_name=dataset_name)
            


#%%

### DEFINIÇÃO DE CONJUNTO DE TREINO E TESTE ENTRE AS BASES DE DADOS A SEREM UTILIZADAS
#\TODO ALL OF IT

if 'mach_learn' in run_options: 

    for dataset in datasets:

        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = datasets[0]['name'].split('.')[0]

        if type(dataset['y_train']) == list:
            y_ = dataset['y_train'][0]
        else:
            y_ = dataset['y_train']
        
        ## ATRIBUTOS DO MEIO - ENVIRONMENT     
        col_env = ['pH', 'Temp (C)']
        
        ## ATRIBUTOS DO HOSPEDEIRO - HOST 
        col_host = [ i for i in dataset['var_names'] if "host" in str.lower(i)]    # colunas do host: colunas que contem 'host' no nome 
        
        ## ATRIBUTOS DO LIGANTE - LIGANT
        col_lig = []
        col_lig = dataset['var_names'].drop(col_host)                              # colunas do ligante: colunas que sobraram do meio e do host 
        col_lig = list(col_lig.drop(col_env))

        # X_treino,X_teste,y_treino,y_teste=train_test_split(X_, dataset['y_train'], test_size=0.20, random_state=50)

        dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = mll.train_test_split(X_, y_, test_size=0.20, random_state=50)
        # dataset['n_samples'] = len(dataset['X_train'])


                    
    ### DEFINIÇÃO DE HIPERPARÂMETROS LEVES DA EXECUÇÃO DA EVOLUÇÃO DIFERENCIAL PARA TESTE DE MODELOS

        pop_size    = 50                                                               # tamanho da populacao de individuos
        max_iter    = 50                                                               # quantidade maxima de iteracoes do DE 
        n_splits    = 5                                                                # número de divisões da base realizada no k-fold
        
        run0        = 0
        n_runs      = 1
        
        # ml_methods  = ['XGB', 'ELM'] 
        ml_methods = ['KNN', 'DTC', 'EN', 'BAG', 'ELM', 'XGB']                                                  # métodos de aprendisado de máquina utilizados
        
        test_size = [0.1, 0.7, 0.1]
        
        
        for ts in mll.np.arange(*test_size):
        
            dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = mll.train_test_split(X_, y_, test_size=ts, random_state=50)
            dataset['n_sample_train'] = len(dataset['X_train'])
        
        
            lr          = mll.run_DE_optmization_train_ml_methods(datasets, ml_methods, \
                                                            de_run0 = 0, de_runf = 1, de_pop_size=50, de_max_iter=50, \
                                                            kf_n_splits=5, \
                                                            save_path='./pkl/', save_basename='host_guest_ml___', save_test_size = str(ts))    
            for res in lr:
                
                res['ERROR_TEST'] = mll.evaluate(res['ESTIMATOR'], res['EST_NAME'], dataset['X_test'].to_numpy(), dataset['y_test'], metrics = ['RMSE', 'MAPE', 'RRMSE', 'score'])
        
                pk = res['name_pickle']
        
                data = pd.DataFrame([res])
                data.to_pickle(pk)
        
        
            data = pd.DataFrame(lr)
            data.to_pickle('all_data_test_size_'+str(ts)+'.pkl')

# porra toda



# %%




