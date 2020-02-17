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

import time

#%%

## CRIAR PASTA DE IMAGENS
try:
    os.mkdir('./imgs')
except:
    pass

try:
    os.mkdir('./RESULTADOS')
except:
    pass

try:
    path='./pkl/'
    os.mkdir(path)
except:
    pass

#%%

# Escolha das celulas que rodarao:

run_options = ['clust']

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

### PRÉ-ANÁLISE EXPLORATÓRIA
    ##  DEFINIÇÃO DE CLASSES DE ATRIBUTOS - ATRIBUTOS RELATIVOS AO MEIO | LIGANTE | HOSPEDEIRO

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
 
#%%

### ANÁLISE EXPLORATÓRIA
    ##  ANÁLISE DE CORRELAÇÃO EM CADA CLASSE DE ATRIBUTOS
    ##  GRÁFICOS BOXPLOT DA DISTRIBUIÇÃO DOS DADOS DE ENTRADA
    ##  IDENTIFICAÇÃO DE OUTLIERS BASEADA EM QUARTIS
    ##  TESTES DE NORMALIDADE E DISTRIBUIÇÃO DAS VARIÁVEIS
    ##  ANALISE DE CORRELAÇÃO CRUZADA E NÃO LINEAR
    

if 'exp' in run_options:

    for dataset in datasets:
        
        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = datasets[0]['name'].split('.')[0]
        
        
        ## ANALISE DE ATRIBUTOS DO MEIO
    
        edl.correlacoes(X_, col_env, "env", matrix = True, grafic = True, ext = 'png', save = True, show = True, file_name = dataset_name)
        edl.boxplots(X_, col_env, "env", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_env, "env", file_name = dataset_name)


        ## ANALISE DE ATRIBUTOS DO HOSPEDEIRO
        
        edl.correlacoes(X_, col_host, "host", matrix = True, grafic = True, ext = 'png', save = True, show = True, file_name = dataset_name)
        edl.boxplots(X_, col_host, "host", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_host, "host", file_name = dataset_name)


        ## ANALISE DE ATRIBUTOS DO LIGANTE
        
        edl.correlacoes(X_, col_lig, "ligante", matrix = True, grafic = True, ext = 'png', save = True, show = True, file_name = dataset_name)   
        edl.boxplots(X_, col_lig, "ligante", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_lig, "ligante", file_name = dataset_name)
        
        
        ## ANALISE DE TODOS ATRIBUTOS
    
        edl.correlacoes(X_, opt_sel_col['all_atr'], "all", matrix = True, \
                        grafic = True, ext = 'png', save = True, \
                        show = False, file_name = dataset_name)   
        
        
        ## ANALISE DE OUTLIERS
        
        out_host    = edl.outlier_identifier(X_, col_host)
        out_lig     = edl.outlier_identifier(X_, col_lig)
        out         = edl.outlier_identifier(X_, col_lig+col_host)
        
        df_out, qtd = edl.df_outliers(X_, out, folder_name = dataset_name)


        ## ANÁLISE DE DISTRIBUIÇÃO DE ATRIBUTOS E HISTOGRAMAS
        
        # Tratamento visual de outliers - dicionário com chave sendo o nome do atributo e valor sendo tupla com patamar mínimo e maximo
        val_trat = {'AMW': (0, 90), 'LabuteASA': (0, 1500), 'NumLipinskiHBA': (0, 100), 'NumRotableBonds': (0, 280),
                    'HOST_SlogP': (-50, 15), 'HOST_SMR': (0, 350), 'TPSA': (0, 1000), 'HOST_AMW': (0, 1600), 'HOST_LabuteASA': (0, 650)}
        
        edl.distribution_hist_outlier_trat(X_, trat=False, folder_name=dataset_name)
        edl.distribution_hist_outlier_trat(X_, trat=True, val_trat=val_trat, folder_name=dataset_name)
        
        
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

### ANÁLISE EXPLORATÓRIA OUTLIERS
    ## RESULTADOS DA ANÁLISE DE OUTLIERS 

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

### ANÁLISE EXPLORATÓRIA AGRUPAMENTO
    ## ESTUDO DO AGRUPAMENTOS DOS DADOS DE ENTRADA 
    ## ANÁLISE DIMENSIONAL E DE ESTRATÉGIA DE AGRUOAMENTO


min_dim = 1
max_dim = 3

if 'clust' in run_options:

    for dataset in datasets:

        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = datasets[0]['name'].split('.')[0]
        
        
        for atrs in opt_sel_col:
        
            for d in range(min_dim, min(max_dim+1, len(opt_sel_col[atrs]))):
                
                file_name = dataset_name+'_dim_'+str(d)
                
                red_x, results, covm = cll.run_pca(X_, opt_sel_col[atrs], str(atrs), newDim=d, save_txt=True, file_name=file_name, folder_name=dataset_name)
                
                if d==3:
                    cll.run_clust(red_x, clustering_names=['DBSCAN', 'KMeans', 'Ward'], file_name=file_name+'_'+atrs, folder_name=dataset_name)
            


#%%

### DEFINIÇÃO DE CONJUNTO DE TREINO E TESTE ENTRE AS BASES DE DADOS A SEREM UTILIZADAS
### DEFINIÇÃO DE HIPERPARÂMETROS LEVES DA EXECUÇÃO DA EVOLUÇÃO DIFERENCIAL PARA TESTE DE MODELOS
                    

if 'mach_learn' in run_options: 
    
    pop_size    = 5                                                               # tamanho da populacao de individuos
    max_iter    = 5                                                               # quantidade maxima de iteracoes do DE 
    n_splits    = 2                                                                # número de divisões da base realizada no k-fold
    
    run0        = 0
    n_runs      = 1
    
    ## MÉTODOS DE APRENDISADO DE MÁQUINA UTILIZADOS
    
    # ml_methods  = ['XGB', 'ELM'] 
    # ml_methods = ['KNN', 'DTC', 'EN', 'BAG', 'ELM', 'XGB']
    # 'MLP' eh um problema
    # ml_methods = ['GB', 'SVM', 'KRR']
    ml_methods = ['EN', 'XGB', 'DTC', 'BAG', 'KNN', 'ANN', 'ELM', 'SVM', 'GB', 'KRR'] 


    ## PERCENTUAIS PARA TAMANHOS DE CONJUNTOS DE TESTES TESTADOS: [INICIAL, FINAL, PASSO]
    test_size = [0.1, 0.7, 0.1]
    
    t0 = time.time()

    for dataset in datasets:

        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = datasets[0]['name'].split('.')[0]

        if type(dataset['y_train']) == list:
            y_ = dataset['y_train'][0]
        else:
            y_ = dataset['y_train']

        # X_treino,X_teste,y_treino,y_teste=train_test_split(X_, dataset['y_train'], test_size=0.20, random_state=50)

        # dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = mll.train_test_split(X_, y_, test_size=0.20, random_state=50)
        # dataset['n_samples'] = len(dataset['X_train'])
        
        
        for ts in mll.np.arange(*test_size):
            
            ## DEFINIÇÃO DE CONJUNTOS DE TREINO E TESTE
            dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = mll.train_test_split(X_, y_, test_size=ts, random_state=50)
            dataset['n_sample_train'] = len(dataset['X_train'])
        
            ## TREINAMENTO E VALIDAÇÃO DAS MÁQUINAS
            lr = mll.run_DE_optmization_train_ml_methods(datasets, ml_methods, \
                                                         de_run0 = run0, de_runf = n_runs, de_pop_size=pop_size, de_max_iter=max_iter, \
                                                         kf_n_splits=n_splits, \
                                                         save_basename='host_guest_ml___', save_test_size = str(ts))    
            
            ## ESTATÍSTICAS SOBRE OS DADOS DE TESTE
            for res in lr:
                
                res['ERROR_TEST'] = mll.evaluate(res['ESTIMATOR'], res['EST_NAME'], dataset['X_test'].to_numpy(), dataset['y_test'], metrics = ['RMSE', 'MAPE', 'RRMSE', 'score', 'R2_SCORE'], save_test_size = str(ts))
        
                pk = res['name_pickle']
        
                data = pd.DataFrame([res])
                data.to_pickle(pk)
        
        
            data = pd.DataFrame(lr)
            data.to_pickle('RESULTADOS/MACHINE_LEARNING/all_data_test_size_'+str(ts)+'.pkl')
            
    t1 = time.time()
    
    print(t1 - t0)


# %%




