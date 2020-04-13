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

run_options = ['pre_trat']

reduce_dataset = True
keep_non_trat_dataset = False
save_csv_trat = True

# Possibilidades:
# pre_trat: tratamento e retirada preeliminar de instancias
# exp: analise exploratoria
# out_an: analise de outliers (necessario que a analise exploratoria ocorra)
# clust: clusteriazacao
# mach_learn: machine learn

#%%
### LEITURA DOS DATASETS

## IDENTIFICAÇÃO DE ARQUIVOS PARA LEITURA
datasets = []
xls      = gl.glob('./data/using/*.csv')                                       # Encontra todos os arquivos .csv na pasta

ids = []

for f in xls:
    
    X               = pd.read_csv(f)                                           # Leitura de cada arquivo .csv em xls       
    
    cols_to_ids     = ['BindingDB', 'Host', 'Guest'] 
    cols_to_remove  = ['Delta_H0 (k/mol)', \
                       '-T Delta_S0 (kJ/mol)', 'Ligand SMILES', 'Host SMILES']                                 # colunas para remover da base de dados
    cols_target     = ['Delta_G0 (kJ/mol)']                                    # colunas com o target
    
    X.drop(cols_to_remove,  axis=1, inplace=True)                              # remove as colunas selecionadas anteriormente
    X.dropna(inplace=True)
    
    print(f)
    print(X[cols_target].shape)
    
    X.reset_index(inplace=True, drop=True)
    drop = list(X[['Host', 'Guest']].drop_duplicates().index)
    X = X.iloc[drop,:]
    X.reset_index(inplace=True, drop=True)
#    X.set_index('index')
    
    ids = X[cols_to_ids]
    X.drop(cols_to_ids,  axis=1, inplace=True)
    
    print(X[cols_target].shape)
    
    X['Temp'] = [float(s.split()[0]) for s in X['Temp']]
    
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
    dataset['ids'], dataset['id_cols_name']       = ids, cols_to_ids
    
    datasets.append(dataset)

#%%

### PRÉ-ANÁLISE EXPLORATÓRIA
    ##  DEFINIÇÃO DE CLASSES DE ATRIBUTOS - ATRIBUTOS RELATIVOS AO MEIO | LIGANTE | HOSPEDEIRO

## ATRIBUTOS DO MEIO - ENVIRONMENT     
col_env = ['pH', 'Temp']

## ATRIBUTOS DO LIGANTE - LIGANT
col_lig = ['SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumAtoms', 'Formal Charge']
#col_lig = []
#col_lig = dataset['var_names'].drop(col_host)                              # colunas do ligante: colunas que sobraram do meio e do host 
#col_lig = list(col_lig.drop(col_env))

## ATRIBUTOS DO HOSPEDEIRO - HOST
col_host = [i + ' (#1)' for i in col_lig]
col_host = col_host[:-1]
#col_host = [ i for i in dataset['var_names'] if "host" in str.lower(i)]    # colunas do host: colunas que contem 'host' no nome 
        
opt_sel_col = {'col_env': col_env,
               'col_host': col_host,
               'col_lig': col_lig,
               'all_atr': col_env + col_lig + col_host}    

if reduce_dataset:
    for d in datasets:
        X_aux = pd.DataFrame(d['X_train'], columns=d['var_names']) 
        d['var_names'] = pd.Index(opt_sel_col['all_atr'])
        d['X_train'] = X_aux[opt_sel_col['all_atr']].values
        

#%%

### TRATAMENTO PREELIMINAR DE BASE
    ##  RETIRADA DE INSTÂNCIAS COM BASE EM BAIXA AMOSTRAGEM DE CARACTERÍSTICAS DE MEIO
    
if 'pre_trat' in run_options:
   
    n_datasets = []
    
    ids_trat = []
    
    for dataset in datasets:

        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        Y_ = pd.DataFrame(dataset['y_train'][0], columns=dataset['target_names'])
        
        ext_atr = dataset['id_cols_name']+opt_sel_col['all_atr']+list(dataset['target_names'])
        
        Dt_ = pd.concat([X_, Y_], axis=1)
        Dt_ = pd.concat([dataset['ids'], Dt_], axis=1)
        
        ## LIMITES PROPOSTOS PARA pH
        Dt_  = Dt_[ext_atr][(Dt_['pH'] >= 6.9) & (Dt_['pH'] <= 7.4)]
        
        ## LIMITES PROPOSTOS PARA Temp
        Dt_  = Dt_[ext_atr][(Dt_['Temp'] > 14.5) & (Dt_['Temp'] <  30.1)]      #[24.85; 27]
        
        n                                               = dataset['name'].split('.xlsx')[0].split('/')[-1].split('.')
        name_trat                                       = n[0] + '_trat_env.' + n[-1]
        
        if save_csv_trat:
            Dt_.to_csv(name_trat, index=False)
        
        ids_trat = Dt_[dataset['id_cols_name']]
        Dt_.drop(dataset['id_cols_name'],  axis=1, inplace=True)               # remove as colunas selecionadas anteriormente
        
        ndataset                                        = {} 
    
        ndataset['var_names'], ndataset['target_names'] = dataset['var_names'], dataset['target_names']
        #n                                               = dataset['name'].split('.xlsx')[0].split('/')[-1].split('.')
        ndataset['name']                                = name_trat
        ndataset['X_train'], ndataset['y_train'],       = Dt_[dataset['var_names']].values, [Dt_[dataset['target_names']].values]
        ndataset['n_samples'], ndataset['n_features']   = Dt_[dataset['var_names']].shape
        ndataset['task']                                = 'regression'
        ndataset['ids'], ndataset['id_cols_name']       = ids_trat, dataset['id_cols_name']
        
        n_datasets.append(ndataset)
    
    if keep_non_trat_dataset:
        datasets = datasets + n_datasets
    else:
        datasets = n_datasets
#%%

### ANÁLISE EXPLORATÓRIA
    ##  ANÁLISE DE CORRELAÇÃO EM CADA CLASSE DE ATRIBUTOS
    ##  GRÁFICOS BOXPLOT DA DISTRIBUIÇÃO DOS DADOS DE ENTRADA
    ##  IDENTIFICAÇÃO DE OUTLIERS BASEADA EM QUARTIS
    ##  TESTES DE NORMALIDADE E DISTRIBUIÇÃO DAS VARIÁVEIS
    ##  ANALISE DE CORRELAÇÃO CRUZADA E NÃO LINEAR
    

if 'exp' in run_options:

    for dataset in datasets:
        
        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'], dtype=float)
#        X_[['pH', 'Temp']] = X_[['pH', 'Temp']].astype(float)
        dataset_name = dataset['name'].split('.')[0]
        print("\n\n####################\n\n")
        print(dataset_name)
        
        
        print("## ANALISE DE ATRIBUTOS DO MEIO")
    
        edl.correlacoes(X_, col_env, "env", matrix = True, grafic = True, ext = 'png', save = True, show = False, file_name = dataset_name)
        edl.boxplots(X_, col_env, "env", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_env, "env", file_name = dataset_name)


        print("## ANALISE DE ATRIBUTOS DO HOSPEDEIRO")
        
        edl.correlacoes(X_, col_host, "host", matrix = True, grafic = True, ext = 'png', save = True, show = False, file_name = dataset_name)
        edl.boxplots(X_, col_host, "host", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_host, "host", file_name = dataset_name)


        print("## ANALISE DE ATRIBUTOS DO LIGANTE")
        
        edl.correlacoes(X_, col_lig, "ligante", matrix = True, grafic = True, ext = 'png', save = True, show = False, file_name = dataset_name)   
        edl.boxplots(X_, col_lig, "ligante", complete = False, file_name = dataset_name)
        edl.boxplots(X_, col_lig, "ligante", file_name = dataset_name)
        
        
        print("## ANALISE DE TODOS ATRIBUTOS")
    
        edl.correlacoes(X_, opt_sel_col['all_atr'], "all", matrix = True, \
                        grafic = True, ext = 'png', save = True, \
                        show = False, file_name = dataset_name)   
        
        
        print("## ANALISE DE OUTLIERS")
        
        out_host    = edl.outlier_identifier(X_, col_host)
        out_lig     = edl.outlier_identifier(X_, col_lig)
        out         = edl.outlier_identifier(X_, col_lig+col_host)
        
        df_out, qtd = edl.df_outliers(X_, out, folder_name = dataset_name)


        print("## ANÁLISE DE DISTRIBUIÇÃO DE ATRIBUTOS E HISTOGRAMAS")
        
        # Tratamento visual de outliers - dicionário com chave sendo o nome do atributo e valor sendo tupla com patamar mínimo e maximo
#        val_trat = {'AMW': (0, 90), 'LabuteASA': (0, 1500), 'NumLipinskiHBA': (0, 100), 'NumRotableBonds': (0, 280),
#                    'HOST_SlogP': (-50, 15), 'HOST_SMR': (0, 350), 'TPSA': (0, 1000), 'HOST_AMW': (0, 1600), 'HOST_LabuteASA': (0, 650)}
        
        edl.distribution_hist_outlier_trat(X_, trat=False, folder_name=dataset_name)
#        edl.distribution_hist_outlier_trat(X_, trat=True, val_trat=val_trat, folder_name=dataset_name)
        
        
        print("## ANALISE DE CORRELAÇÃO CRUZADA E NÃO LINEAR")
        
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


if 'clust' in run_options:

    min_dim = 1
    max_dim = 3
    
    for dataset in datasets:

        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = dataset['name'].split('.')[0]
        
        print("\n\n####################\n\n")
        print(dataset_name)
        
        
        for atrs in opt_sel_col:
            
            print("\n## Agrupamentos considerando os atributos:")
            print(atrs)
            
            if atrs == 'col_host': continue
        
            for d in range(min_dim, min(max_dim+1, len(opt_sel_col[atrs]))):
                
                print("\n## Redução de Dimensão")
                print(d)
                
                file_name = dataset_name+'_dim_'+str(d)
                
                print("\n## PCA")
                red_x, results, covm = cll.run_pca(X_, opt_sel_col[atrs], str(atrs), newDim=d, save_txt=True, file_name=file_name, folder_name=dataset_name)
                
                if d==1 or d==2 or d==3:
                    print("\n## CLUSTERING")
                    cll.run_clust(red_x, clustering_names=['DBSCAN', 'KMeans', 'Ward'], file_name=file_name+'_'+atrs, folder_name=dataset_name)
            

#%%

### DEFINIÇÃO DE CONJUNTO DE TREINO E TESTE ENTRE AS BASES DE DADOS A SEREM UTILIZADAS
### DEFINIÇÃO DE HIPERPARÂMETROS LEVES DA EXECUÇÃO DA EVOLUÇÃO DIFERENCIAL PARA TESTE DE MODELOS
                    

if 'mach_learn' in run_options: 
    
    pop_size    = 50                                                               # tamanho da populacao de individuos
    max_iter    = 50                                                               # quantidade maxima de iteracoes do DE 
    n_splits    = 5                                                                # número de divisões da base realizada no k-fold
    
    run0        = 0
    n_runs      = 1
    
    ## MÉTODOS DE APRENDISADO DE MÁQUINA UTILIZADOS
    
    # ml_methods=['KNN']
    # ml_methods  = ['XGB', 'ELM'] 
    # ml_methods = ['KNN', 'DTC', 'EN', 'BAG', 'ELM', 'XGB']
    # 'MLP' eh um problema
    # ml_methods = ['GB', 'SVM', 'KRR']
#    ml_methods = ['EN', 'XGB', 'DTC', 'BAG', 'KNN', 'ANN', 'ELM', 'SVM', 'GB', 'KRR'] 
    ml_methods = ['EN', 'XGB', 'DTC', 'BAG', 'KNN', 'ANN', 'SVM', 'GB', 'KRR'] 

    ## PERCENTUAIS PARA TAMANHOS DE CONJUNTOS DE TESTES: [INICIAL, FINAL, PASSO]
    test_size = [0.1, 0.7, 0.1]
    
    t0 = time.time()

    for dataset in datasets:

        X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])
        dataset_name = dataset['name'].split('.')[0]
        
        print("\n\n\n####################################################################\n\n\n")
        print("DATASET NAME: " + str(dataset_name))
        print("\n\n\n####################################################################\n\n\n")

        if type(dataset['y_train']) == list:
            y_ = dataset['y_train'][0]
        else:
            y_ = dataset['y_train']

        # X_treino,X_teste,y_treino,y_teste=train_test_split(X_, dataset['y_train'], test_size=0.20, random_state=50)

        # dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = mll.train_test_split(X_, y_, test_size=0.20, random_state=50)
        # dataset['n_samples'] = len(dataset['X_train'])
        
        
        for ts in list(mll.np.around(mll.np.arange(*test_size), 3)):
            
            print("\n\n\n####################################################################\n\n\n")
            print("RUNNING OPT WITH TESTE SIZE: " + str(ts))
            print("\n\n\n####################################################################\n\n\n")
            
            ## DEFINIÇÃO DE CONJUNTOS DE TREINO E TESTE
            dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = mll.train_test_split(X_, y_, test_size=ts, random_state=50)
            dataset['n_sample_train'] = len(dataset['X_train'])
        
            ## TREINAMENTO E VALIDAÇÃO DAS MÁQUINAS
            lr = mll.run_DE_optmization_train_ml_methods(dataset, ml_methods, \
                                                         de_run0 = run0, de_runf = n_runs, de_pop_size=pop_size, de_max_iter=max_iter, \
                                                         kf_n_splits=n_splits, \
                                                         save_basename='host_guest_ml___', save_test_size = str(ts))    
            
            ## ESTATÍSTICAS SOBRE OS DADOS DE TESTE
            for res in lr:
                
                res['ERROR_TEST'] = mll.evaluate(res['ESTIMATOR'], res['EST_NAME'], \
                                   dataset['X_test'].to_numpy(), dataset['y_test'], \
                                   metrics = ['RMSE', 'MAPE', 'RRMSE', 'score', 'R2_SCORE'], \
                                   save_test_size = str(ts), dataset_name=dataset_name)
        
                pk = res['name_pickle']
        
                data = pd.DataFrame([res])
                data.to_pickle(pk)
        
        
            data = pd.DataFrame(lr)
            data.to_pickle('RESULTADOS/MACHINE_LEARNING/'+str(dataset_name)+'_all_data_test_size_'+str(ts)+'.pkl')
            
    t1 = time.time()
    
    print(t1 - t0)


# %%




