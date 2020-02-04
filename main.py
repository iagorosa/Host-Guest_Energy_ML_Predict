#%%
# -*- coding: utf-8 -*-    

## IMPORT DE BIBLIOTECAS PARA FUNÇÕES GLOBAIS
import glob        as gl
import pandas      as pd
import time        as tm

## ADICIONAR PASTA DE BIBLIOTECAS CRIADAS NA VARIÁVEL DE CAMINHOS RECONHECIDOS NA EXECUÇÃO
import sys
sys.path.append('/home/medina/Documentos/UFJF/PGMC/Ciencia_de_Dados/Host-Guest_Energy_ML_Predict')

## IMPORT DE BIBLIOTECAS CRIADAS
import   ml_lib      as mll
import   exp_data    as edl
import   clust_lib   as cll

## CRIAR PASTA DE IMAGENS
import os
try:
    os.mkdir('./imgs')
except:
    pass
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
    edl.boxplots(X_, col_lig, "ligante", file_name = dataset_name)
    edl.boxplots(X_, col_lig, "ligante", complete = False, file_name = dataset_name)
    
    
    ## ANALISE DE OUTLIERS
    
    out_host    = edl.outlier_identifier(X_, col_host)
    out_lig     = edl.outlier_identifier(X_, col_lig)
    out         = edl.outlier_identifier(X_, list(col_lig)+col_host)
    
    df_out, qtd = edl.df_outliers(X_, out, folder_name = dataset_name)
    
    
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

### DEFINIÇÃO DE HIPERPARÂMETROS LEVES DA EXECUÇÃO DA EVOLUÇÃO DIFERENCIAL PARA TESTE DE MODELOS

pop_size    = 50                                                               # tamanho da populacao de individuos
max_iter    = 50                                                               # quantidade maxima de iteracoes do DE 
n_splits    = 5                                                                # 

run0        = 0
n_runs      = 1

ml_methods  = ['XGB', 'ELM']                                                   # métodos de aprendisado de máquina utilizados

lr          = mll.regressions(datasets, ml_methods)                            # porra toda


#%%

red_x, _, _ = run_pca(X, col_lig, 'lig', newDim=2)

#%%

run_clust(red_x, clustering_names=['DBSCAN', 'KMeans', 'Ward'])

#%%