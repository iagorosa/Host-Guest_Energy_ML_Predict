#%%

import sys
sys.path.append('/home/medina/Documentos/UFJF/PGMC/Ciencia_de_Dados/Host-Guest_Energy_ML_Predict')

# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution as de
import glob as gl
import pylab as pl
import pygmo as pg
import os
from sklearn import preprocessing
from functions import *

import time
from sklearn import cluster, metrics
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from numpy.linalg import norm

import seaborn as sns
import scipy.stats as scs
from statsmodels.stats.diagnostic import lilliefors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA


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

col_env = ['pH', 'Temp (C)'] # colunas do meio 

for dataset in datasets:
    
    X_ = pd.DataFrame(dataset['X_train'], columns=dataset['var_names'])

    col_lig = []
    col_host = [ i for i in dataset['var_names'] if "host" in str.lower(i)] # colunas do host: colunas que contem 'host' no nome 
    
    # colunas do ligante: colunas que sobraram do meio e do host
    col_lig = dataset['var_names'].drop(col_host) 
    col_lig = col_lig.drop(col_env)


#%%
# Colocar no loop depois 

# Informacoes do Host

# correlacoes(X, col_host, "host", matrix = True, grafic = True, ext = 'png', save = True, show = True)
boxplots(X, col_host, "host", complete = False)
boxplots(X, col_host, "host")


# Informacoes do Ligante

# correlacoes(X, col_lig, "ligante", matrix = True, grafic = True, ext = 'png', save = True, show = True)   

boxplots(X, col_lig, "ligante")
boxplots(X, col_lig, "ligante", complete = False)


     
#%%

pop_size = 50 # tamanho da populacao de individuos
max_iter=50   # quantidade maxima de iteracoes do DE 
n_splits = 5  # 

run0 = 0
n_runs = 1

for run in range(run0, n_runs):
    random_seed=run*10+100

    for dataset in datasets:#[:1]:

        # Definicao das variaveis associadas aos datasets
        target, y_              = dataset['target_names'], dataset['y_train']
        dataset_name, X_        = dataset['name'], dataset['X_train']
        n_samples, n_features   = dataset['n_samples'], dataset['n_features']
        task                    = dataset['task']

        # print(dataset_name, target, n_samples, n_features,)
        np.random.seed(random_seed)

        list_results=[]
        print('='*80+'\n'+dataset_name+': '+target+'\n'+'='*80+'\n')
        
        # defindo o target y conforme a task associada
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

        # lista com todos os possiveis algoritmos otmizadores para o DE
        list_opt_name = ['EN', 'XGB', 'DTC', 'VC', 'BAG', 'KNN', 'ANN', 'ELM', 'SVM', 'MLP', 'GB', 'KRR', 'CAT']

        # lista das opcoes de algoritmos selecionados da lista acima
        name_opt = ['XGB', 'ELM']
        optimizers=[      

            (name, *get_parameters(name), args, random_seed) for name in name_opt 
            ]

        for (clf_name, lb, ub, fun, args, random_seed) in optimizers:
                #print(clf_name, random_seed)
                #print(clf_name, fun, random_seed)
                np.random.seed(random_seed)

                algo = pg.algorithm(pg.de(gen = max_iter, variant = 1, seed=random_seed))

                algo.set_verbosity(1)
                prob = pg.problem(evoML(args, fun, lb, ub))
                pop = pg.population(prob,pop_size, seed=random_seed)
                pop = algo.evolve(pop)
                '''
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
        fun = fun_xgb_fs

    elif opt == 'CAT':
        lb = [0.0,  10,  1,    0.,  1., 0.0]#+ [0.0]*n_features
        ub = [1.0, 900, 16, 1000., 50., 1.0]#+ [1.0]*n_features
        fun = fun_cat_fs

    #return lb, ub
    return lb, ub, fun

# %%


def correlacoes(X, atributes, atr_type, matrix = True, grafic = False, ext = 'png', show = False, save = True, scale = 1.4):
    
    # atr_type = 'host' if str.lower(atributes[0][:4]) == 'host' else 'ligante'
    X_ = X[atributes].dropna()

    if atr_type == 'host':
        try: 
            name_atr = [ i[5:] for i in atributes ]
        except:
            name_atr = atributes
    else:
        name_atr = atributes

    X_.columns = name_atr
    
    if grafic == True:
        pl.rcParams.update(pl.rcParamsDefault)

        pl.figure()

        #colocar titulo no grafico
        g = sns.pairplot(X_, vars=name_atr)
        g.fig.suptitle('Pairplot '+str.capitalize(atr_type))
        
        if save == True:
            pl.savefig('./imgs/grap_corr_'+atr_type+'.'+ext, dpi=300)
        if show == True:
            pl.show()
    
    if matrix == True:
        pl.figure(figsize=(20,10))
        sns.set(font_scale=scale)
        sns.heatmap(X_.corr(), xticklabels=name_atr, yticklabels=name_atr, linewidths=.5, annot=True)
        
        locs, labels = pl.xticks()
        pl.setp(labels, rotation=15)
        
        pl.title('Matriz de Correlação '+str.capitalize(atr_type), fontsize=22)
        pl.tight_layout()
        
        if save == True:
            pl.savefig('./imgs/mat_corr_'+atr_type+'.'+ext, dpi=300)
        if show == True:
            pl.show()
    
    pl.close('all')

#%%

def boxplots(X, atributes, atr_type, complete = True):

    # atr_type = 'host' if str.lower(atributes[0][:4]) == 'host' else 'ligante'

    if atr_type == 'host':
        name_atr = [ i[5:] for i in atributes ]
    else:
        name_atr = atributes

    x = X.loc[:, atributes]
    x.columns = name_atr

    if complete == True:
        pl.figure()

        x.boxplot()
        pl.xticks(rotation=15)
        pl.title("Boxplot Atributos do "+str.capitalize(atr_type))
        pl.tight_layout()
        pl.savefig("./imgs/boxplots_"+atr_type+".png", dpi=300)
        
        pl.show()

    else:
        for atr in name_atr:
            pl.figure()
            
            x[atr].plot.box()
            # atr = atr[5:] if str.lower(atr[:4]) == 'host' else atr
            
            pl.title("Boxplot do "+ str.capitalize(atr) + " " +str.capitalize(atr_type))
            pl.grid(axis='y')
            pl.tight_layout()
            pl.savefig("./imgs/"+atr_type+"/boxplot_"+atr+"_"+atr_type+".png", dpi=300)
            
            pl.show()

    
    pl.close('all')

# %%

def outlier_identifier(X, atributes):

    X_ = X.loc[:, atributes].dropna()

    Q1 = X_.quantile(0.25)
    Q3 = X_.quantile(0.75)
    IQR = Q3 - Q1
    
    print(IQR)

    out = (X_ < (Q1 - 1.5 * IQR)) | (X_ > (Q3 + 1.5 * IQR))
    # print(type(out))
    # print( (X_ < (Q1 - 1.5 * IQR)) | (X_ > (Q3 + 1.5 * IQR)) )
    return out

#%%

out_host = outlier_identifier(X, col_host)
out_lig  = outlier_identifier(X, col_lig)
out = outlier_identifier(X, list(col_lig)+col_host)


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
print(X.T[qtd.index].T)

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
print(X.T[qtd.index].T)

# %%

qtd = out.sum(axis=1)[out.sum(axis=1)>0]
df_out = X.T[qtd.index].T

#%%

col_out = np.zeros(len(out))
shape = out.shape
col_arr = np.array(out.columns)

for index in out.shape[0]:
    # aux = 
    col in out.shape[1]:
        if index[col] == True:
            


#%%

col = np.array(out.columns)
res = out.values * col.T
res_ = [list(i[i != '']) for i in res]

outliers_col = [res_[i] for i in list(df_out.index)]
df_out['outlier'] = outliers_col

df_out.to_csv('outliers.csv')

# %%

ini = True
trat = True

val_trat = {'AMW': 4000, 'LabuteASA':1500, 'NumLipinskiHBA': 100, 'NumRotableBonds': 280, 'HOST_SlogP': 15, 'HOST_SMR': 350, 'TPSA': 1000, 'HOST_AMW': 1600, 'HOST_LabuteASA': 650, }

for atr in X.columns[2:]:

    if atr[:4] == 'HOST':
        atr_name = atr[5:]
        atr_type = 'host'
    else:
        atr_name = atr
        atr_type = 'ligante'

    pl.figure()

    if trat == True:
        try:
            Y = X[atr][X[atr] < val_trat[atr]]
            aux = X[atr][X[atr] > val_trat[atr]]
            aux.to_csv("./csv/outliers_"+atr_name+"_"+atr_type+"_"+str(val_trat[atr])+".csv")
        except:
            Y = X[atr]
        trat_tex = '_trat'
    else:
        Y = X[atr]
        trat_tex = ''

    Y.hist(histtype='bar', density=True, ec='black', zorder=2)

    min = int(round(Y.min()-0.5))
    max = int(round(Y.max()+0.5))

    print(min)
    print(max)

    pl.xticks(range(min, max, round((max-min)/10+0.5)))
    
    pl.xlabel(atr)
    pl.ylabel("Frequência")
    
    pl.title("Histograma " + atr_name + " " + str.capitalize(atr_type) )
    pl.grid(axis='x')

    # estatistica
    mu, std = scs.norm.fit(Y)

    print(mu, std)
    print(x)

    # Plot the PDF.
    xmin, xmax = pl.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scs.norm.pdf(x, mu, std)
    pl.plot(x, p, 'r--', linewidth=2)

    # Teste de hipotese de normalidade com 5% de significancia:
    # H0: A amostra provem de uma população normal
    # H1: A amostra nao provem de uma distribuicao normal

    # Testes de shapiro e lillefors: 
    s   = scs.shapiro(Y)
    lil = lilliefors(Y)

    ymin, ymax = pl.ylim()
    pl.text(xmin+xmin*0.01, ymax-ymax*0.12, 'Shapiro: '+str(round(s[1], 5) )+'\nLilliefors: '+str(round(lil[1], 5)), bbox=dict(facecolor='red', alpha=0.4), zorder=4 )

    if ini == True:
        D = pd.DataFrame(Y.describe())
        ini = False
    else:
        D.loc[list(Y.describe().index), atr] = Y.describe()
        
    D.loc['skewness', atr] = scs.skew(Y)
    D.loc['kurtosis', atr] = scs.kurtosis(Y, fisher=False)


    pl.tight_layout()
    pl.savefig("imgs/hists/"+atr_name+"_"+atr_type+trat_tex+".png")
    pl.show()

    pl.close()

D.to_csv('descricao_resumo'+trat_tex+'.csv')
    

# a, b = pl.ylim()


#%%


X_ = X[col_lig] 
poly = PolynomialFeatures(2)
pp = poly.fit_transform(X_)


df_pf_lig = pd.DataFrame(pp)

correlacoes(df_pf_lig, df_pf_lig.columns, atr_type='ligante', matrix=True, grafic=False, show=False, save=True, ext= 'pdf', scale=0.7)

# %%

X_ = X[col_host] 
poly = PolynomialFeatures(2)
pp = poly.fit_transform(X_)


df_pf_host = pd.DataFrame(pp)

correlacoes(df_pf_host, df_pf_host.columns, atr_type='host', matrix=True, grafic=False, show=False, save=True, ext= 'pdf', scale=0.7)


#%%


def run_pca(X, atributes, atr_type, newDim=2, normMethod='MinMax'):
    
    # atr_type = 'host' if str.lower(atributes[0][:4]) == 'host' else 'ligante'
    X_ = X[atributes].dropna()
    
    if normMethod == 'MinMax': 
        # normalize the data in the space of 0 to 1
        for i in atributes: 
            # If column is uniform discard it
            #if np.all(dt[0:i] == dt[:i], axis=1):
            #    dt = np.delete(dt, i, axis=1)
            #    #dt = np.delete(dt, i, axis=2)
            #    continue
                
            if sum(X_[i]) != 0:
                #print("\n\nCOLUNA MM: " + str(i))
                #print("\nDIVISOR DO MINMAX: " + str(abs(dt[:, i]).max()))
                X_[i] = (X_[i] /  abs(X_[i]).max())**2
                
                #print(dt[:, i])
                    
    #print(dt)
    # run PCA for the normalized data
    pca = PCA(n_components=newDim)
    print("\nAntes do PCA\n")  
    print(X_)
    print(atributes)
    X_t = pca.fit(X_).transform(X_)
    print("\nDepois do PCA\n")    
    print(X_t)
    # PCA process results
    results = pca.components_
    print("\nResultados PCA: " + str(results))
    covm = pca.explained_variance_ratio_
    print("\nVariance PCA: " + str(covm))
    
    return X_t, results, covm
 
    
#%%
    
def run_clust(X, clustering_names=['DBSCAN', 'KMeans', 'Ward'], saveFig=True, MAX_CLUSTERS=10, PER_CONNECT=0.5):

        ##########################################  PRE-METHODS DEFINITION ###########################################
    
        # connectivity matrix for structured Ward
        n_neig = int(len(X) * PER_CONNECT)
        connectivity = kneighbors_graph(X, n_neighbors=n_neig, include_self=True)
        
        # make connectivity symmetric
        affinity = 'euclidean'
        connectivity = 0.5 * (connectivity + connectivity.T)
        connectivity, n_components = cluster.hierarchical._fix_connectivity(X, connectivity, affinity)
        
        # define cutoff for DBSCAN
        n_eps = 0.1
    
        ##########################################  METHODS DEFINITION ##############################################
        
        clustering_algorithms = []
        
        if 'DBSCAN' in clustering_names:
            try:
                dbscan = cluster.DBSCAN(eps=n_eps, min_samples = 5,algorithm='kd_tree', metric='euclidean')
                clustering_algorithms.append(dbscan)
            except Exception as e:
                print("Problems were found while running DBSCAN clustering algorithm. \nProblem: " + str(e))
        
        if 'KMeans' in clustering_names:
            try:
                k, _, _ = __best_k_of_clusters('KMeans', X, MAX_CLUSTERS)
                two_means = cluster.KMeans(n_clusters=k, init='k-means++')
                clustering_algorithms.append(two_means)
            except Exception as e:
                print("Problems were found while running KMeans clustering algorithm. \nProblem: " + str(e))
    
        if 'Ward' in clustering_names:
            try:
                k, _, _ = __best_k_of_clusters('Ward', X, MAX_CLUSTERS, connectivity=connectivity)
                ward = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=connectivity)
                clustering_algorithms.append(ward)
            except Exception as e:
                print("Problems were found while running Ward clustering algorithm. \nProblem: " + str(e))
        
    
        ##########################################  CLUSTERS & PLOTS ###############################################
        
        ################################
        # colors used after on the plot
        theColors='bgrcmykbgrcmykbgrcmykbgrcmyk'
        colors = np.array([x for x in theColors])
        colors = np.hstack([colors] * 20)
    
#        len(clustering_names) * 2 + 3, 9.5
        figsize=()
        plt.figure()
        plt.subplots_adjust(left=.05, right=.98, bottom=.1, top=.96,
                            wspace=.2, hspace=.2)
    
        plot_num = 1
        ################################

        #t0 = time.time()
    
        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            algorithm.fit(X)
            
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
            
            # plot
            plt.subplot(1, len(clustering_algorithms), plot_num)
            plt.title(name, size=18)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
            
            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
                
#            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
#                    transform=plt.gca().transAxes, size=15,
#                    horizontalalignment='right')
            plot_num += 1
    
        #t1 = time.time()
        
        if saveFig == True:
            plt.savefig('cluster_dispersion_graph.png')
        
        plt.show()

#%%

def __evaluate_silhouette_score(X, labels):
    '''
    Evaluates the silhouette score with metrics, if not possivle -1e12 (samll value) is thrown.
    
    PARAMETERS
    ----------
    X : ???
        Lacks description.
    labels : ???
        Lacks description.

    RETURNS
    -------
    double
        The silhouette score.
    '''
    
    try:
        return metrics.silhouette_score(X, labels)
    except:
        return -1e12          

def __sse(algorithm, algorithm_name, X):
    '''
    Calculate the Sum of Squares Error (SSE).
    
    PARAMETERS
    ----------
    algorithm : ???
        Lacks description.
    algorithm_name : string
        algorithm name.
    X : ???
        Lacks description.

    RETURNS
    -------
    double
        The Sum of Squares Error (SSE) value.
    '''
    
    data_frame = pd.DataFrame(X)
    algorithm.cluster_medoids_ = []
    
    if algorithm_name != 'KMeans':
        algorithm.cluster_centers_ = []
        
        for lb in set(algorithm.labels_):
            labels = algorithm.labels_         
            algorithm.cluster_centers_.append(data_frame[labels == lb].mean(axis=0).values)
        
    medians_index, _ = metrics.pairwise_distances_argmin_min(algorithm.cluster_centers_, data_frame.values)               
    
    for m in medians_index:
        algorithm.cluster_medoids_.append(np.array([X[m,0], X[m,1]]))
    
    # mean of quadratic sum of errors
    sse = 0
    for lb in set(algorithm.labels_):
           labels = algorithm.labels_
           sse = sse + sum([(np.linalg.norm(d - algorithm.cluster_medoids_[lb]))**2 for d in np.array(data_frame[labels == lb].values)])
          
    sse = sse / len(set(algorithm.labels_))
    
    return sse
    
def __angle_data_elbow(x1, y1, x2, y2):
    '''
    Calculate the angle between two straights defined
    with two consecutive points of Error Sum of Squares (SSE).
    
    PARAMETERS
    ----------
    x1 : ???
        Lacks description.
    y1 : ???
        Lacks description.
    x2 : ???
        Lacks description.
    y2 : ???
        Lacks description.

    RETURNS
    -------
    double
        The angle between two straights.
    '''
    
    u = np.array([x1, y1])
    v = np.array([x2, y2])
    cos = abs(np.dot(u,v)) / (norm(u) * norm(v))
    angle = np.arccos(np.clip(cos, -1, 1))
    return ((angle * 180) / np.pi)

def __best_k_of_clusters(algorithm_name, X, upK, downK=1, connectivity=0, silhouette=False, elbow=False, sil_elbow=True):
    '''
    Blind search function to terminate the best number of clusters
    for the data using a specific clustering method.
    
    PARAMETERS
    ----------
    algorithm_name : string
        The name of the algorithm.
    X : ???
        Lacks description.
    upK : ???
        Lacks description.
    downK : int? (default = 1)
        Lacks description.
    connectivity : int? (default = 0)
        Lacks description.
    silhouette : boolean (default = False)
        Will silhouette will be evaluated?
    elbow : boolean (default = False)
        Will elbow will be evaluated?
    sil_elbow : boolean (default = False)
        Will ??? will be evaluated?
        
    RETURNS
    -------
    double
        The best K value to be used w clusters.
    '''
    
    k_silh = -1 
    silh_max = 0 
    k_elbow = 1
    max_angle = 0
    list_n_k = list()
    list_sse = list()     
    list_silh = list()
    if downK == 1: list_silh.append(0)   
    
    if (upK >= 2) and (sum([silhouette, elbow, sil_elbow]) == 1):
        if(algorithm_name == 'KMeans'):
            algorithm = cluster.KMeans(n_clusters = 1, init='k-means++')
        elif(algorithm_name == 'Ward'):
            algorithm = cluster.AgglomerativeClustering(n_clusters=1, linkage='ward',connectivity=connectivity)
         
        for iterator in range(downK, upK+1):
            
            algorithm.n_clusters = iterator            
            algorithm.fit(X)
            cluster_labels = algorithm.fit_predict(X)
            
            list_n_k.append(iterator)
            
            if silhouette or sil_elbow:
                if (iterator > 1):            
                    silh = __evaluate_silhouette_score(X, cluster_labels) 
                    list_silh.append(silh)
                    
                    if(silh_max < silh):
                        silh_max = silh
                        k_silh = iterator
                        
            if elbow or sil_elbow:
                list_sse.append(__sse(algorithm, algorithm_name, X))
                
                if(iterator > (downK+1)):
                    i_atual = iterator-downK
                    y1 = list_sse[i_atual - 1] - list_sse[i_atual - 2]
                    y2 = list_sse[i_atual] - list_sse[i_atual - 1]
                    a = __angle_data_elbow(1, y1, 1, y2)
                    if(a > max_angle):
                        max_angle = a
                        k_elbow = iterator - 1
        
    else: 
        return ('-', '-', '-')
    
    if silhouette:
        return (k_silh, '-', '-')
    elif elbow:
        return ('-', k_elbow, max_angle)
    elif sil_elbow:
        return (k_silh, k_elbow, max_angle)     
     
#%%

red_x, _, _ = run_pca(X, col_lig, 'lig', newDim=2)

#%%

run_clust(red_x, clustering_names=['DBSCAN', 'KMeans', 'Ward'])

#%%