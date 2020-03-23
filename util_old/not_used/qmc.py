#!/usr/bin/env python3

'''
Quality-Models_Clusterizer.py (ou o nome que tiver)
    Evaluates proteins based on molpdf, DOPE, DOPEHR, GA341, NDOPE scores
    from Modeller [1], an homology or comparative modeling of protein three-dimensional
    structures software, and allowed and outlier data from Molprobity [2] a
    structure-validation web-service/software.
    
    [1] - A. Fiser, R.K. Do, and A. Sali. Modeling of loops in protein structures, Protein Science 9. 1753-1773, 2000.
    [2] - V.B. Chen, W.B. Arendall, III, J.J. Headd, D.A. Keedy, R.M. Immormino, G.J. Kapral, L.W. Murray,
    J.S. Richardson, and D.C. Richardsona. MolProbity: all-atom structure validation for macromolecular crystallography
    Acta Crystallogr D Biol Crystallogr. 2010 Jan 1; 66(Pt 1):12-21.
    '''

# Deletar essas duas linhas (uso pra rodar no meu bash do windows haeuhaoeui)
import matplotlib
matplotlib.use('Agg')

# Pode inverter os nomes se quiser, coloquei em ordem alfabetica so
__author__ = 'Ruan Medina, Artur Rossi'
__author_email__ = 'ruan.medina@engenharia.ufjf.br, arturossi10@gmail.com'
__version__ = '0.5'
__all__ = ['qmc']

### IMPORTS ###

import re
import os
import time
import argparse
import logging as log

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

#from pdfgrep import do_grep

# command line to create and move .pdf files (MolProbity Ramachandran analysis) to the correct folder
# mkdir Modelos && mv *.pdf ./Modelos

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
        elif(algorithm_name == 'SpectralClustering'):
            algorithm = cluster.SpectralClustering(n_clusters=1, eigen_solver=None, 
												   random_state=None, n_init=10, gamma=1., 
												   affinity='rbf', n_neighbors=10, eigen_tol=0.0, 
												   degree=3, coef0=1, kernel_params=None)
         
        for iterator in range(downK, upK+1):
            
            algorithm.n_clusters = iterator            
            algorithm.fit(X)
            cluster_labels = algorithm.fit_predict(X)
            
            list_n_k.append(iterator)
            
            if silhouette or sil_elbow:
                if (iterator > 1):            
                    silh = __evaluate_silhouette_score(X, cluster_labels) 
                    list_silh.append(silh)
                    log.info('\t k=' + str(iterator) + '  Sil value: ' + str(silh))
                    
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
                    log.info('\t\t\t\t\t k = ' + str(iterator) + ' --- angle: ' + str(a))
                    if(a > max_angle):
                        max_angle = a
                        k_elbow = iterator - 1
        
    else: 
        log.warning('Invalid PARAMETERS')
        return ('-', '-', '-')
    
    if silhouette:
        return (k_silh, '-', '-')
    elif elbow:
        return ('-', k_elbow, max_angle)
    elif sil_elbow:
        return (k_silh, k_elbow, max_angle)   
    
def __medoid_name(X, pdb_names, columns, values):
    '''
    Auxiliary function to find the medoids name.
    
    PARAMETERS
    ----------
    X : ???
        Lacks description.
    pdb_names : string
        Names of pdbs.
    columns : ???
        Lacks description.
    values : ???
        Lacks description.
        
    RETURNS
    -------
    List[String]
        List of pdb names.
        
    '''
    
    c1, c2 = columns
    x, y = values
    m = np.array(X)
    i = int(np.argwhere(np.logical_and(m[:, c1] == x, m[:, c2] == y))[0])
    return pdb_names[i]

def __aprx_medoid(X, center):
    '''
    Find the closest model from the center of the cluster.
    
    PARAMETERS
    ----------
    X : ???
        Lacks description.
    center : ???
        Lacks description.
        
    RETURNS
    -------
    List[String]
        List of pdb names.
        
    '''
    
    d_min = np.sqrt((X[0,0] - center[0])**2 + (X[0,1] - center[1])**2)
    x_ = X[0,:]
    
    for x in X:
        d = np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        
        if d < d_min:
            d_min = d
            x_ = x
            
    return x_

def __innerQmc(outPath='./', path='Models/', FILE_DOT_OUT='analysis.out', CSV_NAME='model.csv', MAX_CLUSTERS=10, PER_CONNECT=0.5, SIL_DAMPING=0.1,
               NORM_METHOD='StandardScaler', clustering_names=['AffinityPropagation', 'DBSCAN', 'KMeans', 'MeanShift', 'SpectralClustering', 'Ward'],
               modellerScores = ['molpdf', 'DOPE', 'DOPEHR', 'GA341', 'NDOPE'], molprobityScores=['outlier', 'allowed'],
               theColors='bgrcmykbgrcmykbgrcmykbgrcmyk', saveFig=False, molprobity=False):
    '''
    The Quality-Models Clusterizer private method, its performs the analysis,
    call the pther methods and evaluate the dataset.
    
    PARAMETERS
    ----------
    outPath : string (Default = ./ )
        The path to save the csv and data analysis.
    path : string (Default = ./Models/ )
        The path of Molprobity pdf. (All files must be on same folder and
        its names MUST be on modeller output file!).
    FILE_DOT_OUT : string (default = analysis.out)
        Name of output file.
    CSV_NAME : string (default = analysis.out)
        Name of .csv file with data from Modeller and Molprobity outputs
    MAX_CLUSTERS : int (default = 10)
        Maximum number of clusters for k-dependent methods.
    PER_CONNECT : double (default = 0.5)
        Percentage of the data size used as number of neighbors for Ward.
    SIL_DAMPING : double (default = 0.1)
        Minimum percentage of silhouette number to be considered as actual increasing.
    NORM_METHOD : string (default = StandardScaler)
        Method for normilize the data. Options : {'StandardScaler', 'MinMax'}
    saveFig : Boolean (default = False)
        Save a figure of all cluster results Yes (True)/No (False).
    clustering_names : List[string] (default = ['AffinityPropagation', 'DBSCAN', 'KMeans',
                                                'MeanShift', 'SpectralClustering', 'Ward'])
        List of Method names. Supported methods are: KMeans, AddinityPropagation, MeanShift,
        SpecrtalClustering, Ward, DBSCAN.
    modellerScores: List[string] (default = ['molpdf', 'DOPE', 'DOPEHR', 'GA341', 'NDOPE'])
        List of Modeller attributes to evaluate.
        Options : {'molpdf', 'DOPE', 'DOPEHR', 'GA341', 'NDOPE'}
    molprobityScores: List[string] (default = ['outlier', 'allowed'])
        List of Molprobity attributes to evaluate.
        Options : {'outlier', 'allowed'}
    theColors : string (default = bgrcmykbgrcmykbgrcmykbgrcmyk)
        A stirng which each letter is a matplotlib color. (b : blue; g : green; r : red;
        c : cyan; m : magenta; y : yellow; k : black; w : white)
    
    RETURNS
    -------
        
    '''
    
    ##########################################  PREPARING DATA  ################################################
    log.info('\n\n\t\tQuality-Models Clusterizer\n\n')
    
    if not modellerScores or not any(x in ['molpdf', 'DOPE', 'DOPEHR', 'GA341', 'NDOPE'] for x in modellerScores):
        log.error("modellerScores list has no valid value or its empty.\nValid values are: molpdf, DOPE, DOPEHR, GA341, NDOPE\n\nABORTING EXECUTION")
        exit()
    
    if not molprobityScores or not any(x in ['outlier', 'allowed'] for x in molprobityScores):
        log.error("molprobityScores list has no valid value or its empty.\nValid values are: outlier, allowed\n\nABORTING EXECUTION")
        exit()
        
    if molprobity: 
        os.system('mkdir Modelos')
        
    log.info('#######  Preparing data...')
    t0 = time.time()
    
    clustering_names.sort();
    
    # colors used after on the plot
    colors = np.array([x for x in theColors])
    colors = np.hstack([colors] * 20)

    plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    plt.subplots_adjust(left=.05, right=.98, bottom=.1, top=.96,
                        wspace=.2, hspace=.2)

    plot_num = 1

    D = []
    
    with open(FILE_DOT_OUT, 'r') as content:
        c = content.read()
        c_list = c.split('>>')[1].split('--------')[-1].strip().split('\n')
        
        for line in c_list:
            v = line.split()
            pdb, var = v[0], v[1::]
            rt  = pdb.split('.pdb')[0]
            
            if bool(re.match('^[-]+$', rt)):
                continue
                
            pdf = path + rt + '.pdf'
            var = [float(i) for i in var]
            
            #print(pdf)
            
            # This code should be uncommented when you have not already generated the 'MolProbity Ramachandran analysis' for the pdb files to be analyzed.
            # It's necessary to install molprobity to run it.
            if molprobity:
                os.system('java -Xmx256m -cp /home/medina/Documentos/Paper_Agrupamento_Proteinas/molprobity/lib/chiropraxis.jar chiropraxis.rotarama.Ramalyze -pdf '+pdb+' '+pdf)
                os.system('mv *.pdf ./Modelos')
            aux_path = './Modelos/' + rt + '.pdf'
            
            d = dict()
            
            gen = do_grep(aux_path, 'allowed')
            outputs = [output for output in gen]
            
            if 'allowed' in molprobityScores:
                try:
                    d['allowed' ] = float(re.sub(',', '.', outputs[0].split('%')[0].split('\'')[1].strip()))
                except:
                    d['allowed'] = 0
            #s = os.popen('pdfgrep allowed  '+pdf).read()
            #p = float(re.sub(',','.',s.split('%')[0].strip()))    
            
            #s = os.popen('pdfgrep outliers  '+pdf).read()
            gen = do_grep(aux_path, 'outliers')
            outputs = [output for output in gen]
            
            if 'outlier' in molprobityScores:
                try:
                    d['outlier' ] = int(outputs[0].split('outliers')[0].split('were')[-1].strip())        
                except:
                    d['outlier' ] = 0
            
            d['pdb']=rt
            
            if 'molpdf' in modellerScores:
                d['molpdf'  ] = var[0]
                
            if 'DOPE' in modellerScores:
                d['DOPE'    ] = var[1]
                
            if 'DOPEHR' in modellerScores:
                d['DOPEHR'  ] = var[2]
            
            #if 'GA341' in modellerScores:
            #    d['GA341'   ] = var[3]
            
            if 'NDOPE' in modellerScores:
                d['NDOPE'   ] = var[4]
                
            D.append(d)
            
            
    
    D = pd.DataFrame(D)
    
    # Find uniform columns
#    nunique = D.apply(pd.Series.nunique)
#    cols_to_drop = nunique[nunique == 1].index
#    D.drop(cols_to_drop, axis=1)
    
    # Create a csv with data
    D.to_csv(path + CSV_NAME, index=False)

    # Create a csv with data
    #aux =  pd.read_csv(path + CSV_NAME)

    # Concatenate scores
    listOfAtrr = modellerScores + molprobityScores
    
    allowedScores = ['molpdf', 'DOPE', 'DOPEHR', 'NDOPE', 'outlier', 'allowed']
    
    # Remove uniform columns
#    for dropThis in cols_to_drop:
#        #print(dropThis)
#        listOfAtrr.remove(dropThis)
#        allowedScores.remove(dropThis)

    #print(allowedScores)
    # Remove not allowed values
    listOfAtrr = list(filter(lambda i: i in allowedScores, listOfAtrr))
    #print(listOfAtrr)
    X = D[listOfAtrr]
    
    #print(X)
    pdb_names = D['pdb']

    dt = np.asarray(X)
    
    #print(dt)
    
    if NORM_METHOD == 'MinMax': 
        # normalize the data in the space of 0 to 1
        for i in range(len(dt[0])): 
            # If column is uniform discard it
            #if np.all(dt[0:i] == dt[:i], axis=1):
            #    dt = np.delete(dt, i, axis=1)
            #    #dt = np.delete(dt, i, axis=2)
            #    continue
                
            if sum(dt[:, i]) != 0:
                #print("\n\nCOLUNA MM: " + str(i))
                #print("\nDIVISOR DO MINMAX: " + str(abs(dt[:, i]).max()))
                dt[:, i] = (dt[:, i] /  abs(dt[:, i]).max())**2
                
                #print(dt[:, i])
                
    else:
        if NORM_METHOD != 'StandardScaler':
            log.warn("NORM_METHOD must be either MinMax or StandardScaler, running as StandardScaler, since it is the default method")
        # normalize the data with mean 0 and stf 1
        for i in range(len(dt[0])): 
            mean_c = np.mean(dt[:, i])
            std_c = np.std(dt[:, i])
            
            #print("\n\nCOLUNA SS: " + str(i))
            #print("\nMEDIA CALC: " + str(mean_c))
            #print("\nDESVIO CALC: " + str(std_c))            
            
            if std_c < 1e-4:
                std_c = 1
            dt[:, i] = ((dt[:, i] - mean_c) /  std_c)
            
            #print(dt[:, i])
        

    #print(dt)
    # run PCA for the normalized data
    pca = PCA(n_components=2)
    print("\nAntes do PCA\n")  
    #print(X)
    print(D[listOfAtrr])
    X = pca.fit(dt).transform(dt)
    print("\nDepois do PCA\n")    
    print(X)
    # PCA process results
    results = pca.components_
    print("\nResultados PCA: " + str(results))
    covm = pca.explained_variance_ratio_
    print("\nVariance PCA: " + str(covm))
    
    if not os.path.exists('./../' + NORM_METHOD + '_pca_results.txt'):
        f = open('./../' + NORM_METHOD + '_pca_results.txt', 'w')
        
        head_line = 'pbd'
        for c in range(2):
            for at in allowedScores:
                head_line = head_line + ', ' + at + '_coor' + str(c+1)                
        head_line = head_line + ', coef_var_coor1, coef_var_coor2\n'
        print("HEAD LINE PCA: " + head_line)
        f.write(head_line)
        f.close()
        
    f = open('./../' + NORM_METHOD + '_pca_results.txt', 'a+')
    f.write(rt.split('.')[0]+', '+str([*results[0], *results[1], *covm])[1:-1]+'\n')
    f.close()
    
    f = open('./../' + NORM_METHOD + '_corr_mtx.txt', 'a+')
    corr_mtx = pd.DataFrame(X).corr()
    corr_mtxd = pd.DataFrame(dt).corr()
    print("\nCorrelation Matriz: \n")
    print(corr_mtx)
    print(corr_mtxd)
    f.close()

    # connectivity matrix for structured Ward
    n_neig = int(len(X) * PER_CONNECT)
    connectivity = kneighbors_graph(X, n_neighbors=n_neig, include_self=True)
    
    # make connectivity symmetric
    affinity = 'euclidean'
    connectivity = 0.5 * (connectivity + connectivity.T)
    connectivity, n_components = cluster.hierarchical._fix_connectivity(X, connectivity, affinity)
    
    # define cutoff for DBSCAN    
    if NORM_METHOD == 'MinMax': n_eps = 0.1
    else:
        if NORM_METHOD != 'StandardScaler':
            log.warn("NORM_METHOD must be either MinMax or StandardScaler, running as StandardScaler, since it is the default method")
        n_eps = 2*2.57*0.05
    
    t1 = time.time()
    log.info('\tTime spended (preparing data): %f s' % (t1-t0))

    ##########################################  METHODS DEFINITION ##############################################

    #clustering_names = ['AffinityPropagation', 'DBSCAN', 'KMeans', 'MeanShift', 'SpectralClustering', 'Ward']

    log.info('\n#######  Defining clustering methods...')
    t0 = time.time()

    # create clustering estimators
    
    clustering_algorithms = []
    
    if 'AffinityPropagation' in clustering_names:
        try:
            affinity_propagation = cluster.AffinityPropagation(damping=0.9) #,preference=-1)
            clustering_algorithms.append(affinity_propagation)
        except Exception as e:
            log.warn("Problems were found while running Affinity Propagation clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
            print("Problems were found while running Affinity Propagation clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
    
    if 'DBSCAN' in clustering_names:
        try:
            dbscan = cluster.DBSCAN(eps=n_eps, min_samples = 5,algorithm='kd_tree', metric='euclidean')
            clustering_algorithms.append(dbscan)
        except Exception as e:
            log.warn("Problems were found while running DBSCAN clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
            print("Problems were found while running DBSCAN clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
    
    if 'KMeans' in clustering_names:
        log.info('\n\t(K-means) Searching best k-number... ')
        try:
            k, _, _ = __best_k_of_clusters('KMeans', X, MAX_CLUSTERS)
            log.info('\tk_best = ' + str(k))
            two_means = cluster.KMeans(n_clusters=k, init='k-means++')
            clustering_algorithms.append(two_means)
        except Exception as e:
            log.warn("Problems were found while running KMeans clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
            print("Problems were found while running KMeans clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
    
    if 'MeanShift' in clustering_names:
        try:
            ms = cluster.MeanShift()
            clustering_algorithms.append(ms)
        except Exception as e:
            log.warn("Problems were found while running MeanShift clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
            print("Problems were found while running MeanShift clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
    
    if 'SpectralClustering' in clustering_names:
        log.info('\n\t(Spectral) Searching best k-number... ')
        try:
            k, _, _ = __best_k_of_clusters('SpectralClustering', X, MAX_CLUSTERS)
            #print(k)
            log.info('\tk_best = ' + str(k))
            #spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity='nearest_neighbors')
            spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver=None, 
                                                  random_state=None, n_init=10, gamma=1., 
                                                  affinity='rbf', n_neighbors=10, eigen_tol=0.0, 
                                                  degree=3, 
                                                  coef0=1, kernel_params=None)
            clustering_algorithms.append(spectral)
        except Exception as e:
            log.warn("Problems were found while running Spectral clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
            print("Problems were found while running Spectral clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
    
    if 'Ward' in clustering_names:
        log.info('\n\t(Ward) Searching best k-number... ')
        try:
            k, _, _ = __best_k_of_clusters('Ward', X, MAX_CLUSTERS, connectivity=connectivity)
            log.info('\tk_best = ' + str(k))
            ward = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=connectivity)
            clustering_algorithms.append(ward)
        except Exception as e:
            log.warn("Problems were found while running Ward clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
            print("Problems were found while running Ward clustering algorithm for " + NORM_METHOD + " normalization, skipping its execution.\nProblem: " + str(e))
    
    
        

    
            

    
    #clustering_algorithms = [two_means, affinity_propagation, ms, spectral, ward, dbscan]

    t1 = time.time()
    log.info('\n\tTime spended (defining clustering methods): %f s' % (t1-t0))

    ##########################################  CLUSTERS & PLOTS ###############################################

    log.info('\n####### Cluster & Dispersion graphs...')
    t0 = time.time()

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        algorithm.fit(X)
        
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)
        
        # plot
        plt.subplot(2, len(clustering_algorithms)//2, plot_num)
        plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
        
        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                transform=plt.gca().transAxes, size=15,
                horizontalalignment='right')
        plot_num += 1

    t1 = time.time()
    log.info('\tTime spended (clst. disp. graf.): %f s' % (t1-t0))

    ##########################################  OUTPUT FILES ###################################################

    log.info('\n####### Generating output files...')
    t0 = time.time()

    # File containing clusters data
    cluster_date_dir = 'Clusters_Data_'+NORM_METHOD
    if not os.path.isdir(outPath + cluster_date_dir):
        os.makedirs(outPath + cluster_date_dir)
     
    for name, algorithm in zip(clustering_names, clustering_algorithms):
        
        # Read labels of the algorithm
        X_labels = algorithm.labels_

        # Try to write the representative model for the clusters on analysis.out
        try:
            # Adding results on the moddeler file analysis.out
            with open(FILE_DOT_OUT, 'a') as arq:
                if clustering_names[0] == name:
                    arq.writelines('\n\n##############################################################################################')     
                arq.writelines('\n>>Clustering results - Representative structure - ' + name)       
                arq.writelines('\nCluster\t\tFile_Name\n')       

                Vec = []
                
                # If the clustering method has cluster_centers_ attribute and it isn't nor KMeans neither MeanShift (on these Medoid != Centroid)
                # In this set of clustering methods - AffinityPropagation
                if hasattr(algorithm, 'cluster_centers_') and (name != 'KMeans') and (name != 'MeanShift'):
                    centers = algorithm.cluster_centers_[:]
                    r = int(centers[:,0].size)                

                    for j in range(r):
                        
                        m = __aprx_medoid(X, centers[j,:])
                        nm = __medoid_name(X, pdb_names, [0,1], [m[0], m[1]])
                        
                        arq.write(str(j) + '\t\t')                    
                        arq.write(nm + '\n')

                        x_aux = dict()
                        x_aux['Nome_pdb' ] = nm #str(c) 
                        x_aux['Cluster'  ] = j
                        Vec.append(x_aux)

                else:
                    algorithm.cluster_centers_ = []
                    for lb in set(algorithm.labels_):
                        labels = algorithm.labels_
                        data_frame = pd.DataFrame(X)
                        algorithm.cluster_centers_.append(data_frame[labels == lb].mean(axis=0).values)

                    medians, _ = metrics.pairwise_distances_argmin_min(algorithm.cluster_centers_, data_frame.values)                    

                    j = 0 
                    
                    # find medoids
                    for m in medians:
                        nm = __medoid_name(X, pdb_names, [0,1], [X[m,0], X[m,1]])

                        arq.write(str(j) + '\t\t')
                        arq.write(str(nm) + '.pdb\n')

                        x_aux = dict()
                        c = 'MEDOID:\t' + str(nm) + '.pdb'
                        x_aux['Cluster'  ] = j
                        x_aux['\tFilename' ] = str(c) 
                        Vec.append(x_aux)

                        j = j + 1  
                
                if clustering_names[-1] == name:
                    arq.writelines('##############################################################################################')

                # create results vector for the clustering method
                for i in range(pdb_names.size):
                    x_aux = dict()
                    c = '\t'+pdb_names[i] + '.pdb' 

                    x_aux['Cluster'  ] = X_labels[i]
                    x_aux['\tFilename' ] = str(c)  
                    Vec.append(x_aux)

                # sort results vector by n-cluster
                Vec = sorted(Vec, key=lambda k:k['Cluster'])
                Vec = pd.DataFrame(Vec)
                            
                # n-cluster == -1 are Outlier data (for DBscan)
                Vec.loc[Vec.Cluster == -1, ['Cluster']] = 'Outlier'
                
                # Write .csv results
                Vec.to_csv(outPath + cluster_date_dir + '/' + name + '_Data.csv', index=False)

        except Exception as ex:
            log.error('Error 1: {0}'.format(ex))

    t1 = time.time()
    log.info('\tTime spended (Generating output files): %f s' % (t1-t0))

    log.info('\n\n\t\tThat\'s it!\n\n\n')

    if saveFig == True:
        plt.savefig(NORM_METHOD+'_dispersion_graph.png')
        
    plt.show()

def qmc(outPath='./', path='./', FILE_DOT_OUT='analysis.out', CSV_NAME='model.csv', MAX_CLUSTERS=10, PER_CONNECT=0.5, SIL_DAMPING=0.1, 
        NORM_METHOD='StandardScaler', clustering_names=['AffinityPropagation', 'DBSCAN', 'KMeans', 'MeanShift', 'SpectralClustering', 'Ward'], 
        modellerScores = ['molpdf', 'DOPE', 'DOPEHR', 'GA341', 'NDOPE'], molprobityScores=['outlier', 'allowed'], 
        theColors='bgrcmykbgrcmykbgrcmykbgrcmyk', saveFig=True, verbose=False, molprobity=False):
    '''
    The Quality-Models Clusterizer only public function. It deals with inputs
    
    PARAMETERS
    ----------
    outPath : string (Default = ./ )
        The path to save the csv and data analysis.
    path : string (Default = ./Models/ )
        The path of Molprobity pdf. (All files must be on same folder and
        its names MUST be on modeller output file!).
    FILE_DOT_OUT : string (default = analysis.out)
        Name of output file.
    CSV_NAME : string (default = analysis.out)
        Name of .csv file with data from Modeller and Molprobity outputs
    MAX_CLUSTERS : int (default = 10)
        Maximum number of clusters for k-dependent methods.
    PER_CONNECT : double (default = 0.5)
        Percentage of the data size used as number of neighbors for Ward.
    SIL_DAMPING : double (default = 0.1)
        Minimum percentage of silhouette number to be considered as actual increasing.
    NORM_METHOD : string (default = StandardScaler)
        Method for normilize the data. Options : {'StandardScaler', 'MinMax'}
    clustering_names : List[string] (default = ['AffinityPropagation', 'DBSCAN', 'KMeans',
                                                'MeanShift', 'SpectralClustering', 'Ward'])
        List of Method names. Supported methods are: KMeans, AddinityPropagation, MeanShift,
        SpecrtalClustering, Ward, DBSCAN.
    modellerScores: List[string] (default = ['molpdf', 'DOPE', 'DOPEHR', 'GA341', 'NDOPE'])
        List of Modeller attributes to evaluate.
        Options : {'molpdf', 'DOPE', 'DOPEHR', 'GA341', 'NDOPE'}
    molprobityScores: List[string] (default = ['outlier', 'allowed'])
        List of Molprobity attributes to evaluate.
        Options : {'outlier', 'allowed'}
    theColors : string (default = bgrcmykbgrcmykbgrcmykbgrcmyk)
        A stirng which each letter is a matplotlib color.
        Options : {b : blue; g : green; r : red; c : cyan; m : magenta; y : yellow;
        k : black; w : white}
    saveFig : Boolean (default = False)
        Save a figure of all cluster results Yes (True) / No (False)
    verbose : Boolean (default = False)
        Set verbose mode On(True) / Off(False)
        
    RETURNS
    -------
    
    '''
    
    # Verbose mode treatment
    #if verbose == True:
    #    log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG)
    #    log.info('Verbose output enabled!')
    #else:
    #    log.basicConfig(format='%(levelname)s: %(message)s', level-log.NOTSET)
    
    print("Using: " + NORM_METHOD + "\n")
    # Check for invalid chars in color string and remove them
    regex = re.compile('^[bgrcmykw]+$')
    if not(bool(regex.match(theColors))):
        theColors = regex.sub('', theColors)
    
    # Check if color string is empty, if so, set it to default value
    if theColors:
        theColors = 'bgrcmykbgrcmykbgrcmykbgrcmyk'

    __innerQmc(outPath=outPath, path=path, FILE_DOT_OUT=FILE_DOT_OUT, MAX_CLUSTERS=MAX_CLUSTERS, PER_CONNECT=PER_CONNECT,
               SIL_DAMPING=SIL_DAMPING, NORM_METHOD=NORM_METHOD, clustering_names=clustering_names, modellerScores=modellerScores,
               molprobityScores=molprobityScores, theColors=theColors, saveFig=saveFig, molprobity=molprobity)
    
    
    

#qmc(NORM_METHOD = "MinMax", path="./", saveFig=True, molprobity=True)
#qmc(NORM_METHOD = "StandardScaler", path="./", saveFig=True, molprobity=False)
