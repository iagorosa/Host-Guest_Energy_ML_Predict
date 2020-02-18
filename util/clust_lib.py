#%%

from exp_dt_lib import *
from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


import time

#%%

def run_clust(X, clustering_names=['DBSCAN', 'KMeans', 'Ward'], saveFig=True, file_name='', folder_name='', MAX_CLUSTERS=10, PER_CONNECT=0.5):

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
        
        # dataset shape
        dim = X.shape[-1]
        if dim == 2:
            silhouette=False
            sil_elbow=True
        else:
            silhouette=True
            sil_elbow=False
            
            
    
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
                k, _, _ = __best_k_of_clusters('KMeans', X, MAX_CLUSTERS, folder_name=folder_name, file_name=file_name, silhouette=silhouette, sil_elbow=sil_elbow)
                two_means = cluster.KMeans(n_clusters=k, init='k-means++')
                clustering_algorithms.append(two_means)
            except Exception as e:
                print("Problems were found while running KMeans clustering algorithm. \nProblem: " + str(e))
    
        if 'Ward' in clustering_names:
            try:
                k, _, _ = __best_k_of_clusters('Ward', X, MAX_CLUSTERS, connectivity=connectivity, folder_name=folder_name, file_name=file_name, silhouette=silhouette, sil_elbow=sil_elbow)
                ward = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=connectivity)
                clustering_algorithms.append(ward)
            except Exception as e:
                print("Problems were found while running Ward clustering algorithm. \nProblem: " + str(e))
        
    
        ##########################################  CLUSTERS & PLOTS ###############################################
        
        
        if dim == 1:
            
            ################################
            # colors used after on the plot
            theColors='bgrcmykbgrcmykbgrcmykbgrcmyk'
            colors = np.array([x for x in theColors])
            colors = np.hstack([colors] * 20)
        
    #        len(clustering_names) * 2 + 3, 9.5
            pl.figsize=()
            pl.figure()
            pl.subplots_adjust(left=.05, right=.98, bottom=.1, top=.96,
                                wspace=.2, hspace=.4)
        
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
                pl.subplot(len(clustering_algorithms), 1, plot_num)
                pl.title(name, size=18)
                pl.scatter(X[:, 0], np.zeros_like(X[:, 0]), color=colors[y_pred].tolist(), s=10)
                
                if hasattr(algorithm, 'cluster_centers_'):
                    centers = algorithm.cluster_centers_
                    center_colors = colors[:len(centers)]
                    pl.scatter(centers[:, 0], np.zeros_like(centers[:, 0]), s=100, c=center_colors)
                
                if (plot_num != len(clustering_algorithms)):
                    pl.xticks([], [])
                
                pl.yticks([], [])
                
                try:
                    silh = __evaluate_silhouette_score(X, y_pred)
#                    print(silh)
                    x_lim = pl.xlim() 
                    y_lim = pl.ylim()
                    pl.text(x_lim[-1]-x_lim[-1]*0.29, y_lim[-1]-y_lim[-1]*0.31, 'silh = '+str(round(silh, 3)))
                except:
                    pass

                    
    #            pl.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
    #                    transform=pl.gca().transAxes, size=15,
    #                    horizontalalignment='right')
                plot_num += 1
            
        if dim == 2:    
        
            ################################
            # colors used after on the plot
            theColors='bgrcmykbgrcmykbgrcmykbgrcmyk'
            colors = np.array([x for x in theColors])
            colors = np.hstack([colors] * 20)
        
    #        len(clustering_names) * 2 + 3, 9.5
            pl.figsize=()
            pl.figure()
            pl.subplots_adjust(left=.05, right=.98, bottom=.1, top=.96,
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
                pl.subplot(1, len(clustering_algorithms), plot_num)
                pl.title(name, size=18)
                pl.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
                
                if hasattr(algorithm, 'cluster_centers_'):
                    centers = algorithm.cluster_centers_
                    center_colors = colors[:len(centers)]
                    pl.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
                    
                if (plot_num != 1):
                    pl.yticks([], [])
                    
                try:
                    silh = __evaluate_silhouette_score(X, y_pred)
#                    print(silh)
                    x_lim = pl.xlim() 
                    y_lim = pl.ylim()
                    pl.text(x_lim[-1]-x_lim[-1]*0.70, y_lim[-1]-y_lim[-1]*0.09, 'silh = '+str(round(silh, 3)))
                except:
                    pass
                    
    #            pl.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
    #                    transform=pl.gca().transAxes, size=15,
    #                    horizontalalignment='right')
                plot_num += 1
    
        #t1 = time.time()
        
        if dim == 3:
            
            ################################
            # colors used after on the plot
            theColors='bgrcmykbgrcmykbgrcmykbgrcmyk'
            colors = np.array([x for x in theColors])
            colors = np.hstack([colors] * 20)
            
            for name, algorithm in zip(clustering_names, clustering_algorithms):
                # predict cluster memberships
                algorithm.fit(X)
                
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)
                    
                fig = pl.figure()
                ax = fig.add_subplot(111, projection='3d')
                
                ax.scatter(list(pd.DataFrame(X)[0]), list(pd.DataFrame(X)[1]), list(pd.DataFrame(X)[2]), 
                           c=colors[y_pred].tolist(), marker='o', s=100)
                
                ax.set_ylabel('Y coor')
                ax.set_zlabel('Z coor')
                
                try:
                    silh = __evaluate_silhouette_score(X, y_pred)
#                    print(silh)
#                    x_lim = pl.xlim() 
#                    y_lim = pl.ylim()
#                    z_lim = pl.zlim()
#                    pl.text(x_lim[-1]-x_lim[-1]*0.70, y_lim[-1]-y_lim[-1]*0.09, z_lim[-1]-z_lim[-1]*0.09, 'silh = '+str(round(silh, 3)))
                    ax.set_xlabel('X coor      silh = ' + str(round(silh, 3)))
                except:
                    ax.set_xlabel('X coor')
                    
                
                pl.savefig('./imgs/'+folder_name+'/PCA_Clust_results/plot_clust/'+file_name+'_dim_'+str(dim)+'_'+name+'_cluster_dispersion_graph.png', dpi=300)

            
            
        
        pl.tight_layout()
        
        if (saveFig) and (folder_name != ''):
            try:
                os.mkdir('./imgs/'+folder_name)
            except:
                pass
        
            try:
                os.mkdir('./imgs/'+folder_name+'/PCA_Clust_results')
            except:
                pass
            
            try:
                os.mkdir('./imgs/'+folder_name+'/PCA_Clust_results/plot_clust')
            except:
                pass
        
        if saveFig == True and dim < 3:
            pl.savefig('./imgs/'+folder_name+'/PCA_Clust_results/plot_clust/'+file_name+'_dim_'+str(dim)+'_cluster_dispersion_graph.png', dpi=300)
        
        pl.show()

#%%

def run_pca(X, atributes, atr_type='', newDim=2, normMethod='MinMax', save_txt=True, file_name='', folder_name=''):
    
    # atr_type = 'host' if str.lower(atributes[0][:4]) == 'host' else 'ligante'
    X_ = X[atributes].dropna()
    
    if normMethod == 'MinMax': 
        
        scaler = MinMaxScaler()
        X_ = scaler.fit_transform(X_)
#        
#        # normalize the data in the space of 0 to 1
#        for i in atributes: 
#            # If column is uniform discard it
#            #if np.all(dt[0:i] == dt[:i], axis=1):
#            #    dt = np.delete(dt, i, axis=1)
#            #    #dt = np.delete(dt, i, axis=2)
#            #    continue
#                
#            if sum(X_[i]) != 0:
#                #print("\n\nCOLUNA MM: " + str(i))
#                #print("\nDIVISOR DO MINMAX: " + str(abs(dt[:, i]).max()))
#                X_[i] = (X_[i] /  abs(X_[i]).max())**2
#                
#                #print(dt[:, i])
                    
    #print(dt)
    # run PCA for the normalized data
    pca = PCA(n_components=newDim)
    #print("\nAntes do PCA\n")  
    #print(X_)
    #print(atributes)
    X_t = pca.fit(X_).transform(X_)
    #print("\nDepois do PCA\n")    
    #print(X_t)
    # PCA process results
    results = pca.components_
    #print("\nResultados PCA: " + str(results))
    covm = pca.explained_variance_ratio_
    #print("\nVariance PCA: " + str(covm))
    
    if save_txt:
        
        if folder_name != '':
            try:
                os.mkdir('./imgs/'+folder_name)
            except:
                pass
        
            try:
                os.mkdir('./imgs/'+folder_name+'/PCA_Clust_results')
            except:
                pass
        
        output_str = ''
        output_str = output_str + "######################## Base " + folder_name + " reduction for dim " + str(newDim) + "\n\n"
        output_str = output_str + "\nAtributos da base: "+ str(atributes) +"\n\nResultados PCA: " + str(results)  + "\n\n" \
                                + "Variance PCA: " + str(covm) + "\n\n\n"
        output_str = output_str + "\nDepois do PCA\n" + str(X_t) + "\n\n\nAntes do PCA\n" + str(atributes) + str(X_) + "\n\n\n\n"
        
        text_file = open('./imgs/'+folder_name+'/PCA_Clust_results/'+'resultados_'+file_name+"_"+atr_type+'.txt', "w")
        text_file.write(output_str)
        text_file.close()
        
    return X_t, results, covm
 
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

def __best_k_of_clusters(algorithm_name, X, upK, downK=1, connectivity=0, silhouette=False, elbow=False, sil_elbow=True, save_graf=True, folder_name='', file_name=''):
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
    
    if save_graf and (folder_name != ''):
            try:
                os.mkdir('./imgs/'+folder_name)
            except:
                pass
        
            try:
                os.mkdir('./imgs/'+folder_name+'/PCA_Clust_results')
            except:
                pass
            
            try:
                os.mkdir('./imgs/'+folder_name+'/PCA_Clust_results/silh_res')
            except:
                pass
            
            dim = X.shape[-1]
             
            pl.figure()

            pl.title(algorithm_name+'_dim_'+str(dim)+' Silhouette search graph')
            pl.grid()
            pl.tight_layout()
            
            pl.plot(list_n_k, list_silh)
             
            pl.savefig('./imgs/'+folder_name+'/PCA_Clust_results/silh_res/'+algorithm_name+'_'+file_name+'_silhouette_search_graph.png', dpi=300)
        
            
    
    if silhouette:
        return (k_silh, '-', '-')
    elif elbow:
        return ('-', k_elbow, max_angle)
    elif sil_elbow:
        return (k_silh, k_elbow, max_angle) 