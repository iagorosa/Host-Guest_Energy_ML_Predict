#%%
import pandas as pd

import pylab as pl
import seaborn as sns
import numpy as np
from numpy.linalg import norm
import os

import scipy.stats as scs
from statsmodels.stats.diagnostic import lilliefors

from sklearn.preprocessing import PolynomialFeatures

#%%

def correlacoes(X, atributes, atr_type, matrix = True, grafic = False, ext = 'png', show = False, save = True, file_name='', extra_name='', scale = 1.4):
    '''
        comentario util
    '''
    
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
    
    if (grafic or matrix) and (file_name != ''):
        try:
            os.mkdir('./imgs/'+file_name)
        except:
            pass
        try:
            os.mkdir('./imgs/'+file_name+'/'+atr_type)
        except:
            pass


    if grafic == True:
        pl.rcParams.update(pl.rcParamsDefault)

        pl.figure()

        #colocar titulo no grafico
        g = sns.pairplot(X_, vars=name_atr)
        g.fig.suptitle('Pairplot '+str.capitalize(atr_type))
        
        if save == True:
            pl.savefig('./imgs/'+file_name+'/'+atr_type+'/grap_corr_'+atr_type+extra_name+'.'+ext, dpi=300)
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
            pl.savefig('./imgs/'+file_name+'/'+atr_type+'/mat_corr_'+atr_type+extra_name+'.'+ext, dpi=300)
        if show == True:
            pl.show()
    
    pl.close('all')

#%%

def boxplots(X, atributes, atr_type, complete = True, file_name=''):

    # atr_type = 'host' if str.lower(atributes[0][:4]) == 'host' else 'ligante'

    if atr_type == 'host':
        name_atr = [ i[5:] for i in atributes ]
    else:
        name_atr = atributes

    x = X.loc[:, atributes]
    x.columns = name_atr

    if file_name != '':
        try:
            os.mkdir('./imgs/'+file_name)
        except:
            pass
        try:
            os.mkdir('./imgs/'+file_name+'/'+atr_type)
        except:
            pass

    if complete == True:
        pl.figure()

        x.boxplot()
        pl.xticks(rotation=15)
        pl.title("Boxplot Atributos do "+str.capitalize(atr_type))
        pl.tight_layout()
        pl.savefig("./imgs/"+file_name+'/'+atr_type+"/boxplots_"+atr_type+".png", dpi=300)
        
        pl.show()

    else:
        for atr in name_atr:
            pl.figure()
            
            x[atr].plot.box()
            # atr = atr[5:] if str.lower(atr[:4]) == 'host' else atr
            
            pl.title("Boxplot do "+ str.capitalize(atr) + " " +str.capitalize(atr_type))
            pl.grid(axis='y')
            pl.tight_layout()
            pl.savefig("./imgs/"+file_name+"/"+atr_type+"/boxplot_"+atr+"_"+atr_type+".png", dpi=300)
            
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

def polynomial_features(X, col, degree):

    X = X[col] 
    poly = PolynomialFeatures(degree)
    pp = poly.fit_transform(X)

    df_pf = pd.DataFrame(pp)

    return df_pf

#%%
def distribution_hist_outlier_trat(X, trat=False, val_trat='', folder_name=''):
    '''
        histogramas com ajuste de um modelo normal
        trat: se verdadeiro, ativa o tratamento dos outliers e considera limiarespara os valores nos histogramas
    '''
       
    if folder_name != '':
        try:
            os.mkdir('./imgs/'+folder_name)
        except:
            pass

        try:
            os.mkdir('./imgs/'+folder_name+'/hists')
        except:
            pass
        
        try:
            os.mkdir('./imgs/'+folder_name+'/hists/csv_trat')
        except:
            pass     
        
    
    ini = True

    for atr in X.columns:
        
        atr_name = ''.join('Temp (C)'.split())

        pl.figure()

        if trat == True:
            try:
                Y   = X[atr][X[atr] > val_trat[atr][0] and X[atr] < val_trat[atr][1]] 
                
                aux = X[atr][X[atr] < val_trat[atr][0] or X[atr] > val_trat[atr][1]]
                aux.to_csv("./imgs/"+folder_name+"/hists/csv_trat/outliers_"+atr_name+"_"+str(val_trat[atr])+".csv")
            except:
                Y = X[atr]
            trat_tex = '_trat'
        else:
            Y = X[atr]
            trat_tex = ''

        Y.hist(histtype='bar', density=True, ec='black', zorder=2)

        min_ = int(round(Y.min()-0.5))
        max_ = int(round(Y.max()+0.5))

        #print(min_)
        #print(max_)

        pl.xticks(range(min_, max_, round((max_-min_)/10+0.5)))
       
        pl.xlabel(atr)
        pl.ylabel("Frequência")
       
        pl.title("Histograma " + atr)
        pl.grid(axis='x')

        # estatistica
        mu, std = scs.norm.fit(Y)

        # Plot the PDF.
        xmin, xmax = pl.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = scs.norm.pdf(x, mu, std)
        pl.plot(x, p, 'r--', linewidth=2)

        #print(mu, std)
        #print(x)

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
        pl.savefig("imgs/"+folder_name+"/hists/"+atr_name+"_"+trat_tex+".png")
        pl.show()

        pl.close()

    D.to_csv('imgs/'+folder_name+'/hists/descricao_resumo'+trat_tex+'.csv')
    
#%%

def df_outliers(X, out, save=True, folder_name=''):
    '''
    DATAFRAME PARA AS INSTANCIAS QUE TEM ALGUM DADO FALTANTE
    '''
    qtd = out.sum(axis=1)[out.sum(axis=1)>0]                                   # numero da instancia e quantidade de dados faltante em cada uma
    df_out = X.T[qtd.index].T                                                  # pega as instancias com dados faltantes pelos indices

    col = np.array(out.columns) 
    res = out.values * col.T
    res_ = [list(i[i != '']) for i in res]                                     # pega o nome da coluna com dado faltante em cada instancia

    outliers_col = [res_[i] for i in list(df_out.index)]                       # lista das colunas com dados faltantes em cada instancia encontrada em df_out
    df_out['outlier'] = outliers_col 

    if save == True:
        
        if folder_name != '':
            try:
                os.mkdir('./imgs/'+folder_name)
            except:
                pass
        
        df_out.to_csv('./imgs/'+folder_name+'/outliers.csv')

    return df_out, qtd
    
