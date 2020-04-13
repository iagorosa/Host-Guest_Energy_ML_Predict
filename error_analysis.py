#%%

import glob   as gl
import numpy  as np
import pandas as pd
import pylab as pl

#%%

test_size = [0.1, 0.7, 0.1]

ml_methods = ['EN', 'XGB', 'DTC', 'BAG', 'KNN', 'ANN', 'SVM', 'GB', 'KRR'] 
# ml_methods = ['EN', 'XGB', 'DTC', 'BAG', 'KNN', 'ANN', 'SVM', 'GB', 'KRR'] 
# ml_methods = ['KNN', 'XGB']

error_types = ['ERROR_TEST', 'ERROR_TRAIN']
df_erro = []

problem = ['alpha_trat', 'beta_trat', 'gamma_trat', 'error_']

for prob in problem:
    df_erro = []
    for et in error_types:
        df = pd.DataFrame([])
        for mlm in ml_methods:
    
            xls      = gl.glob('./RESULTADOS/MACHINE_LEARNING/'+mlm+'/CSV_'+et+'/'+prob+'*.csv') 
            xls = sorted(xls)
            
    #        # Ruan mudou por aqui e pode estar incompleto
    #        xls = sorted(xls)
    #        for x in xls: 
    #            print(x.split('_')[-8].split('/')[-1])
    
            X = pd.read_csv(xls[0], header = None)
            X.index = [mlm]*len(X)
            X = X.reset_index()
    
            for f in xls[1:]:
                X_aux = pd.read_csv(f, header = None) 
    
                X = pd.concat([X, X_aux[1]], axis=1)
    
            columns = ['METHOD', 'ERROR'] + list(np.around(np.arange(*test_size), 3))
            X.columns = columns
    
            df = pd.concat([df, X], axis=0)
    
        df = df.reset_index(drop=True)
    
        error_list = df['ERROR'].unique()
    
        for el in error_list:
            pl.figure(figsize=(10,8))
            for df1 in df[df['ERROR'] == el].iterrows():
                pl.plot(df1[1][2:], "o--", label=df1[1]['METHOD'])
    
            title = 'Training Error' if et == 'ERROR_TRAIN' else 'Test Error'
            pl.title(title + " " + el, fontsize=18)
            pl.xlabel("% test", fontsize=14)
            pl.ylabel("error", fontsize=14)
            pl.legend(loc='best')
            pl.grid()
            pl.tight_layout()
            pl.savefig("./RESULTADOS/MACHINE_LEARNING/imgs/"+et+"/"+prob+"_"+el+".png", dpi=300)
            pl.show()
            pl.close()
    
        df_erro.append(df)
        
    
    
    df_erro.append(df_erro[0][(df_erro[0]['ERROR'] != 'score') & (df_erro[0]['ERROR'] != 'time')])
    df_erro[-1].reset_index(inplace=True, drop=True)

    dif = abs(df_erro[-1].iloc[:,2:] - df_erro[1].iloc[:,2:])

    df_erro_p = pd.concat([df_erro[1].iloc[:, :2], dif], axis=1)
    
    
    
    df = df_erro_p
    error_list = df['ERROR'].unique()

    for el in error_list:
        pl.figure(figsize=(10,8))
        for df1 in df[df['ERROR'] == el].iterrows():
            pl.plot(df1[1][2:], "o--", label=df1[1]['METHOD'])
    
        title = 'Difference Error'
        pl.title(title + " " + el, fontsize=18)
        pl.xlabel("% test", fontsize=14)
        pl.ylabel("error", fontsize=14)
        pl.legend(loc='best')
        pl.grid()
        pl.tight_layout()
        pl.savefig("./RESULTADOS/MACHINE_LEARNING/imgs/DIF_ERROR/"+prob+"_"+el+".png", dpi=300)
        pl.show()
        pl.close()



# %%

df_erro.append(df_erro[0][(df_erro[0]['ERROR'] != 'score') & (df_erro[0]['ERROR'] != 'time')])
df_erro[-1].reset_index(inplace=True, drop=True)

dif = abs(df_erro[-1].iloc[:,2:] - df_erro[1].iloc[:,2:])


df_erro_p = pd.concat([df_erro[1].iloc[:, :2], dif], axis=1)

# %%

df = df_erro_p

error_list = df['ERROR'].unique()

for el in error_list:
    pl.figure(figsize=(10,8))
    for df1 in df[df['ERROR'] == el].iterrows():
        pl.plot(df1[1][2:], "o--", label=df1[1]['METHOD'])

    title = 'Difference Error'
    pl.title(title + " " + el, fontsize=18)
    pl.xlabel("% test", fontsize=14)
    pl.ylabel("error", fontsize=14)
    pl.legend(loc='best')
    pl.grid()
    pl.tight_layout()
    pl.savefig("./RESULTADOS/MACHINE_LEARNING/imgs/DIF_ERROR/"+el+".png", dpi=300)
    pl.show()
    pl.close()

#%%