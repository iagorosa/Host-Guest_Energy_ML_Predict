#%%

import glob   as gl
import numpy  as np
import pandas as pd
import pylab as pl

#%%

test_size = [0.1, 0.7, 0.1]

# ml_methods = ['EN', 'XGB', 'DTC', 'BAG', 'KNN', 'ANN', 'ELM', 'SVM', 'GB', 'KRR'] 
ml_methods = ['EN', 'XGB', 'DTC', 'BAG', 'KNN', 'ANN', 'SVM', 'GB', 'KRR'] 
# ml_methods = ['KNN', 'XGB']

error_types = ['ERROR_TEST', 'ERROR_TRAIN']
df_erro = []

for et in error_types:
    df = pd.DataFrame([])
    for mlm in ml_methods:

        xls      = gl.glob('./RESULTADOS/MACHINE_LEARNING/'+mlm+'/CSV_'+et+'/'+'*.csv') 
        xls = sorted(xls)

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
        pl.savefig("./RESULTADOS/MACHINE_LEARNING/imgs/"+et+"/"+el+".png", dpi=300)
        pl.show()
        pl.close()

    df_erro.append(df)



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