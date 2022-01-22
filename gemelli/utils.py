import numpy as np
import pandas as pd
import re


def parse_dapi_logs(f):
    df_logs = pd.read_csv(f,sep=',?[^\d]+:',header=None,skiprows=[0],engine='python',index_col=False)

    df_logs = df_logs.iloc[:,1:] # cutoff extra first column

    if df_logs.isna().sum().sum() > 0: # likely multiple runs concatenated in the same file
        print('parsing file with multiple training run entries.')
        with open(f) as f_:
            lines = f_.readlines()
        header_lines = np.array([l.startswith('=') for l in lines]).nonzero()[0]
        if len(header_lines)>1:
            df_logs = pd.read_csv(f,sep=',?[^\d]+:',header=None,skiprows=list(header_lines),engine='python',index_col=False)
            df_logs = df_logs.iloc[:,1:] # cutoff extra first column
        else:
            pass
    df_logs.iloc[:,-1] = df_logs.iloc[:,-1].str[:-1].astype(float) # remove comma from end of lines

    with open(f) as f_:
        _ = f_.readline()
        line = f_.readline()
    df_logs.columns = re.split(r':[^a-zA-Z]+, ',line)[:-1]
    df_logs['global_step'] = (df_logs['epoch']-1)*df_logs['iters'].max() + df_logs['iters']
    df_logs = df_logs.drop_duplicates(subset=['global_step'],keep='last') # drop duplicates of the same epoch/iteration
    return df_logs