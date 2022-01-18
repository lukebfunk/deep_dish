import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm

from gemelli.plot import clustered_confusion


parser = argparse.ArgumentParser()
parser.add_argument('tables',nargs='+',type=str)
parser.add_argument('--normalize',default='true',type=str)
parser.add_argument('--vlim',nargs=2,type=float,default=[None,None])
parser.add_argument('--log',type=bool,default=False)
parser.add_argument('--metric',type=str,default='euclidean')
parser.add_argument('--sort',type=str,default='control_accuracy')
args = parser.parse_args()

controls = pd.read_csv('/nrs/funke/funkl/data/gemelli_positive_controls.txt',header=None)[0].tolist()

for n, (csv) in enumerate(args.tables):
    print(f'processing {csv}...')
    df_cm = pd.read_csv(csv,index_col=0)

    df_cm['positive_control'] = False
    df_cm.loc[df_cm.index.isin(controls),'positive_control'] = True
    control_labels = df_cm['positive_control'].map({True:'black',False:'white'}) 
    df_cm = df_cm.drop(columns = ['positive_control'])

    if args.sort == 'control_accuracy':
        control = df_cm.index == 'nontargeting'
        numerator = (df_cm.loc[:,~control].sum(axis=1)+df_cm.loc[control,control].values[0])
        denominator = (df_cm.sum(axis=1)+df_cm.loc[control].sum().values[0])
        sorted_control_accuracy = (numerator/denominator).sort_values(ascending=False).index
        df_cm = df_cm.loc[sorted_control_accuracy,sorted_control_accuracy]
        condensed_distance=None
    elif args.sort == 'IoU':
        arr = []
        for i in range(df_cm.shape[0]-1):
            for j in range(i+1,df_cm.shape[0]):
                intersection = df_cm.iloc[i,j]+df_cm.iloc[j,i]
                union = df_cm.iloc[i].sum()+df_cm.iloc[j].sum()
                arr.append(intersection/union)
        condensed_distance = 1-np.array(arr)
    elif args.sort == 'IoU_local':
        arr = []
        for i in range(df_cm.shape[0]-1):
            for j in range(i+1,df_cm.shape[0]):
                intersection = df_cm.iloc[i,j]+df_cm.iloc[j,i]
                union = df_cm.iloc[[i,j],[i,j]].sum().sum()
                if union != 0:
                    arr.append(intersection/union)
                else:
                    arr.append(0)
        condensed_distance = 1-np.array(arr)
    elif args.sort == 'IoU_directional':
        arr = []
        for i in range(df_cm.shape[0]-1):
            for j in range(i+1,df_cm.shape[0]):
                IoU_1 = (df_cm.iloc[i,j]/df_cm.iloc[i].sum())
                IoU_2 = (df_cm.iloc[j,i]/df_cm.iloc[j].sum())
                arr.append((1/2)*(IoU_1+IoU_2))
        condensed_distance = 1-np.array(arr)
    else:
        try:
            condensed_distance = pdist(df_cm.values,metric=args.sort)
        except:
            condensed_distance = None

    if args.normalize == 'true':
        df_cm = df_cm.divide(df_cm.sum(axis=1),axis=0)
    elif args.normalize == 'predicted':
        df_cm = df_cm.divide(df_cm.sum(axis=0),axis=1)
    elif args.normalize == 'none':
        df_cm = df_cm
    else:
        raise ValueError('normalize must be one of {"true","predicted","none"}')

    # after normalization
    if args.sort == 'true_positive_rate':
        tpr = pd.Series(df_cm.values[np.eye(df_cm.pipe(len),dtype=bool)],index=df_cm.index)
        sorted_tpr = tpr.sort_values(ascending=False).index
        df_cm = df_cm.loc[sorted_tpr,sorted_tpr]
        condensed_distance=None

    cg = clustered_confusion(
        df_cm,
        log=args.log,
        vmin=args.vlim[0],
        vmax=args.vlim[1],
        condensed_distance=condensed_distance,
        col_colors=control_labels,
        row_colors=control_labels,
        dendrogram_ratio=0.15,
        colors_ratio=0.005,
        cbar_pos=(0.02,0.85,0.01,0.125),
    )
    cg.savefig('figures/'+re.search(r'.*/([^/]+).csv',csv).group(1)+f'.{args.sort}.png',dpi=300)