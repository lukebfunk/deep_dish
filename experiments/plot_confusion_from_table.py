import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from tqdm.auto import tqdm

from gemelli.plot import confusion


parser = argparse.ArgumentParser()
parser.add_argument('tables',nargs='+',type=str)
parser.add_argument('--cluster',default=False,type=bool)
parser.add_argument('--normalize',default='true',type=str)
parser.add_argument('--annotate',default=False,type=bool)
parser.add_argument('--fontsize',default=7.,type=float)
parser.add_argument('--titles',nargs='*',type=str, default=None)
parser.add_argument('--output',type=str,default='confusion.png')
args = parser.parse_args()

if args.titles:
    assert len(args.titles)==len(args.tables), 'If supplying titles, must be same number as tables'

nrows = np.floor(np.sqrt(len(args.tables))).astype(int)
ncols = np.ceil(len(args.tables)/nrows).astype(int)
figsize = (3.2*ncols,4.8*nrows)
print(f'making figure with {ncols} columns, {nrows} rows, size = {figsize}')
fig,axes=plt.subplots(nrows,ncols,figsize=figsize)

for n, (csv, ax) in enumerate(zip(args.tables,axes.flatten())):
    print(f'processing {csv}...')
    df_cm = pd.read_csv(csv,index_col=0)

    # print(f'normalizing {csv}...')
    if args.normalize == 'true':
        df_cm = df_cm.divide(df_cm.sum(axis=1),axis=0)
    elif args.normalize == 'predicted':
        df_cm = df_cm.divide(df_cm.sum(axis=0),axis=1)
    elif args.normalize == 'none':
        pass
    else:
        raise ValueError('normalize must be one of {"true","predicted","none"}')
    
    # print(f'plotting {csv}...')
    if not args.cluster:
        ax = confusion(df_cm,ax=ax,annotate=args.annotate,fontsize=args.fontsize)
    else:
        # ax = sns.clustermap(df_cm)
        raise ValueError

    # print('labeling {csv}')
    if n % ncols == 0:
        ax.set_ylabel("true label")
    if n // ncols == (nrows-1):
        ax.set_xlabel("predicted label")

    if args.titles:
        ax.set_title(args.titles[n])
    else:
        ax.set_title(re.search(r'.*/([^/]+).csv',csv).group(1))

print(f'saving {args.output}...')
# plt.savefig(args.output,dpi=300,bbox_inches='tight')
plt.show()