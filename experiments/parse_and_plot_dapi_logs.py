import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gemelli.utils import parse_dapi_logs

parser = argparse.ArgumentParser()
parser.add_argument('files',nargs='+',type=str)
args = parser.parse_args()

plt.rcParams['font.size'] = 7
plt.rcParams['lines.linewidth'] = 1.

for f in args.files:
    print(f'parsing {f}...')
    df = parse_dapi_logs(f)
    metrics = [c for c in df.columns if c not in ['epoch','iters','time','data','global_step']]
    cols = np.ceil(len(metrics)/2).astype(int)
    rows = np.ceil((len(metrics))/cols).astype(int)
    fig,ax = plt.subplots(rows,cols,figsize=(8,4),gridspec_kw={'hspace':0.1,'wspace':0.4},sharex=True,sharey=False)
    for metric,ax in zip(metrics,ax.flatten()):
        ax = sns.lineplot(data=df.groupby('epoch').last(),x='epoch',y=metric,ax=ax)
        if any([m in metric for m in ['cycle','idt']]):
            ax.set_yscale('log')
    plt.suptitle(' '.join(f.strip('.txt').rsplit('/',2)[-2:]))
    plt.savefig(f.strip('.txt')+'.png',dpi=300,bbox_inches='tight')
    df.to_csv(f.strip('.txt')+'.csv',index=False)