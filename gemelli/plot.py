from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
import seaborn as sns

def confusion(df_cm,ax=None,annotate=True,fontsize=7):
    if not ax:
        print('making new plot')
        fig,ax = plt.subplots()

    labels = df_cm.columns
    n_classes = len(labels)

    cm = df_cm.values

    im = ax.imshow(cm,interpolation='nearest',cmap='magma')

    cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)
    text_thresh = (cm.max()+cm.min())/2.0

    if annotate:
        for i,j in product(range(n_classes),range(n_classes)):
            color = cmap_max if cm[i,j] < text_thresh else cmap_min
            ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',color=color,fontsize=fontsize)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
    )
    ax.set_xticklabels(labels,size=fontsize)
    ax.set_yticklabels(labels,size=fontsize)
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode='anchor',ha='right')
    ax.tick_params(axis='x',pad=7)

    return ax

def clustered_confusion(df_cm,method='average',condensed_distance=None,fontsize=5,log=False,**kwargs):
    if condensed_distance is not None:
        linkage = hierarchy.linkage(condensed_distance,method=method)
        kwargs.update(dict(col_linkage=linkage,row_linkage=linkage))
    else:
        kwargs.update(dict(col_cluster=False,row_cluster=False))
    if log:
        df_cm = np.log10(df_cm)
    cg = sns.clustermap(df_cm,yticklabels=True,xticklabels=True,**kwargs)
    plt.setp(cg.ax_heatmap.get_xticklabels(),size=fontsize)
    plt.setp(cg.ax_heatmap.get_yticklabels(),size=fontsize)
    cg.ax_cbar.set_ylabel('true positive fraction')
    cg.ax_heatmap.set_xlabel('predicted')
    cg.ax_heatmap.set_ylabel('true')
    return cg
