from glob import glob
import subprocess
import numpy as np
import pandas as pd

df_cm = pd.read_csv('../results/vgg_128_128class_test_confusion.epoch298-val_accuracy0.09.csv',index_col=0)
tpr = pd.Series(df_cm.values[np.eye(df_cm.pipe(len),dtype=bool)],index=df_cm.index)
sorted_tpr = list(tpr.sort_values(ascending=False).index)

command = 'bsub -n 5 -J {gene}_test -gpu "num=1" -q gpu_rtx python ~/gemelli/experiments/11_one_vs_control_testing.py --gene {gene} --fmaps 16 --gpus 1'

for gene in sorted_tpr[:64]:
    subprocess.Popen(command.format(gene=gene),shell=True)