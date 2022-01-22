import subprocess
import numpy as np
import pandas as pd

df_cm = pd.read_csv('../results/vgg_128_128class_test_confusion.epoch298-val_accuracy0.09.csv',index_col=0)
tpr = pd.Series(df_cm.values[np.eye(df_cm.pipe(len),dtype=bool)],index=df_cm.index)
sorted_tpr = list(tpr.sort_values(ascending=False).index)

command = 'bsub -n 5 -J {gene} -gpu "num=1" -q gpu_rtx python ~/gemelli/experiments/11_one_vs_control_training.py --gene {gene} --model_name {gene} --fmaps 16 --gpus 1 --lr 1e-6'

for gene in sorted_tpr:
    subprocess.Popen(command.format(gene=gene),shell=True)