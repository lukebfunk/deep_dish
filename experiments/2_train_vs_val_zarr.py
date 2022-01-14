import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import zarr

df_train = pd.read_csv('~/gemelli/dataset_info/0_train_samples.csv')
df_val = pd.read_csv('~/gemelli/dataset_info/0_val_samples.csv')

def dataset_to_zarr_groups(df,name):
    z_out = zarr.open(name)
    images = [
        zarr.open(s)[a].oindex[df_samples['array_index'].values]
        for (s,a),df_samples in tqdm(df.groupby(['store','array']))
    ]
    z_out['images'] = np.concatenate(images)



dataset_to_zarr_groups(df_train,'example_train.zarr')
dataset_to_zarr_groups(df_val,'example_val.zarr')

