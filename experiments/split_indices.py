from glob import glob
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import zarr

arr = []
gene_ids = dict()
next_gene_id = 0
stores = glob('cell_patches/*.zarr')
print('found the following zarr stores:\n'+'\n'.join(stores))
for store in stores:
    if store == 'cell_patches/nontargeting.zarr':
        continue
    print(f'reading zarr metadata for {store}...')
    gene = re.search(r'.*cell_patches/(.+).zarr',store).group(1)
    if gene not in gene_ids:
        gene_ids[gene] = next_gene_id
        next_gene_id +=1
    gene_id = gene_ids[gene]
    z = zarr.open(store)
    for guide,group in z.groups():
        for phase,phase_group in group.groups():
            arr.append(
                pd.DataFrame(phase_group['images'].attrs['images'])
                .assign(
                    class_predict=phase,
                    sgRNA=guide,
                    gene_symbol=gene,
                    store=store,
                    array=f'{guide}/{phase}/images',
                    gene_id=gene_id
                    )
                .reset_index()
                .rename(columns={'index':'array_index'})
                )

df = pd.concat(arr)
print(f'before removing duplicates: {df.pipe(len)} cells')
df = df.drop_duplicates(subset=['plate','well','tile','label'])
print(f'after removing duplicates: {df.pipe(len)} cells')
print('splitting: 70% train, 20% validation, 10% test per sgRNA and cell cycle phase')
def split(df):
    indices = df['array_index'].values
    train,rest = train_test_split(indices,train_size=0.7,random_state=42)
    val,test = train_test_split(rest,train_size=(2/3),random_state=42)
    df['split'] = 'train'
    df.loc[df['array_index'].isin(val),'split'] = 'val'
    df.loc[df['array_index'].isin(test),'split'] = 'test'
    return df
df = df.groupby(['gene_symbol','class_predict']).apply(split)
print('saving')
df.query('split=="train"').to_csv('~/deep_dish/dataset_info/train_samples.csv',index=False)
df.query('split=="val"').to_csv('~/deep_dish/dataset_info/val_samples.csv',index=False)
df.query('split=="test"').to_csv('~/deep_dish/dataset_info/test_samples.csv',index=False)
print('finished')