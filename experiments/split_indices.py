import argparse
from glob import glob
from ops.utils import gb_apply_parallel
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import zarr

parser = argparse.ArgumentParser()
parser.add_argument('split_version',type=int)
parser.add_argument('--exclude_nt',type=bool,default=False)
args = parser.parse_args()

arr = []
gene_ids = dict()
next_gene_id = 0
stores = glob('cell_patches/*.zarr')
print('found the following zarr stores:\n'+'\n'.join(stores))
# hardcode to test
# stores=['CCT7','DTL','ITGAV','PSMB4','RPS9','POLR2J','SPC24','SF3B3','LMNB1','RPL26']
# stores = [f'cell_patches/{s}.zarr' for s in stores]
for store in stores:
    if args.exclude_nt:
        if store == 'cell_patches/nontargeting.zarr':
            continue
    gene = re.search(r'.*cell_patches/(.+).zarr',store).group(1)
    if gene in ['SMU1','TUT1']:
        continue
    print(f'reading zarr metadata for {store}...')
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
print(f'before removing duplicates: {df.pipe(len)} total cells, {df["gene_id"].nunique()} classes')
df = df.drop_duplicates(subset=['plate','well','tile','label'])
print(f'after removing duplicates: {df.pipe(len)} total cells, {df["gene_id"].nunique()} classes')
print('splitting: 70% train, 20% validation, 10% test per gene target and cell cycle phase')
def split(df):
    df = df.reset_index(drop=True)
    train,rest = train_test_split(df.index.values,train_size=0.7,random_state=42)
    val,test = train_test_split(rest,train_size=(2/3),random_state=42)
    df['split'] = 'train'
    df.loc[val,'split'] = 'val'
    df.loc[test,'split'] = 'test'
    return df
df = df.pipe(gb_apply_parallel,['gene_symbol','class_predict'],split)
print('saving')
df.query('split=="train"').to_csv(f'~/gemelli/dataset_info/{args.split_version}_train_samples.csv',index=False)
df.query('split=="val"').to_csv(f'~/gemelli/dataset_info/{args.split_version}_val_samples.csv',index=False)
df.query('split=="test"').to_csv(f'~/gemelli/dataset_info/{args.split_version}_test_samples.csv',index=False)
print('finished')