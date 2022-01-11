import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import zarr

class CellPatchesDataset(Dataset):
    def __init__(self,filename):
        self.metadata = pd.read_csv(filename)
        self.gene_symbols = {
            gi:gs for gs,gi in pd.unique(
                list(zip(self.metadata['gene_symbol'], self.metadata['gene_id']))
            )
        }
        self.n_classes = len(self.gene_symbols)

    def __len__(self):
        return self.metadata.pipe(len)

    def __getitem__(self,index):
        index_metadata = self.metadata.iloc[index]
        z = zarr.open(index_metadata['store'])
        x = z[index_metadata['array']][index_metadata['array_index']]
        assert x.dtype == np.uint16
        x = (x/(2**16 - 1)).astype(np.float32)
        return x, index_metadata['gene_id']