import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import zarr

class CellPatchesDataset(Dataset):
    def __init__(self,filename,input_size=(256,256),use_mask=False):
        self.metadata = pd.read_csv(filename)
        self.gene_symbols = {
            gi:gs for gs,gi in pd.unique(
                list(zip(self.metadata['gene_symbol'], self.metadata['gene_id']))
            )
        }
        self.n_classes = len(self.gene_symbols)
        self.input_size=input_size
        if self.input_size!=(256,256):
            crop = (np.array((256,256))-np.array(self.input_size))//2
            assert (crop<0).sum()==0
            self.xy_slicers = (slice(crop[0],-crop[0]),slice(crop[1],-crop[1]))
        else:
            self.xy_slicers = (slice(None),slice(None))
        self.use_mask=False

    def __len__(self):
        return self.metadata.pipe(len)

    def __getitem__(self,index):
        index_metadata = self.metadata.iloc[index]
        z = zarr.open(index_metadata['store'])
        x = z[index_metadata['array']][(index_metadata['array_index'],slice(None),*self.xy_slicers)]
        assert x.dtype == np.uint16
        x = (x/(2**16 - 1)).astype(np.float32)
        if self.use_mask:
            mask = z[index_metadata['array'].replace('images','cells')][(index_metadata['array_index'],*self.xy_slicers)]
            x *= mask
        return x, index_metadata['gene_id']