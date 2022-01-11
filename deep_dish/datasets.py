import pandas as pd
from torch.utils.data import Dataset
import zarr

class CellPatchesDataset(Dataset):
    def __init__(self,filename):
        self.metadata = pd.read_csv(filename)

    def __len__(self):
        return self.metadata.pipe(len)

    def __getitem__(self,index):
        index_metadata = self.metadata.iloc[index]
        z = zarr.open(index_metadata['store'])
        return z[index_metadata['array']][index_metadata['array_index']], index_metadata['gene_symbol']