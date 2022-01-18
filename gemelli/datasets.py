import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import zarr

class CellPatchesDataset(Dataset):
    def __init__(self,filename,input_size=(256,256),use_mask=False,augment=True,restrict_query=None):
        metadata = pd.read_csv(filename)

        # for restricting training set
        if restrict_query is not None:
            metadata = metadata.query(restrict_query).sort_values('gene_id')
            # reassign class labels (same order as before)
            self.gene_symbols = {
                n:gs for n,(gs,_) in enumerate(pd.unique(
                    list(zip(metadata['gene_symbol'], metadata['gene_id'])))
                )
            }
            gene_ids = {gs:gi for gi,gs in self.gene_symbols.items()}
            metadata['gene_id'] = metadata['gene_symbol'].map(gene_ids)
        else:
            self.gene_symbols = {
                gi:gs for gs,gi in pd.unique(
                    list(zip(metadata['gene_symbol'], metadata['gene_id']))
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

        self.augment=augment
        self.augmentations = transforms.Compose([
            transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(128)
        ])

        self.labels = np.array(metadata['gene_id'])#.astype(np.int32)
        self.stores = list(metadata['store'])
        self.arrays = list(metadata['array'])
        self.array_indices = np.array(metadata['array_index']).astype(np.int64)
        self.use_mask=use_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):

        array = self.arrays[index]
        array_index = self.array_indices[index]

        z = zarr.open(self.stores[index])
        x = z[array][(array_index,slice(None),*self.xy_slicers)]
        assert x.dtype == np.uint16
        x = (x/(2**16 - 1)).astype(np.float32)

        if self.use_mask:
            mask = z[array.replace('images','cells')][(array_index,*self.xy_slicers)]
            x *= mask

        x = torch.from_numpy(x)

        if self.augment:
            x = self.augmentations(x)

        label = self.labels[index]
        return x, label