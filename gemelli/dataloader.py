import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

from .sampler import DistributedWeightedRandomSampler

class BalancedDataLoader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers):
        ys = dataset.labels
        counts = np.bincount(ys)
        label_weights = 1.0 / counts
        weights = label_weights[ys]

        print(f"Total number of images: {counts.sum()}")

        print("Number of images per class:")
        for gene_id, (c, w) in enumerate(zip(counts, label_weights)):
            print(f"\t{dataset.gene_symbols[gene_id]}:\tid={gene_id}\tn={c}\tweight={w}")

        sampler = WeightedRandomSampler(weights,len(weights))

        super().__init__(dataset,batch_size=batch_size,drop_last=True,sampler=sampler,num_workers=num_workers)

class DistributedBalancedDataLoader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers):
        ys = dataset.labels
        counts = np.bincount(ys)
        label_weights = 1.0 / counts
        weights = label_weights[ys]

        print(f"Total number of images: {counts.sum()}")

        print("Number of images per class:")
        for gene_id, (c, w) in enumerate(zip(counts, label_weights)):
            print(f"\t{dataset.gene_symbols[gene_id]}:\tid={gene_id}\tn={c}\tweight={w}")

        sampler = DistributedWeightedRandomSampler(weights,len(weights),drop_last=True)

        super().__init__(dataset,batch_size=batch_size,drop_last=True,sampler=sampler,num_workers=num_workers)