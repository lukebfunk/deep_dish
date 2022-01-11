import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

class BalancedDataLoader(DataLoader):
    def __init__(self,dataset,batch_size):
        ys = dataset.metadata['gene_id']
        counts = np.bincount(ys)
        label_weights = 1.0 / counts
        weights = label_weights[ys]

        print("Number of images per class:")
        for gene_id, (c, w) in enumerate(zip(counts, label_weights)):
            print(f"\t{dataset.gene_symbols[gene_id]}:\tn={c}\tweight={w}")

        sampler = WeightedRandomSampler(weights,len(weights))

        super().__init__(dataset,batch_size=batch_size,drop_last=True,sampler=sampler)