import torch

class DistributedWeightedRandomSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        weights,
        epoch_num_samples,
        replacement = True,
        num_replicas = None,
        rank = None,
        seed = 0,
        drop_last = False
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        self.weights=torch.FloatTensor(weights)
        self.epoch_num_samples = epoch_num_samples
        self.replacement = replacement

        super().__init__(
            torch.utils.data.TensorDataset(torch.LongTensor(range(self.epoch_num_samples))),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed+self.epoch)
        indices = torch.multinomial(input=self.weights,num_samples=self.epoch_num_samples,replacement=self.replacement,generator=g).tolist()
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)