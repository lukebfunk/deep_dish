import pytorch_lightning as pl
import argparse

from deep_dish.datasets import CellPatchesDataset
from deep_dish.dataloader import BalancedDataLoader
from deep_dish.train import VggModule

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

train_dataset  = CellPatchesDataset('~/deep_dish/dataset_info/train_samples.csv')
val_dataset = CellPatchesDataset('~/deep_dish/dataset_info/val_samples.csv')

assert train_dataset.n_classes==val_dataset.n_classes

num_channels = train_dataset[0][0].shape[0]
model = VggModule(input_channels=num_channels, output_classes=train_dataset.n_classes)

trainer = pl.Trainer.from_argparse_args(args)

trainer.fit(model,BalancedDataLoader(train_dataset, batch_size=16),BalancedDataLoader(val_dataset, batch_size=16))