import argparse
from glob import glob
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import re
import torch

from gemelli.datasets import CellPatchesDataset
from gemelli.train import VggModule

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

input_size=(128,128)

test_dataset = CellPatchesDataset('~/gemelli/dataset_info/1_test_samples.csv',input_size=input_size,use_mask=False,augment=False)
labels = [test_dataset.gene_symbols[c] for c in range(test_dataset.n_classes)]
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256,num_workers=48)

versions = [9,1,1]
fmaps_ = [80,96,128]

for fmaps,version in zip(fmaps_,versions):

    checkpoint = glob(f'logs/vgg_{fmaps}_128class/version_{version}/checkpoints/epoch*.ckpt')[0]

    print(f'total classes: {test_dataset.n_classes}')

    num_channels = test_dataset[0][0].shape[0]
    print(f'input shape: {test_dataset[0][0].shape}')

    model = VggModule.load_from_checkpoint(
        checkpoint,
        input_channels=num_channels,
        output_classes=test_dataset.n_classes,
        fmaps=fmaps,
        input_size=input_size,
        cm_normalization=None
    )

    trainer = pl.Trainer.from_argparse_args(args)

    logged_results = trainer.test(
        model,
        test_dataloader
    )

    cm = model.cm.cpu().numpy().astype(int)

    df_cm = pd.DataFrame(cm,index=labels,columns=labels).rename_axis(index='true',columns='predicted')

    checkpoint_name = re.search(r'.*checkpoints/(.+).ckpt',checkpoint).group(1)

    df_cm.to_csv(f'../results/vgg_{fmaps}_128class_test_confusion.{checkpoint_name}.csv')
