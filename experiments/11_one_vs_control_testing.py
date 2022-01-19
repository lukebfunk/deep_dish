import argparse
from glob import glob
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import re
import torch

from gemelli.datasets import CellPatchesDataset
from gemelli.dataloader import BalancedDataLoader
from gemelli.train import VggModule

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--use_mask',type=bool,default=False)
parser.add_argument('--fmaps',type=int,default=16)
# parser.add_argument('--lr',type=float,default=1e-6)
# parser.add_argument('--model_name',default=None)
parser.add_argument('--gene',type=str,required=True)

args = parser.parse_args()

input_size=(128,128)
pretransform_input_size=(256,256)

if args.gene == 'DTL':
    version = 2
else:
    version = 0

restrict_query = f'gene_symbol==["{args.gene}","nontargeting"]'

test_dataset = CellPatchesDataset('~/gemelli/dataset_info/1_test_samples.csv',input_size=input_size,use_mask=args.use_mask,augment=False,restrict_query=restrict_query)

labels = [test_dataset.gene_symbols[c] for c in range(test_dataset.n_classes)]

print(f'total classes: {test_dataset.n_classes}')
print(f'total samples: {len(test_dataset)}')
num_channels = test_dataset[0][0].shape[0]
print(f'input shape: {test_dataset[0][0].shape}')

checkpoint = glob(f'one_vs_control_logs/{args.gene}/version_{version}/checkpoints/epoch*.ckpt')[0]

model = VggModule.load_from_checkpoint(
    checkpoint,
    input_channels=num_channels,
    output_classes=test_dataset.n_classes,
    fmaps=args.fmaps,
    input_size=input_size,
    # learning_rate=args.lr,
    cm_normalization=None,
    log_images=False
)

trainer = pl.Trainer.from_argparse_args(args)

_ = trainer.test(
    model,
    torch.utils.data.DataLoader(test_dataset, batch_size=16,num_workers=5)
)

cm = model.cm.cpu().numpy().astype(int)

df_cm = pd.DataFrame(cm,index=labels,columns=labels).rename_axis(index='true',columns='predicted')

accuracy = cm[np.eye(test_dataset.n_classes,dtype=bool)].sum()/cm.sum()

checkpoint_name = re.search(r'.*checkpoints/(.+).ckpt',checkpoint).group(1)

with open('../results/one_vs_control/test_accuracy.csv','a') as f:
    f.write(f'{args.gene}\t{accuracy:.4f}\t{checkpoint_name}\n')

df_cm.to_csv(f'../results/one_vs_control/{args.gene}_test_confusion.{checkpoint_name}.csv')