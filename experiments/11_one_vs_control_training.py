import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from gemelli.datasets import CellPatchesDataset
from gemelli.dataloader import BalancedDataLoader
from gemelli.train import VggModule

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--use_mask',type=bool,default=False)
parser.add_argument('--fmaps',type=int,default=16)
parser.add_argument('--lr',type=float,default=1e-6)
parser.add_argument('--model_name',default=None)
parser.add_argument('--gene',type=str,required=True)

args = parser.parse_args()

input_size=(128,128)
pretransform_input_size=(256,256)

restrict_query = f'gene_symbol==["{args.gene}","nontargeting"]'

train_dataset  = CellPatchesDataset('~/gemelli/dataset_info/1_train_samples.csv',input_size=pretransform_input_size,use_mask=args.use_mask,augment=True,restrict_query=restrict_query)
val_dataset = CellPatchesDataset('~/gemelli/dataset_info/1_val_samples.csv',input_size=input_size,use_mask=args.use_mask,augment=False,restrict_query=restrict_query)

assert train_dataset.n_classes==val_dataset.n_classes
print(f'total classes: {train_dataset.n_classes}')
print(f'total samples: {len(train_dataset)}')
num_channels = train_dataset[0][0].shape[0]
print(f'input shape: {train_dataset[0][0].shape}')

model = VggModule(input_channels=num_channels, output_classes=train_dataset.n_classes, fmaps=args.fmaps, input_size=input_size, learning_rate=args.lr, log_images=False)

logger = TensorBoardLogger("one_vs_control_logs",name=args.model_name)

checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        filename='epoch{epoch:02d}-val_accuracy{val_accuracy:.2f}',
        auto_insert_metric_name=False,
        save_last=True,
        mode='max'
    )

trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback], max_epochs=-1)

trainer.fit(
    model,
    BalancedDataLoader(train_dataset, batch_size=16,num_workers=5),
    torch.utils.data.DataLoader(val_dataset, batch_size=16,num_workers=5)
)