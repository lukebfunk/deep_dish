import argparse
from itertools import product
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch

from gemelli.datasets import CellPatchesDataset
from gemelli.train import ResNetModule, VggModule

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--use_mask',type=bool,default=False)
parser.add_argument('--checkpoint_path',default=None,type=str)
parser.add_argument('--model_type',type=str,default='vgg')
parser.add_argument('--fmaps',type=int,default=32)
args = parser.parse_args()

if args.model_type=='vgg':
    module = VggModule
elif args.model_type=='resnet':
    module = ResNetModule
else:
    raise ValueError('model_type must be one of {"vgg","resnet"}')

input_size=(128,128)

test_dataset = CellPatchesDataset('~/gemelli/dataset_info/0_test_samples.csv',input_size=input_size,use_mask=args.use_mask,augment=False)

print(f'total classes: {test_dataset.n_classes}')

num_channels = test_dataset[0][0].shape[0]
print(f'input shape: {test_dataset[0][0].shape}')

# model = ResNetModule(input_channels=num_channels, output_classes=train_dataset.n_classes, fmaps=args.fmaps, input_size=input_size, learning_rate=args.lr)

model = module.load_from_checkpoint(args.checkpoint_path, input_channels=num_channels, output_classes=test_dataset.n_classes, fmaps=args.fmaps, input_size=input_size)

trainer = pl.Trainer.from_argparse_args(args)

logged_results = trainer.test(
    model,
    torch.utils.data.DataLoader(test_dataset, batch_size=32,num_workers=10)
)

cm = model.cm.cpu().numpy()

print(f'total test samples: {cm.sum().astype(int)}')

fig,ax=plt.subplots()

im = ax.imshow(cm,interpolation='nearest',cmap='magma')

cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)

text_thresh = (cm.max()+cm.min())/2.0

for i,j in product(range(test_dataset.n_classes),range(test_dataset.n_classes)):
    color = cmap_max if cm[i,j] < text_thresh else cmap_min
    ax.text(j,i,int(cm[i,j]),ha='center',va='center',color=color)

labels = [test_dataset.gene_symbols[c] for c in range(test_dataset.n_classes)]

ax.set(
    xticks=np.arange(test_dataset.n_classes),
    yticks=np.arange(test_dataset.n_classes),
    xticklabels=labels,
    yticklabels=labels,
    ylabel="True label",
    xlabel="Predicted label",
)

ax.set_ylim((test_dataset.n_classes - 0.5, -0.5))
plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode='anchor')

plt.savefig(f'figures/{args.model_type}_confusion.png',dpi=300,bbox_inches='tight')