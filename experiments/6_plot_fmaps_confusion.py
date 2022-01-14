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
args = parser.parse_args()

input_size=(128,128)

test_dataset = CellPatchesDataset('~/gemelli/dataset_info/0_test_samples.csv',input_size=input_size,use_mask=False,augment=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,num_workers=10)

fig,axes=plt.subplots(2,3,figsize=(12.8,9.6))

model_types = ['vgg','vgg','vgg','resnet','resnet','resnet']

checkpoints = [
    'logs/vgg_16/version_0/checkpoints/last.ckpt',
    'logs/vgg_32/version_0/checkpoints/last.ckpt',
    'logs/vgg_64/version_0/checkpoints/last.ckpt',
    'logs/resnet_16/version_1/checkpoints/last.ckpt',
    'logs/resnet_32/version_1/checkpoints/last.ckpt',
    'logs/resnet_64/version_0/checkpoints/last.ckpt'
]

fmaps_ = [16,32,64,16,32,64]

for model_type,fmaps,checkpoint,ax in zip(model_types,fmaps_,checkpoints,axes.flatten()):

    if model_type=='vgg':
        module = VggModule
    elif model_type=='resnet':
        module = ResNetModule
    else:
        raise ValueError('model_type must be one of {"vgg","resnet"}')

    print(f'total classes: {test_dataset.n_classes}')

    num_channels = test_dataset[0][0].shape[0]
    print(f'input shape: {test_dataset[0][0].shape}')

    # model = ResNetModule(input_channels=num_channels, output_classes=train_dataset.n_classes, fmaps=args.fmaps, input_size=input_size, learning_rate=args.lr)

    model = module.load_from_checkpoint(checkpoint, input_channels=num_channels, output_classes=test_dataset.n_classes, fmaps=fmaps, input_size=input_size)

    trainer = pl.Trainer.from_argparse_args(args)

    logged_results = trainer.test(
        model,
        test_dataloader
    )

    cm = model.cm.cpu().numpy()

    print(f'total test samples: {len(test_dataset)}')

    im = ax.imshow(cm,interpolation='nearest',cmap='magma')

    cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)

    text_thresh = (cm.max()+cm.min())/2.0

    for i,j in product(range(test_dataset.n_classes),range(test_dataset.n_classes)):
        color = cmap_max if cm[i,j] < text_thresh else cmap_min
        ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',color=color,fontsize=7)

    labels = [test_dataset.gene_symbols[c] for c in range(test_dataset.n_classes)]
    ax.set(
        xticks=np.arange(test_dataset.n_classes),
        yticks=np.arange(test_dataset.n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )
    if fmaps==16:
        ax.set_ylabel("true label")
    if model_type=="resnet":
        ax.set_xlabel("predicted label")

    ax.set_ylim((test_dataset.n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode='anchor',ha='right')
    ax.tick_params(axis='x',pad=7)
    ax.set_title(f'{model_type}, {fmaps} fmaps')

plt.savefig(f'figures/6_plot_fmaps_test_confusion_normalized.png',dpi=300,bbox_inches='tight')