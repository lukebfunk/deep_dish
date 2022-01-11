from pytorch_lightning import LightningModule
from funlib.learn.torch.models import Vgg2D
import torch


class VggModule(LightningModule):
    def __init__(self, input_channels, output_classes):
        super().__init__()
        self.model = Vgg2D(input_size=(256,256),input_fmaps=input_channels,output_classes=output_classes)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self,x):
        x = self.model(x)
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)

        loss = self.loss(logits,y)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
        