from pytorch_lightning import LightningModule
from funlib.learn.torch.models import Vgg2D
import torch
import torchmetrics


class VggModule(LightningModule):
    def __init__(self, input_channels, output_classes, input_size=(256,256)):
        super().__init__()
        self.model = Vgg2D(input_size=input_size,input_fmaps=input_channels,output_classes=output_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self,x):
        x = self.model(x)
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        self.log("val_loss",loss, prog_bar=True,on_epoch=True)
        y_hat = self.softmax(logits)
        self.accuracy(y_hat,y)
        self.log("val_accuracy",self.accuracy,prog_bar=True,on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
        