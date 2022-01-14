from funlib.learn.torch.models import Vgg2D, ResNet
from pytorch_lightning import LightningModule
import torch
import torchmetrics


class VggModule(LightningModule):
    def __init__(self, input_channels, output_classes, fmaps=32, input_size=(256,256), learning_rate=1e-5):
        super().__init__()
        self.model = Vgg2D(input_size=input_size,fmaps=fmaps,output_classes=output_classes,input_fmaps=input_channels)
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.weighted_accuracy = torchmetrics.Accuracy(average='weighted',num_classes=output_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.learning_rate = learning_rate
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=output_classes,compute_on_step=False,normalize='true')
    
    def forward(self,x):
        x = self.model(x)
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        self.log("train_loss",loss)
        y_hat = self.softmax(logits)
        if self.global_step%200 == 0:
            log_images =  x[:8,:3] - torch.amin(x[:8,:3],dim=(0,2,3),keepdim=True)
            log_images = log_images / torch.amax(log_images,dim=(0,2,3),keepdim=True)
            self.logger.experiment.add_images("input",log_images)
            self.logger.experiment.add_text("labels",', '.join(map(str,y[:8].cpu().numpy())))
            self.logger.experiment.add_text("predictions",', '.join(map(str,y_hat[:8].argmax(axis=1).cpu().numpy())))
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        self.log("val_loss",loss, prog_bar=True)
        y_hat = self.softmax(logits)
        self.accuracy(y_hat,y)
        self.weighted_accuracy(y_hat,y)
        self.log("val_accuracy",self.accuracy,prog_bar=True)
        self.log("val_weighted_accuracy",self.weighted_accuracy,prog_bar=True)

    def test_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        y_hat = self.softmax(logits)
        self.confusion_matrix(y_hat,y)
    
    def test_epoch_end(self, outputs):
        self.cm = self.confusion_matrix.compute()
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class ResNetModule(LightningModule):
    def __init__(self,input_channels,output_classes,fmaps=32,input_size=(256,256),learning_rate=1e-5):
        super().__init__()
        self.model = ResNet(input_size=input_size,output_classes=output_classes,fmaps=fmaps,input_fmaps=input_channels)
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.weighted_accuracy = torchmetrics.Accuracy(average='weighted',num_classes=output_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.learning_rate = learning_rate
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=output_classes,compute_on_step=False,normalize='true')

    def forward(self,x):
        x = self.model(x)
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        self.log("train_loss",loss)
        y_hat = self.softmax(logits)
        if self.global_step%200 == 0:
            log_images =  x[:8,:3] - torch.amin(x[:8,:3],dim=(0,2,3),keepdim=True)
            log_images = log_images / torch.amax(log_images,dim=(0,2,3),keepdim=True)
            self.logger.experiment.add_images("input",log_images)
            self.logger.experiment.add_text("labels",', '.join(map(str,y[:8].cpu().numpy())))
            self.logger.experiment.add_text("predictions",', '.join(map(str,y_hat[:8].argmax(axis=1).cpu().numpy())))
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        self.log("val_loss",loss, prog_bar=True)
        y_hat = self.softmax(logits)
        self.accuracy(y_hat,y)
        self.weighted_accuracy(y_hat,y)
        self.log("val_accuracy",self.accuracy,prog_bar=True)
        self.log("val_weighted_accuracy",self.weighted_accuracy,prog_bar=True)

    def test_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        y_hat = self.softmax(logits)
        self.confusion_matrix(y_hat,y)
    
    def test_epoch_end(self, outputs):
        self.cm = self.confusion_matrix.compute()
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer