import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models, utils
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CometLogger

pl.seed_everything(42)

class BaseLitModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return self.net(x)        

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)        


class CIFARTenLitModel(pl.LightningModule):
    """Baseline CIFAR10 Model

    Uses pre-pretrained base network and replaces the 
    last dense layer to output the CIFAR10 classes
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        backbone = models.resnet50(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone
        self.backbone.fc = nn.Linear(2048, 10)        
#         self.fc1 = nn.Linear(2048, 128)
#         self.fc2 = nn.Linear(128, 10)
#         nn.AvgPool2d()

    def forward(self, x):
        x = self.backbone(x)
#         x = F.relu(self.fc1(x))
#         out = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)



    class CIFAR10ResnetDense(BaseLitModel):
        def __init__(self, *args, **kwargs):
                super().__init__()