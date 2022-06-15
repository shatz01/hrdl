import numpy as np
import pandas as pd
import torch
import torchvision

from pytorch_lightning import LightningModule, Trainer
import torchvision.models as models
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn

import pytorchvideo.models.resnet

class ResNetVideo(LightningModule):
    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3, # RGB input from Kinetics
                model_depth=50, # For the tutorial let's just use a 50 layer network
                model_num_class=2, # Kinetics has 400 classes so we need out final head to align
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                )
        self.lr = lr
        
        # self.criteria = F.cross_entropy
        # self.criteria = torch.nn.BCEWithLogitsLoss()
        self.criteria = torch.nn.BCELoss()

        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch
        x = torch.swapaxes(x, 1, 2)

        out = self(x)

        loss = self.criteria(out, torch.nn.functional.one_hot(y, self.hparams.num_classes).float())
        acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss, "acc": acc, "batch_outputs": out.clone().detach()}

    def validation_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch
        x = torch.swapaxes(x, 1, 2)

        out = self(x)
        
        # import pdb; pdb.set_trace()
        val_loss = self.criteria(out, torch.nn.functional.one_hot(y, self.hparams.num_classes).float())
        val_acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss, "val_acc": val_acc, "batch_outputs": out.clone().detach()}

    # on end of train/validation, I can print stuff like this:
    # print(f"REGULAR train loss: {train_loss} | train acc: {train_acc}")
    # print(f"REGULAR val loss: {val_loss} | val acc: {val_acc}")
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
