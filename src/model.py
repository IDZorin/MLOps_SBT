import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleIrisModel(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.layer = nn.Linear(4, 3)  # 4 features -> 3 classes

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
