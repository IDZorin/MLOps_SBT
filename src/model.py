import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTitanicModel(pl.LightningModule):
    def __init__(self, input_size, lr=0.001):
        super().__init__()
        self.lr = lr
        self.layer = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = F.binary_cross_entropy(logits, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = F.binary_cross_entropy(logits, y.float())
        acc = ((logits > 0.5) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
