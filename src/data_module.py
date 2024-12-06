import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from linalg_core import LinearAlgebra


class IrisDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        self.X = df.iloc[:, :-1].values.astype("float32")
        self.y = df.iloc[:, -1].values.astype("int64")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # shape (4,)
        y = self.y[idx]

        # Преобразуем x в матрицу 2x2
        x_2d = x.reshape(2, 2)
        # Используем функцию для транспонирования
        x_2d_t = LinearAlgebra.transpose(x_2d)
        # Превращаем обратно в вектор
        x = torch.from_numpy(x_2d_t.flatten())

        y = torch.tensor(y, dtype=torch.long)

        return x, y


class IrisDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, train_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_size = train_size

    def setup(self, stage=None):
        dataset = IrisDataset(self.data_path)
        self.train_dataset = torch.utils.data.Subset(dataset, range(self.train_size))
        self.val_dataset = torch.utils.data.Subset(
            dataset, range(self.train_size, len(dataset))
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
