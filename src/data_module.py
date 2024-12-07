# src/data_module.py

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np


class TitanicDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)

        # Предобработка данных
        df = df.ffill()  # Исправлено предупреждение

        # Автоматически определяем категориальные столбцы
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Применение one-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Целевая переменная
        self.y = df["survived"].values.astype("int64")

        # Признаки
        self.X = df.drop("survived", axis=1).values.astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


class TitanicDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, train_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_size = train_size
        self.input_size = None  # Для хранения размера входа

    def setup(self, stage=None):
        dataset = TitanicDataset(self.data_path)
        self.train_dataset = torch.utils.data.Subset(dataset, range(self.train_size))
        self.val_dataset = torch.utils.data.Subset(
            dataset, range(self.train_size, len(dataset))
        )
        # Определяем input_size после загрузки данных
        self.input_size = self.train_dataset.dataset.X.shape[1]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
