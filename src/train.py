# src/train.py

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from model import SimpleTitanicModel
from data_module import TitanicDataModule


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Инициализируем DataModule
    dm = TitanicDataModule(
        data_path=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        train_size=cfg.data.train_size,
    )
    # Вызываем setup() для подготовки датасетов
    dm.setup()

    # Получаем input_size из DataModule
    input_size = dm.input_size

    # Инициализируем модель с правильным размером входа
    model = SimpleTitanicModel(input_size=input_size, lr=cfg.model.lr)

    # Настраиваем Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )

    # Запускаем обучение
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
