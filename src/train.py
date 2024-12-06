import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from model import SimpleIrisModel
from data_module import IrisDataModule


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    dm = IrisDataModule(
        data_path=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        train_size=cfg.data.train_size,
    )
    model = SimpleIrisModel(lr=cfg.model.lr)
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, gpus=cfg.trainer.gpus)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
