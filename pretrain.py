#!/usr/bin/env python3

import hydra
import numpy as np

# import torch.distributed as dist
import torchaudio
import torchaudio.transforms
from omegaconf import DictConfig
from pynvml import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights

import wandb
from runsetup import runsetup


def plot_filter_range(vicreg, logger):
    # Show a plot of what the filter values are like
    # on an excerpt from music
    (audio, _rate) = torchaudio.load("daddy.wav")
    audio.to(vicreg.device)
    y = vicreg.gram(audio.unsqueeze(1)).flatten()
    y = y.detach().cpu().numpy()
    np.random.shuffle(y)
    y = y[:1000]
    x = np.arange(0, len(y))
    data = [[x, y] for (x, y) in zip(x.tolist(), sorted(y.tolist()))]
    table = wandb.Table(data=data, columns=["x", "y"])
    logger.experiment.log(
        {"audio range": wandb.plot.line(table, "x", "y", title="Filter range")}
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    (
        train_batch_num_dataloader,
        val_batch_num_dataloader,
        test_batch_num_dataloader,
        logger,
    ) = runsetup(cfg)

    vicreg = VicregAudioParams(cfg)
    if cfg.log == "wand":
        plot_filter_range(vicreg, logger)

    vicreg_model_checkpoint = ModelCheckpoint(
        every_n_train_steps=cfg.vicreg.checkpoint_every_nbatches,
        #            dirpath="chkpts/",
        filename="vicreg-{epoch:02d}-{step:04d}",
        monitor=None,
        save_last=True,
    )
    # TODO: Remove limit_train_batches
    vicreg_trainer = Trainer(
        logger=logger,
        limit_train_batches=cfg.vicreg.limit_train_batches,
        max_epochs=1,
        # precision=cfg.precision,
        detect_anomaly=True,  # useful logs about when and where the Nan or inf anomaly happens
        accelerator=cfg.accelerator,
        strategy=cfg.strategy,
        devices=cfg.devices,
        deterministic=True,
        callbacks=[vicreg_model_checkpoint],
        # callbacks = [vicreg_model_checkpoint, ORTCallback()],
        # Doesn't work with our CUDA version :(
        # https://github.com/Lightning-AI/lightning-bolts
        # callbacks = ORTCallback(),
    )
    #        from copy import deepcopy
    #        deepcopy(vicreg_trainer.callback_metrics)
    vicreg_trainer.fit(
        vicreg,  # vicreg_scaler, vicreg_optimizer,
        train_dataloaders=train_batch_num_dataloader,
    )

    if cfg.log == "wand":
        wandb.finish()


if __name__ == "__main__":
    app()
