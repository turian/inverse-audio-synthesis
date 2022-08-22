#!/usr/bin/env python3

import hydra

# import torch.distributed as dist
from omegaconf import DictConfig
from pynvml import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights

import wandb
from audio_to_params import AudioToParams
from runsetup import runsetup
from vicreg_audio_params import VicregAudioParams


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    (
        train_batch_num_dataloader,
        val_batch_num_dataloader,
        test_batch_num_dataloader,
        logger,
    ) = runsetup(cfg)

    vicreg = VicregAudioParams.load_from_checkpoint("vicreg.ckpt", cfg=cfg)
    audio_to_params = AudioToParams(cfg, vicreg)
    audio_to_params_model_checkpoint = ModelCheckpoint(
        every_n_train_steps=cfg.audio_to_params.checkpoint_every_nbatches,
        dirpath="chkpts/",
        filename="audio_to_params-{epoch:02d}-{step:04d}",
        monitor=None,
        save_last=True,
    )
    # TODO: Remove limit_train_batches
    audio_to_params_trainer = Trainer(
        logger=logger,
        limit_train_batches=cfg.audio_to_params.limit_train_batches,
        max_epochs=1,
        # precision=cfg.precision,
        detect_anomaly=True,  # useful logs about when and where the Nan or inf anomaly happens
        accelerator=cfg.accelerator,
        strategy=cfg.strategy,
        devices=cfg.devices,
        deterministic=True,
# TODO: Other stuff from pretrain Trainer??
        callbacks=[audio_to_params_model_checkpoint],
        # callbacks = [audio_to_params_model_checkpoint, ORTCallback()],
        # Doesn't work with our CUDA version :(
        # https://github.com/Lightning-AI/lightning-bolts
        # callbacks = ORTCallback(),
    )
    #        from copy import deepcopy
    #        deepcopy(audio_to_params_trainer.callback_metrics)
    audio_to_params_trainer.fit(
        audio_to_params,  # audio_to_params_scaler, audio_to_params_optimizer,
        train_dataloaders=train_batch_num_dataloader,
    )

    audio_to_params_trainer.test(audio_to_params, dataloaders=test_batch_num_dataloader)

    if cfg.log == "wand":
        wandb.finish()


if __name__ == "__main__":
    app()
