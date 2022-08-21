#!/usr/bin/env python3

import numpy as np

import os
import os.path

import hydra
import torch

# import torch.distributed as dist
import torchaudio
import torchaudio.transforms
from omegaconf import DictConfig, OmegaConf
from pynvml import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights

import wandb
from audio_to_params import AudioToParams
from vicreg_audio_params import VicregAudioParams

@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #os.environ["WANDB_CACHE_DIR"] = "/fsx/turian/.cache"
    wandb.login()

    seed_everything(42, workers=True)

    # BUG: We use a batch_size of 128 for vicreg pretraining and a batch_size of
    # 4 for downstream inverse synthesis. However, we are not careful about
    # our train/test splits so test for downstream might have been used as
    # training for vicreg. I don't think this is a big deal tho.
    batch_nums = torch.tensor(list(range(cfg.num_batches)))
    # batch_num_dataset = torch.utils.data.DataSet(batch_nums)
    # batch_num_dataset = batch_num_dataset.to(device)

    ntest_batches = cfg.ntest_batches
    ntrain_batches = int((cfg.num_batches - ntest_batches) * 0.9)
    # ntest_batches = cfg.num_batches - ntrain_batches - nval_batches
    nval_batches = cfg.num_batches - ntrain_batches - ntest_batches
    (
        train_batch_num_dataset,
        val_batch_num_dataset,
        test_batch_num_dataset,
    ) = torch.utils.data.random_split(
        batch_nums,
        [ntrain_batches, nval_batches, ntest_batches],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_batch_num_dataloader = torch.utils.data.DataLoader(train_batch_num_dataset)
    val_batch_num_dataloader = torch.utils.data.DataLoader(val_batch_num_dataset)
    test_batch_num_dataloader = torch.utils.data.DataLoader(test_batch_num_dataset)

    if cfg.log == "wand":
        # if not os.path.exists("/tmp/turian-wandb/wandb/"):
        #   os.makedirs("/tmp/turian-wandb/wandb/", exist_ok=True)
        logger = WandbLogger(
            # Set the project where this run will be logged
            project="vicreg-synth1b1-pqmfs",
            #      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #      name=f"experiment_{run}",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            # Log model checkpoints as they get created during training
            log_model="all",
            # save_dir="/tmp/turian-wandb",
        )
        # We don't use gradients much and the use a lot of logging space
        # logger.watch(vicreg)

        #plot_filter_range(vicreg, logger)
    else:
        logger = None

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

    audio_to_params_trainer.test(
        audio_to_params, dataloaders=test_batch_num_dataloader
    )

    if cfg.log == "wand":
        wandb.finish()


if __name__ == "__main__":
    app()
