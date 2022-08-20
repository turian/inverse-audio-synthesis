#!/usr/bin/env python3
# vicreg on synth1b1 with 3 channel pqmfs
#
# TODO:
# * Add EMA
# * Interleave pretraining and downstream
# * multigpu

import datetime
import math
import sys

import flash.core
import hydra
import numpy as np
import pytorch_lightning as pl
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributed as dist
import torch.optim as optim
import torchaudio
import torchaudio.transforms
import torchvision
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable
from pynvml import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

# from torch_audiomentations import Compose, Gain, PolarityInversion
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights
from tqdm.auto import tqdm

import audio_repr_to_params
import wandb
from audioembed import AudioEmbedding
from paramembed import ParamEmbed
from pqmf import PQMF
from utils import utcnowstr
from vicreg import VICReg

"""
# https://stackoverflow.com/a/62508086/82733
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
"""


class VicregAudioParams(pl.LightningModule):
    def __init__(self, cfg: DictConfig, mel_spectrogram) -> None:
        super().__init__()

        self.cfg = cfg

        # Use 3 channels for RGB image (not 4 which is PQMF default)
        self.pqmf = PQMF(N=3)

        # New weights with accuracy 80.858%
        # https://pytorch.org/vision/stable/models.html
        # weights = ResNet50_Weights.IMAGENET1K_V2
        # vision_model = resnet50(weights=weights)
        # vision_model = vision_model.to(device)

        # weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        # vision_model = mobilenet_v3_small(weights=weights)
        # vision_model = vision_model.to(device)
        # torchvision 0.12.0 :(
        self.vision_model = mobilenet_v3_small(
            pretrained=cfg.vicreg.pretrained_vision_model
        )

        ## Initialize the inference transforms
        # preprocess = weights.transforms()

        # torchvision 0.12.0 :(
        self.img_preprocess = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.paramembed = ParamEmbed(
            nparams=cfg.nparams,
            dim=cfg.dim,
            hidden_norm=cfg.param_embed.hidden_norm,
            dropout=cfg.param_embed.dropout,
        )

        self.audio_repr = AudioEmbedding(
            self.pqmf,
            self.vision_model,
            img_preprocess=self.img_preprocess,
            dim=cfg.dim,
        )

        # TODO: Swap order of these everywhere?
        self.vicreg = VICReg(
            cfg=cfg, backbone1=self.paramembed, backbone2=self.audio_repr
        )
        # count_parameters(vicreg)

        # We need a new one of these every time we change the batch size,
        # which varies model to model. And might me we don't holdout correctly :(
        self.synthconfig = SynthConfig(
            batch_size=self.cfg.vicreg.batch_size,
            reproducible=self.cfg.torchsynth.reproducible,
            sample_rate=self.cfg.torchsynth.rate,
            buffer_size_seconds=self.cfg.torchsynth.buffer_size_seconds,
        )
        self.voice = Voice(synthconfig=self.synthconfig)
        # BUG: Why???????
        self.voice.to(self.device)

    def training_step(self, batch, batch_idx):
        # TODO: Try removing CPU move
        assert batch.detach().cpu().numpy().shape == (1,)
        voice_batch_num = batch.detach().cpu().numpy()
        assert len(voice_batch_num) == 1
        voice_batch_num = voice_batch_num[0].item()

        audio, params, is_train = self.voice(voice_batch_num)
        audio = audio.unsqueeze(1)
        #  audio2 = apply_augmentation(audio)

        vicreg_loss, repr_loss, std_loss, cov_loss = self.vicreg(params, audio)
        self.log("vicreg/loss", vicreg_loss)
        self.log("vicreg/repr_loss", repr_loss)
        self.log("vicreg/std_loss", std_loss)
        self.log("vicreg/cov_loss", cov_loss)

        sch = self.lr_schedulers()

        # step every N batches
        if (batch_idx + 1) % 10000 == 0:
            sch.step()

        return vicreg_loss

    def configure_optimizers(self):
        if self.cfg.vicreg.optim.name == "sgd":
            return optim.SGD(self.parameters(), **self.cfg.vicreg.optim.args)
        elif self.cfg.vicreg.optim.name == "lars":
            # TODO: Add cosine scheduler?
            # https://arxiv.org/pdf/2105.04906.pdf
            # Section 4.2: "The learning rate follows a cosine decay
            # schedule Loshchilov & Hutter (2017), starting from 0 with
            # 10 warmup epochs and with final value of 0.002."
            return flash.core.optimizers.LARS(
                self.parameters(),
                weight_decay=self.cfg.vicreg.optim.args.weight_decay,
                # https://arxiv.org/pdf/2105.04906.pdf
                # section 4.2
                lr=self.cfg.vicreg.batch_size
                / 256
                * self.cfg.vicreg.optim.args.base_lr,
            )
        else:
            assert False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.login()

    seed_everything(42, workers=True)

    # BUG: We use a batch_size of 128 for vicreg pretraining and a batch_size of
    # 4 for downstream inverse synthesis. However, we are not careful about
    # our train/test splits so test for downstream might have been used as
    # training for vicreg. I don't think this is a big deal tho.
    batch_nums = torch.tensor(list(range(cfg.num_batches)))
    # batch_num_dataset = torch.utils.data.DataSet(batch_nums)
    # batch_num_dataset = batch_num_dataset.to(device)

    ntrain_batches = int(cfg.num_batches * 0.8)
    nval_batches = int(cfg.num_batches * 0.1)
    ntest_batches = cfg.num_batches - ntrain_batches - nval_batches
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

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.torchsynth.rate,
        n_fft=cfg.mel.n_fft,
        win_length=cfg.mel.win_length,
        hop_length=cfg.mel.hop_length,
        center=cfg.mel.center,
        pad_mode=cfg.mel.pad_mode,
        power=cfg.mel.power,
        norm=cfg.mel.norm,
        onesided=cfg.mel.onesided,
        n_mels=cfg.mel.n_mels,
        mel_scale=cfg.mel.mel_scale,
    )

    # vicreg_scaler = torch.cuda.amp.GradScaler()

    vicreg = VicregAudioParams(cfg, mel_spectrogram)

    if cfg.log == "wand":
        logger = WandbLogger(
            # Set the project where this run will be logged
            project="vicreg-synth1b1-pqmfs",
            #      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #      name=f"experiment_{run}",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            # Log model checkpoints as they get created during training
            log_model="all",
        )
        logger.watch(vicreg)

    else:
        logger = None

    if cfg.vicreg.do_pretrain:
        vicreg_model_checkpoint = ModelCheckpoint(
            every_n_train_steps=cfg.vicreg.checkpoint_every_nbatches,
            dirpath="chkpts/",
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

    audio_repr_to_params.train(
        cfg=cfg,
        #        device=device,
        vicreg=vicreg,
        train_batch_num_dataloader=train_batch_num_dataloader,
        val_batch_num_dataloader=val_batch_num_dataloader,
        test_batch_num_dataloader=test_batch_num_dataloader,
        mel_spectrogram=mel_spectrogram,
    )

    if cfg.log == "wand":
        wandb.finish()


if __name__ == "__main__":
    app()
