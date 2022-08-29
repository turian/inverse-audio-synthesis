import math
from typing import List

import flash.core
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F

# import torch.distributed as dist
import torch.optim as optim
import torchaudio
import torchvision
from nnAudio import features
from omegaconf import DictConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import Optimizer

# from torch_audiomentations import Compose, Gain, PolarityInversion
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights
from tqdm.auto import tqdm

import wandb
from audioembed import AudioEmbedding
from paramembed import ParamEmbed
from pqmf import PQMF
from vicreg import VICReg


class VicregAudioParams(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # Use 3 channels for RGB image (not 4 which is PQMF default)
        self.gram = PQMF(N=3)

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
            self.gram,
            self.vision_model,
            img_preprocess=self.img_preprocess,
            dim=cfg.dim,
        )

        # TODO: Swap order of these everywhere?
        self.vicreg = VICReg(
            cfg=cfg, backbone_audio=self.audio_repr, backbone_param=self.paramembed
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

    def forward(self, audio, params):
        return self.vicreg(audio=audio, params=params)

    def losses(self, audio, params):
        x, y = self.forward(audio, params)

        # BUG: Why isn't this AFTER the all_gather?
        # (But that's how it's done in the original FB code)
        repr_loss = F.mse_loss(x, y)

        x = self.all_gather(x, sync_grads=True)
        y = self.all_gather(y, sync_grads=True)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        y = y.view(y.shape[0] * y.shape[1], y.shape[2])

        # world_size = self.world_size
        world_size = dist.get_world_size()
        assert x.shape[0] == self.cfg.vicreg.batch_size * world_size
        assert y.shape[0] == self.cfg.vicreg.batch_size * world_size
        world_batch_size = self.cfg.vicreg.batch_size * world_size

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (world_batch_size - 1)
        cov_y = (y.T @ y) / (world_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.vicreg.embeddim) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(self.vicreg.embeddim)

        loss = (
            self.cfg.vicreg.sim_coeff * repr_loss
            + self.cfg.vicreg.std_coeff * std_loss
            + self.cfg.vicreg.cov_coeff * cov_loss
        )
        return loss, repr_loss, std_loss, cov_loss

    def _step(self, name, batch, batch_idx):
        # TODO: Try removing CPU move
        assert batch.detach().cpu().numpy().shape == (1,)
        voice_batch_num = batch.detach().cpu().numpy()
        assert len(voice_batch_num) == 1
        voice_batch_num = voice_batch_num[0].item()

        audio, params, is_train = self.voice(voice_batch_num)
        audio = audio.unsqueeze(1)
        #  audio2 = apply_augmentation(audio)

        vicreg_loss, repr_loss, std_loss, cov_loss = self.losses(
            audio=audio, params=params
        )
        self.log(f"vicreg/{name}/loss", vicreg_loss, sync_dist=True)
        self.log(f"vicreg/{name}/repr_loss", repr_loss, sync_dist=True)
        self.log(f"vicreg/{name}/std_loss", std_loss, sync_dist=True)
        self.log(f"vicreg/{name}/cov_loss", cov_loss, sync_dist=True)

        return vicreg_loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("validation", batch, batch_idx)

    def configure_optimizers(self):
        if self.cfg.vicreg.optim.name == "sgd":
            optim = optim.SGD(self.parameters(), lr=self.cfg.vicreg.optim.args.lr)
        elif self.cfg.vicreg.optim.name == "lars":
            optim = flash.core.optimizers.LARS(
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
        # TODO: VICReg and others actually warm-up to 1/2000th the max LR
        # in the first 10 epochs :\
        # Not linear warmup to the actual LR max
        if self.cfg.vicreg.scheduler.name == "LinearWarmupCosineAnnealingLR":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optim, **self.cfg.vicreg.scheduler.args
            )
        else:
            assert False
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                # Updates after a optimizer update
                "interval": "step",
                # How many steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every step.
                "frequency": 1,
            },
        }

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
