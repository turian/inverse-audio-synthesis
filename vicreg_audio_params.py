import math
from typing import List

import flash.core
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# import torch.distributed as dist
import torch.optim as optim
import torchaudio
import torchvision
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

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


# Adapted from https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py#L10-L120
# But starts the cosine curve at 0 not 1
# and warms up to warmup_end_lr_factor * base_lr
class LinearWarmupCosineAnnealingLRSlowStart(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        warmup_end_lr_factor: float = 0.1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr_factor = warmup_end_lr_factor
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr * self.warmup_end_lr_factor - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return [base_lr for base_lr in self.base_lrs]
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


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

    def _step(self, name, batch, batch_idx):
        # TODO: Try removing CPU move
        assert batch.detach().cpu().numpy().shape == (1,)
        voice_batch_num = batch.detach().cpu().numpy()
        assert len(voice_batch_num) == 1
        voice_batch_num = voice_batch_num[0].item()

        audio, params, is_train = self.voice(voice_batch_num)
        audio = audio.unsqueeze(1)
        #  audio2 = apply_augmentation(audio)

        vicreg_loss, repr_loss, std_loss, cov_loss = self.vicreg(
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
        if self.cfg.vicreg.scheduler.name == "LinearWarmupCosineAnnealingLRSlowStart":
            scheduler = LinearWarmupCosineAnnealingLRSlowStart(
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
