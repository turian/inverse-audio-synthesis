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

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# from torch_audiomentations import Compose, Gain, PolarityInversion
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

import torchaudio

from tqdm.auto import tqdm

import wandb
from audioembed import AudioEmbedding
from paramembed import ParamEmbed
from vicreg import VICReg


class VicregAudioParams(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

#        self.audio_model = torchaudio.models.hubert_pretrain_xlarge()
#        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = torchaudio.models.wav2vec2.utils.import_huggingface_model(
                model
        )

        self.paramembed = ParamEmbed(
            nparams=cfg.nparams,
            dim=cfg.dim,
            hidden_norm=cfg.param_embed.hidden_norm,
            dropout=cfg.param_embed.dropout,
        )

        self.audio_repr = AudioEmbedding(
            self.audio_model,
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

    def training_step(self, batch, batch_idx):
        # TODO: Try removing CPU move
        assert batch.detach().cpu().numpy().shape == (1,)
        voice_batch_num = batch.detach().cpu().numpy()
        assert len(voice_batch_num) == 1
        voice_batch_num = voice_batch_num[0].item()

        audio, params, is_train = self.voice(voice_batch_num)
        #  audio2 = apply_augmentation(audio)

        vicreg_loss, repr_loss, std_loss, cov_loss = self.vicreg(
            audio=audio, params=params
        )
        self.log("vicreg/loss", vicreg_loss)
        self.log("vicreg/repr_loss", repr_loss)
        self.log("vicreg/std_loss", std_loss)
        self.log("vicreg/cov_loss", cov_loss)

        return vicreg_loss

    def configure_optimizers(self):
        if self.cfg.vicreg.optim.name == "sgd":
            return optim.SGD(self.parameters(), lr=self.cfg.vicreg.optim.args.lr)
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
