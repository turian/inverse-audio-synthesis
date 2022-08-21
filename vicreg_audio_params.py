import flash.core
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import wandb
import numpy as np
import torchaudio

# import torch.distributed as dist
import torch.optim as optim
import torchvision
from omegaconf import DictConfig

# from torch_audiomentations import Compose, Gain, PolarityInversion
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights
from tqdm.auto import tqdm

from audioembed import AudioEmbedding
from paramembed import ParamEmbed
from pqmf import PQMF
from vicreg import VICReg


class VicregAudioParams(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
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

        self.have_plot_filter_range = False

    def _plot_filter_range(self):
        # Show a plot of what the filter values are like
        # on an excerpt from music
        (audio, _rate) = torchaudio.load("daddy.wav")
        audio.to(self.device)
        y = self.audio_repr(audio.unsqueeze(1)).flatten()
        y = y.detach().cpu().numpy()
        x = np.arange(0, len(y))
        data = [[x, y] for (x, y) in zip(x.tolist(), sorted(y.tolist()))]
        table = wandb.Table(data=data, columns = ["x", "y"])
        self.logger.experiment.log(
            {
                "audio range": wandb.plot.line(table, "x", "y", title="Filter range")
            }
        )

    def training_step(self, batch, batch_idx):
        if not self.have_plot_filter_range:
            self._plot_filter_range()
            self.have_plot_filter_range = True

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
