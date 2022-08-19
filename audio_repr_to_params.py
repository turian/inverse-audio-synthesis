import math
import sys

import soundfile
import torch
import torch.nn as nn
# import torch.distributed as dist
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchsynth.config import SynthConfig
# from torch_audiomentations import Compose, Gain, PolarityInversion
from torchsynth.synth import Voice
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import \
    mobilenet_v3_small  # , MobileNet_V3_Small_Weights
from tqdm.auto import tqdm

import wandb
from utils import utcnowstr
from vicreg import VICReg


class AudioRepresentationToParams(nn.Module):
    def __init__(self, nparams, dim):
        super().__init__()
        self.nparams = nparams
        self.dim = dim
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(self.dim, self.dim)
        self.lin2 = nn.Linear(self.dim, self.dim)
        self.lin3 = nn.Linear(self.dim, self.nparams)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, DIM)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        # Want 0/1 range
        x = self.sigmoid(x)
        return x


def train_audio_to_params(
    cfg: DictConfig,
    device: torch.device,
    vicreg: VICReg,
    train_batch_num_dataloader,
    val_batch_num_dataloader,
    test_batch_num_dataloader,
    mel_spectrogram,
) -> None:
    # We need a new one of these every time we change the batch size,
    # which varies model to model. And might me we don't holdout correctly :(
    synthconfig = SynthConfig(
        batch_size=cfg.audio_repr_to_params.batch_size,
        reproducible=cfg.torchsynth.reproducible,
        sample_rate=cfg.torchsynth.rate,
        buffer_size_seconds=cfg.torchsynth.buffer_size_seconds,
    )
    voice = Voice(synthconfig=synthconfig)
    voice = voice.to(device)

    audio_repr_to_params = AudioRepresentationToParams(nparams=cfg.nparams, dim=cfg.dim)
    audio_repr_to_params = audio_repr_to_params.to(device)
    # TODO: PUt this and vicreg lr in conf
    audio_repr_to_params_optimizer = optim.SGD(
        audio_repr_to_params.parameters(), lr=0.1
    )

    #    audio_repr_to_params_scaler = torch.cuda.amp.GradScaler()

    # One epoch training
    for audio_repr_to_params_train_batch_num, voice_batch_num in tqdm(
        enumerate(train_batch_num_dataloader)
    ):
        assert voice_batch_num.numpy().shape == (1,)
        voice_batch_num = voice_batch_num.numpy()
        assert len(voice_batch_num) == 1
        #        voice_batch_num = voice_batch_num[0].item()
        voice_batch_num = 0

        with torch.no_grad():
            audio, params, is_train = voice(voice_batch_num)
            audio = audio.unsqueeze(1)
            audio.requires_grad_(True)
            #  audio2 = apply_augmentation(audio)

        # TODO: Tune vicreg?
        with torch.no_grad():
            vicreg.train()  # Disable dropout etc.
            # We take the backbone and don't do the avg pool, because we lose too many features
            # Instead we use all layers as the features :)
            predicted_audio_repr = vicreg.backbone2.features(audio)
            # Don't use projector to embedding, just the representation from the backbone
            # predicted_audio_repr = vicreg.project(vicreg.backbone2(audio))

        print(predicted_audio_repr.shape)
        return

        audio_repr_to_params_optimizer.zero_grad()
        for w in audio_repr_to_params.parameters():
            w.requires_grad_()
        predicted_audio_repr.requires_grad_(True)
        predicted_params = audio_repr_to_params.forward(predicted_audio_repr)
        predicted_params = predicted_params.T

        for param_name, param_value in zip(
            voice.get_parameters().keys(), predicted_params
        ):
            param_name1, param_name2 = param_name
            getattr(voice, param_name1).set_parameter_0to1(param_name2, param_value)

        with torch.no_grad():
            voice.freeze_parameters(voice.get_parameters().keys())
            # # WHY??????
            voice = voice.to(device)
            (
                predicted_audio,
                predicted_predicted_params,
                predicted_is_train,
            ) = voice(None)
            voice.unfreeze_all_parameters()

        true_mel = mel_spectrogram(audio)
        predicted_mel = mel_spectrogram(predicted_audio)

        mel_l1_error = torch.mean(torch.abs(true_mel - predicted_mel))

        if cfg.log == "wand":
            wandb.log(
                {
                    "audio_repr_to_params/mel_l1_error": mel_l1_error.detach()
                    .cpu()
                    .numpy()
                }
            )

        print(mel_l1_error)
        mel_l1_error.backward()
        audio_repr_to_params_optimizer.step()


#        audio_repr_to_params_scaler.scale(mel_l1_error).backward()
#        audio_repr_to_params_scaler.step(audio_repr_to_params_optimizer)
#        audio_repr_to_params_scaler.update()

train = train_audio_to_params
