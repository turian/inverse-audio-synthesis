# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# evaluate_audio_representations
# -
# %cd inverse-audio-synthesis


TEST_BATCH_SIZE = 16
PREDICT_BATCH_SIZE = 1024
#PREDICT_BATCH_SIZE = 128


import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf


# %load_ext autoreload
# %autoreload 2


import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf

from omegaconf import OmegaConf
cfg = OmegaConf.load('conf/config.yaml')

# +

import datetime
import math
import sys

import hydra
import numpy as np
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
from pynvml import *
from pytorch_lightning.lite import LightningLite
from torch import Tensor
# from torch_audiomentations import Compose, Gain, PolarityInversion
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import \
    mobilenet_v3_small  # , MobileNet_V3_Small_Weights
from tqdm.auto import tqdm

import audio_repr_to_params
import wandb
from audioembed import AudioEmbedding
from paramembed import ParamEmbed
from pqmf import PQMF
from utils import utcnowstr
from vicreg import VICReg

# +
# #!pip3 install auraloss
#import auraloss

# +

# Run on the GPU if it's available
# TODO: multigpu
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
# We'll generate cfg.batch_size sounds per batch, 4 seconds each
# TODO: On larger GPUs, use larger batch size
synthconfig_test = SynthConfig(
    batch_size=TEST_BATCH_SIZE,
    reproducible=cfg.torchsynth.reproducible,
    sample_rate=cfg.torchsynth.rate,
    buffer_size_seconds=cfg.torchsynth.buffer_size_seconds,
)

# We'll generate cfg.batch_size sounds per batch, 4 seconds each
# TODO: On larger GPUs, use larger batch size
synthconfig_predict = SynthConfig(
    batch_size=PREDICT_BATCH_SIZE,
    reproducible=cfg.torchsynth.reproducible,
    sample_rate=cfg.torchsynth.rate,
    buffer_size_seconds=cfg.torchsynth.buffer_size_seconds,
)

voice_test = Voice(synthconfig=synthconfig_test).to(device)
voice_predict = Voice(synthconfig=synthconfig_predict).to(device)

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
mel_spectrogram = mel_spectrogram.to(device)

# -

checkpoint = torch.load("vicreg.pth", map_location=torch.device('cpu'))

# +

# Use 3 channels for RGB image (not 4 which is PQMF default)
pqmf = PQMF(N=3)
pqmf = pqmf.to(device)

# New weights with accuracy 80.858%
# https://pytorch.org/vision/stable/models.html
# weights = ResNet50_Weights.IMAGENET1K_V2
# vision_model = resnet50(weights=weights)
# vision_model = vision_model.to(device)

# weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
# vision_model = mobilenet_v3_small(weights=weights)
# vision_model = vision_model.to(device)
# torchvision 0.12.0 :(
vision_model = mobilenet_v3_small(pretrained=cfg.vicreg.pretrained_vision_model)
vision_model = vision_model.to(device)

## Initialize the inference transforms
# preprocess = weights.transforms()

# torchvision 0.12.0 :(
img_preprocess = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

paramembed = ParamEmbed(nparams=cfg.nparams, dim=cfg.dim)
paramembed = paramembed.to(device)

audio_repr = AudioEmbedding(pqmf, vision_model, img_preprocess=img_preprocess).to(device)
vicreg = VICReg(cfg=cfg, backbone1=paramembed, backbone2=audio_repr).to(device)
# -

vicreg.load_state_dict(checkpoint)



voice_batch_num = 0

# +
test_audio, test_params, is_train = voice_test(0)
test_audio = test_audio.unsqueeze(1)

with torch.no_grad():
    vicreg.eval()
#    test_audio_repr = vicreg.backbone2(test_audio)
    test_audio_repr = vicreg.projector(vicreg.backbone2(test_audio))
# -

import matplotlib.pyplot

# +
#mrstft = auraloss.freq.MultiResolutionSTFTLoss()

import IPython.display

# +
#TODO: Remove dropout from downstream
# -

# !pip3 install --upgrade soundfile

# +
# !rm vicreg-match*wav
min_dist = [999] * TEST_BATCH_SIZE
import copy
min_dists = [[] for i in range(TEST_BATCH_SIZE)]
print(min_dists)
import matplotlib.pyplot as plt

for batch in tqdm(list(range(1, 1000000))):
    new_audio, new_params, new_train = voice_predict(batch)
    new_audio = new_audio.unsqueeze(1)

    with torch.no_grad():
        vicreg.eval()
#        new_audio_repr = vicreg.backbone2(new_audio)
        new_audio_repr = vicreg.projector(vicreg.backbone2(new_audio))

        dists = torch.cdist(test_audio_repr, new_audio_repr)

    for i in range(TEST_BATCH_SIZE):
        #print("min_dist", min_dist)
        nearest_neighbor = torch.argsort(dists[i])[0].cpu().numpy()
        nearest_dist = dists[i, nearest_neighbor].cpu()
        if nearest_dist < min_dist[i]:
            silence = np.zeros((1, int(44100 * 0.5)))
#            print("sample", i, nearest_dist.numpy(), #mrstft(test_audio[i], new_audio[nearest_neighbor]).numpy(),
#                  nearest_neighbor, 'batch', batch)
            min_dist[i] = nearest_dist.cpu().numpy().item()
            stacked_audio = np.hstack([test_audio[i].cpu(), silence, new_audio[nearest_neighbor].cpu()])
            outfile = "vicreg-match-dist%05.2f-sample%d-batch%06d.wav" % (min_dist[i], i, batch)
            soundfile.write(outfile, stacked_audio.T, 44100)
            min_dists[i].append((batch, min_dist[i]))
#            print("new min_dists", min_dists)

            if batch > 100:
                for line in min_dists:
                    batchs = [batch for batch, val in line]
                    vals = [val for batch, val in line]
                    assert tuple(vals) == tuple(sorted(tuple(vals), reverse=True)), "%s vs %s" % (tuple(vals), tuple(sorted(tuple(vals))))
                    plt.plot(batchs, vals)
                plt.title("%d" % batch)
                plt.savefig('vicreg-match-curve-%06d.png' % batch)
                plt.show()
        #            display(IPython.display.Audio(stacked_audio, rate=44100))
# -




