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
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights
from tqdm.auto import tqdm

import audio_repr_to_params
import wandb
from audioembed import AudioEmbedding
from paramembed import ParamEmbed
from pqmf import PQMF
from utils import utcnowstr
from vicreg import VICReg


def pretrain_vicreg(
    cfg: DictConfig, device, train_batch_num_dataloader, mel_spectrogram, wandrun
) -> None:
    # We need a new one of these every time we change the batch size,
    # which varies model to model. And might me we don't holdout correctly :(
    synthconfig = SynthConfig(
        batch_size=cfg.vicreg.batch_size,
        reproducible=cfg.torchsynth.reproducible,
        sample_rate=cfg.torchsynth.rate,
        buffer_size_seconds=cfg.torchsynth.buffer_size_seconds,
    )
    voice = Voice(synthconfig=synthconfig)
    voice = voice.to(device)

    vicreg_scaler = torch.cuda.amp.GradScaler()

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

    audio_repr = AudioEmbedding(pqmf, vision_model, img_preprocess=img_preprocess)

    # vicreg = VICReg(cfg=cfg, backbone1 = paramembed, backbone2 = paramembed)
    vicreg = VICReg(cfg=cfg, backbone1=paramembed, backbone2=audio_repr)
    # vicreg = VICReg(cfg=cfg, backbone1 = audio_repr, backbone2 = audio_repr)
    vicreg = vicreg.to(device)

    # Probably could use a smarter optimizer?
    # vicreg_optimizer = optim.Adam(vicreg.parameters(), lr=0.000001)
    # vicreg_optimizer = optim.SGD(vicreg.parameters(), lr=0.0032, momentum=0.9)

    if cfg.vicreg.use_lars:
        ## LARS is fucked in our tests. Maybe because we're not distributing and haven't mucked with the FB code enough
        vicreg_optimizer = LARS(
            vicreg.parameters(),
            lr=0,
            weight_decay=cfg.vicreg.wd,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )
    else:
        # Everything is kinda fucked besides good old SGD
        # For pretrained = True
        vicreg_optimizer = optim.SGD(vicreg.parameters(), lr=0.1)
        # vicreg_optimizer = optim.SGD(vicreg.parameters(), lr=0.032)
        ## For pretrained=False
        # vicreg_optimizer = optim.SGD(vicreg.parameters(), lr=0.01)
        # vicreg_optimizer = optim.SGD(vicreg.parameters(), lr=0.01)
        # vicreg_optimizer = optim.SGD(vicreg.parameters(), lr=0.000001)

    if cfg.vicreg.continue_from:
        if device == "cpu":
            checkpoint = torch.load(
                cfg.vicreg.continue_from, map_location=torch.device(device)
            )
        else:
            checkpoint = torch.load(cfg.vicreg.continue_from)
        vicreg.load_state_dict(checkpoint)
        vicrec = vicreg.to(device)

    # Only one node for now
    per_device_batch_size = cfg.vicreg.batch_size
    cfg.num_workers = 1

    if cfg.vicreg.do_pretrain:
        # One epoch training
        for pretrain_batch_num, voice_batch_num in tqdm(
            enumerate(train_batch_num_dataloader)
        ):
            assert voice_batch_num.numpy().shape == (1,)
            voice_batch_num = voice_batch_num.numpy()
            assert len(voice_batch_num) == 1
            voice_batch_num = voice_batch_num[0].item()

            if cfg.log == "wand":
                if pretrain_batch_num % cfg.vicreg.checkpoint_every_nbatches == 0:
                    # Time to checkpoint pretraining train
                    voice_batch_num_str = f"{'%010d' % pretrain_batch_num}"
                    vicreg_checkpoint_filename = (
                        f"/tmp/vicreg_model_{utcnowstr}-{pretrain_batch_num}.pth"
                    )
                    # print(vicreg_checkpoint_filename)
                    torch.save(vicreg.state_dict(), vicreg_checkpoint_filename)
                    artifact = wandb.Artifact(
                        f"vicreg_model-{voice_batch_num_str}", type="model"
                    )
                    artifact.add_file(vicreg_checkpoint_filename)
                    wandrun.log_artifact(artifact)
                    # run.join()

            audio, params, is_train = voice(voice_batch_num)
            audio = audio.unsqueeze(1)
            #  audio2 = apply_augmentation(audio)

            if cfg.vicreg.use_lars:
                lr = adjust_learning_rate(cfg, vicreg_optimizer, loader, step)
                if cfg.log == "wand":
                    wandb.log({"lars_lr": lr})
                vicreg_optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    vicreg_loss = vicreg.forward(params, audio)
            else:
                vicreg_optimizer.zero_grad()
                vicreg_loss = vicreg.forward(params, audio)

            #  loss = vicreg(audio2, audio)
            vicreg_loss = vicreg(params, audio)
            #  loss = vicreg(params, params)
            vicreg_lossval = vicreg_loss.detach().cpu().numpy()
            if math.isnan(vicreg_lossval):
                print("NAN")
                sys.stdout.flush()
                continue
            #            break
            if cfg.log == "wand":
                wandb.log({"vicreg_loss": vicreg_lossval})

            # loss.backward()
            # optimizer.step()

            vicreg_scaler.scale(vicreg_loss).backward()
            vicreg_scaler.step(vicreg_optimizer)
            vicreg_scaler.update()

    return vicreg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.login()

    # Run on the GPU if it's available
    # TODO: multigpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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
    mel_spectrogram = mel_spectrogram.to(device)

    vicreg_scaler = torch.cuda.amp.GradScaler()

    if cfg.log == "wand":
        wandrun = wandb.init(
            # Set the project where this run will be logged
            project="vicreg-synth1b1-pqmfs",
            #      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #      name=f"experiment_{run}",
            #      # Track hyperparameters and run metadata
            #      config={
            #      "learning_rate": 0.02,
            #      "architecture": "CNN",
            #      "dataset": "CIFAR-100",
            #      "epochs": 10,
            #      }
        )
    else:
        wandrun = None

    vicreg = pretrain_vicreg(
        cfg, device, voice, train_batch_num_dataloader, mel_spectrogram, wandrun
    )

    audio_repr_to_params.train(
        cfg=cfg,
        device=device,
        vicreg=vicreg,
        voice=voice,
        train_batch_num_dataloader=train_batch_num_dataloader,
        val_batch_num_dataloader=val_batch_num_dataloader,
        test_batch_num_dataloader=test_batch_num_dataloader,
        mel_spectrogram=mel_spectrogram,
    )

    if cfg.log == "wand":
        wandb.finish()


if __name__ == "__main__":
    app()
