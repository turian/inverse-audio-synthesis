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

import wandb
from audio_repr_to_params import AudioRepresentationToParams
from audioembed import AudioEmbedding
from paramembed import ParamEmbed
from pqmf import PQMF
from utils import utcnowstr
from vicreg import VICReg


def downstream_batch(
    cfg,
    device,
    batch_num,
    vicreg,
    voice,
    mel_spectrogram,
):
    with torch.no_grad():
        test_true_audio, test_true_params, test_true_is_train = voice(batch_num)

    # TODO: Tune vicreg?
    with torch.no_grad():
        # Don't use projector to embedding, just the representation from the backbone
        test_predicted_audio_representation = vicreg.backbone2(
            test_true_audio.unsqueeze(1)
        )
        # vicreg.project(vicreg.backbone2(test_true_audio.unsqueeze(1)))
        test_predicted_audio_representation.shape

    audio_repr_to_params = AudioRepresentationToParams(nparams=cfg.nparams, dim=cfg.dim)
    audio_repr_to_params = audio_repr_to_params.to(device)
    test_predicted_params = audio_repr_to_params.forward(
        test_predicted_audio_representation
    ).T

    for param_name, param_value in zip(
        voice.get_parameters().keys(), test_predicted_params
    ):
        param_name1, param_name2 = param_name
        getattr(voice, param_name1).set_parameter_0to1(param_name2, param_value)

    voice.freeze_parameters(voice.get_parameters().keys())
    # # WHY??????
    voice = voice.to(device)
    (
        test_predicted_audio,
        test_predicted_predicted_params,
        test_predicted_is_train,
    ) = voice(None)
    voice.unfreeze_all_parameters()

    test_true_mel = mel_spectrogram(test_true_audio)
    test_predicted_mel = mel_spectrogram(test_predicted_audio)

    mel_l1_error = torch.mean(torch.abs(test_true_mel - test_predicted_mel))
    return mel_l1_error

    """
    if cfg.log == "wand":
        for i in tqdm(list(range(8))):
            silence = torch.zeros(int(RATE * 0.5))
            silence = silence.to(device)
            test_true_predict_audio = torch.cat(
                [test_true_audio[i], silence, test_predicted_audio[i]]
            )
            this_test_wav_filename = f"test_{'%010d' % batch_num}_{'%03d' % i}.wav"
            this_test_wav_numpy = (
                test_true_predict_audio.unsqueeze(1).detach().cpu().numpy()
            )
            soundfile.write(this_test_wav_filename, this_test_wav_numpy, RATE)
            artifact = wandb.Artifact(this_test_wav_filename, type="model")
            artifact.add_file(vicreg_checkpoint_filename)
            run.log_artifact(artifact)
            wandb.log(
                {
                    this_test_wav_filename: wandb.Audio(
                        this_test_wav_numpy,
                        caption=this_test_wav_filename,
                        sample_rate=RATE,
                    )
                }
            )
    """

def downstream(
    cfg,
    device,
    vicreg,
    voice,
    train_batch_num_dataloader,
    val_batch_num_dataloader,
    test_batch_num_dataloader,
    mel_spectrogram,
) -> None:
    downstream_optimizer = optim.SGD(downstream.parameters(), lr=0.1)

    # One epoch training
    for downstream_train_batch_num, voice_batch_num in tqdm(
        enumerate(train_batch_num_dataloader)
    ):
        break

        assert voice_batch_num.numpy().shape == (1,)
        voice_batch_num = voice_batch_num.numpy()
        assert len(voice_batch_num) == 1
        voice_batch_num = voice_batch_num[0].item()

        if cfg.log == "wand":
            if downstream_train_batch_num % cfg.downstream.checkpoint_every_nbatches == 0:
                # Time to checkpoint pretraining train
                voice_batch_num_str = f"{'%010d' % downstream_train_batch_num}"
                downstream_checkpoint_filename = (
                    f"/tmp/downstream_model_{utcnowstr}-{downstream_train_batch_num}.pth"
                )
                # print(downstream_checkpoint_filename)
                torch.save(downstream.state_dict(), downstream_checkpoint_filename)
                artifact = wandb.Artifact(
                    f"downstream_model-{voice_batch_num_str}", type="model"
                )
                artifact.add_file(downstream_checkpoint_filename)
                run.log_artifact(artifact)
                # run.join()

        audio, params, is_train = voice(voice_batch_num)
        audio = audio.unsqueeze(1)
        #  audio2 = apply_augmentation(audio)

        if cfg.downstream.use_lars:
            lr = adjust_learning_rate(cfg, downstream_optimizer, loader, step)
            if cfg.log == "wand":
                wandb.log({"lars_lr": lr})
            downstream_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                downstream_loss = downstream.forward(params, audio)
        else:
            downstream_optimizer.zero_grad()
            downstream_loss = downstream.forward(params, audio)

        #  loss = downstream(audio2, audio)
        downstream_loss = downstream(params, audio)
        #  loss = downstream(params, params)
        downstream_lossval = downstream_loss.detach().cpu().numpy()
        if math.isnan(downstream_lossval):
            print("NAN")
            sys.stdout.flush()
            continue
        #            break
        if cfg.log == "wand":
            wandb.log({"downstream_loss": downstream_lossval})

        # loss.backward()
        # optimizer.step()

        downstream_scaler.scale(downstream_loss).backward()
        downstream_scaler.step(downstream_optimizer)
        downstream_scaler.update()
