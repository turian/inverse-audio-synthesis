# vicreg on synth1b1 with 3 channel pqmfs
#
# TODO:
# * Add EMA
# * Interleave pretraining and downstream


import datetime
import math

import hydra
import IPython
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
import wandb
from omegaconf import DictConfig, OmegaConf
from pynvml import *
from scipy import signal as sig
from torch import Tensor
# from torch_audiomentations import Compose, Gain, PolarityInversion
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import \
    mobilenet_v3_small  # , MobileNet_V3_Small_Weights
from tqdm.auto import tqdm


def downstream_batch(batch_num, vicreg):
    test_true_audio, test_true_params, test_true_is_train = voice(batch_num)

    test_predicted_audio_embedding = vicreg.projector(
        vicreg.backbone2(test_true_audio.unsqueeze(1))
    )
    test_predicted_audio_embedding.shape

    test_predicted_params = audio_embedding_to_params.forward(
        test_predicted_audio_embedding
    ).T

    for param_name, param_value in zip(
        voice.get_parameters().keys(), test_predicted_params
    ):
        param_name1, param_name2 = param_name
        getattr(voice, param_name1).set_parameter_0to1(param_name2, param_value)

    # TODO: Disable gradients

    voice.freeze_parameters(voice.get_parameters().keys())
    # # WHY??????
    voice.cuda()
    (
        test_predicted_audio,
        test_predicted_predicted_params,
        test_predicted_is_train,
    ) = voice(None)
    voice.unfreeze_all_parameters()

    test_true_mel = mel_spectrogram(test_true_audio)
    test_predicted_mel = mel_spectrogram(test_predicted_audio)

    mel_l1_error = torch.mean(torch.abs(test_true_mel - test_predicted_mel))
    print(mel_l1_error)

    for i in tqdm(list(range(8))):
        silence = torch.zeros(int(RATE * 0.5)).cuda()
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.login()

    # We'll generate BATCH_SIZE sounds per batch, 4 seconds each
    # TODO: On larger GPUs, use larger batch size
    synthconfig = SynthConfig(
        batch_size=BATCH_SIZE,
        reproducible=False,
        sample_rate=RATE,
        buffer_size_seconds=4.0,
    )

    voice = Voice(synthconfig=synthconfig)

    # Run on the GPU if it's available
    if torch.cuda.is_available():
        voice = voice.to("cuda")

    # Use 3 channels for RGB image (not 4 which is PQMF default)
    pqmf = PQMF(N=3).to("cuda")

    # New weights with accuracy 80.858%
    # https://pytorch.org/vision/stable/models.html
    # weights = ResNet50_Weights.IMAGENET1K_V2
    # vision_model = resnet50(weights=weights).to("cuda")

    # weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    # vision_model = mobilenet_v3_small(weights=weights).to("cuda")
    # torchvision 0.12.0 :(
    vision_model = mobilenet_v3_small(pretrained=True).to("cuda")
    # vision_model = mobilenet_v3_small(pretrained=False).to("cuda")

    ## Initialize the inference transforms
    # preprocess = weights.transforms()

    # torchvision 0.12.0 :(
    preprocess = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    parammlp = ParamMLP()
    parammlp.cuda()

    audio_embedding = AudioEmbedding(pqmf, vision_model)

    audio_embedding_to_params = AudioEmbeddingToParams()
    audio_embedding_to_params.cuda()

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    ).cuda()

    vicreg_scaler = torch.cuda.amp.GradScaler()

    run = wandb.init(
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

    # vicreg = VICReg(args=args, backbone1 = parammlp, backbone2 = parammlp).cuda()
    vicreg = VICReg(args=args, backbone1=parammlp, backbone2=audio_embedding).cuda()
    # vicreg = VICReg(args=args, backbone1 = audio_embedding, backbone2 = audio_embedding).cuda()

    # Probably could use a smarter optimizer?
    # vicreg_optimizer = optim.Adam(vicreg.parameters(), lr=0.000001)
    # vicreg_optimizer = optim.SGD(vicreg.parameters(), lr=0.0032, momentum=0.9)

    if USE_LARS:
        ## LARS is fucked in our tests. Maybe because we're not distributing and haven't mucked with the FB code enough
        vicreg_optimizer = LARS(
            vicreg.parameters(),
            lr=0,
            weight_decay=args.wd,
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

    # Only one node for now
    per_device_batch_size = BATCH_SIZE
    args.num_workers = 1

    # loader = torch.utils.data.DataLoader(
    #        dataset,
    #        batch_size=per_device_batch_size,
    #        num_workers=args.num_workers,
    ##        pin_memory=True,
    ##        sampler=sampler,
    #    )

    # for step, (audio, params, is_train) in enumerate(loader):
    for batch_num in tqdm(list(range(PRETRAIN_STEPS))):
        if batch_num % 10 == 0:
            # test
            continue
        if batch_num % 10 == 1:
            # dev
            continue

        if batch_num % 10 == 9 and (
            (batch_num - 9) % PRETRAIN_STEPS_CHECKPOINT_EVERY == 0
        ):
            # Time to checkpoint pretraining train
            batch_num_str = f"{'%010d' % batch_num}"
            vicreg_checkpoint_filename = (
                f"/tmp/vicreg_model_{utcnowstr}-{batch_num_str}.pth"
            )
            # print(vicreg_checkpoint_filename)
            torch.save(vicreg.state_dict(), vicreg_checkpoint_filename)
            artifact = wandb.Artifact(f"vicreg_model-{batch_num_str}", type="model")
            artifact.add_file(vicreg_checkpoint_filename)
            run.log_artifact(artifact)
            # run.join()

        audio, params, is_train = voice(batch_num)
        audio = audio.unsqueeze(1)
        #  audio2 = apply_augmentation(audio)

        if USE_LARS:
            lr = adjust_learning_rate(args, vicreg_optimizer, loader, step)
            wandb.log({"lars_lr": lr})
            vicreg_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                vicreg_loss = vicreg.forward(params, audio)
        else:
            vicreg_loss = vicreg.forward(params, audio)

        #  loss = vicreg(audio2, audio)
        vicreg_loss = vicreg(params, audio)
        #  loss = vicreg(params, params)
        vicreg_lossval = vicreg_loss.detach().cpu().numpy()
        if math.isnan(vicreg_lossval):
            print("NAN")
            break
        wandb.log({"vicreg_loss": vicreg_lossval})

        # loss.backward()
        # optimizer.step()

        vicreg_scaler.scale(vicreg_loss).backward()
        vicreg_scaler.step(vicreg_optimizer)
        vicreg_scaler.update()

    downstream(0, vicreg)
    wandb.finish()


if __name__ == "__main__":
    app()
