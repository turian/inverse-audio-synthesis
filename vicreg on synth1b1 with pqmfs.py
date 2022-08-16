# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# vicreg on synth1b1 with 3 channel pqmfs
#
# TODO:
# * Add EMA
# * Interleave pretraining and downstream


# +
# Use the largest batchsize possible
BATCH_SIZE = 256
# BATCH_SIZE = 8
# BATCH_SIZE = 1024

# Using LARS will require some gnarly pytorch DDP rewrite or pytorch lightning etc port
# USE_LARS = True
USE_LARS = False

# Resnet output 1000 dimensional vectors
DIM = 1000

# Number of torchsynth voice params
NPARAMS = 78

# torchsynth default
RATE = 44100


# PRETRAIN_STEPS = 128
PRETRAIN_STEPS = 1024000

# PRETRAIN_STEPS_CHECKPOINT_EVERY = 10
PRETRAIN_STEPS_CHECKPOINT_EVERY = 10000
# -

# !nvidia-smi

# +
# !pip3 install torchsynth

# !pip3 install pynvml

# !pip3 install torchaudio

# !pip3 install torchvision

# !pip3 install torch-audiomentations
# -


import datetime
# +
import math

import IPython
import numpy as np
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
import torch.optim as optim
import torchaudio
import torchvision
import wandb
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

# -


def utcstr():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")


utcnowstr = utcstr()


wandb.login()

# +

# We'll generate BATCH_SIZE sounds per batch, 4 seconds each
# TODO: On larger GPUs, use larger batch size
synthconfig = SynthConfig(
    batch_size=BATCH_SIZE,
    reproducible=False,
    sample_rate=RATE,
    buffer_size_seconds=4.0,
)

# +

nvmlInit()


def gpu_mem_used():
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.free / 1024 / 1024 / 1024


# +
voice = Voice(synthconfig=synthconfig)

# Run on the GPU if it's available
if torch.cuda.is_available():
    voice = voice.to("cuda")

# +
# #!wget https://github.com/rishikksh20/multiband-hifigan/raw/master/pqmf.py


class PQMF(torch.nn.Module):
    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta

        QMF = sig.firwin(taps + 1, cutoff, window=("kaiser", beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (
                (2 * k + 1)
                * (np.pi / (2 * N))
                * (np.arange(taps + 1) - ((taps - 1) / 2))
            )  # TODO: (taps - 1) -> taps
            phase = (-1) ** k * np.pi / 4
            H[k] = 2 * QMF * np.cos(constant_factor + phase)

            G[k] = 2 * QMF * np.cos(constant_factor - phase)

        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N)).float()
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.N = N

        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, x):
        return self.analysis(x)

    def analysis(self, x):
        return F.conv1d(x, self.H, padding=self.taps // 2, stride=self.N)

    def synthesis(self, x):
        x = F.conv_transpose1d(x, self.updown_filter * self.N, stride=self.N)
        x = F.conv1d(x, self.G, padding=self.taps // 2)
        return x


# -

# Use 3 channels for RGB image (not 4 which is PQMF default)
pqmf = PQMF(N=3).to("cuda")

# +
# Based upon 32K torchsynth sounds
# If you pass white noise, you get SMALLER values: 0.7891 .. -0.6486
maxval = 1.5680482
minval = -1.6843455


# TODO: Would be smarter but trickier to use quantile scaling
def scale8(x, xmin=minval, xmax=maxval):
    xscale = (x - xmin) / (xmax - xmin) * 255
    return torch.clip(xscale, 0, 255).type(torch.cuda.ByteTensor)


#    return torch.cuda.ByteTensor(torch.clip(xscale, 0, 255))


def unscale8(x, xmin=minval, xmax=maxval):
    return x / 255.0 * (xmax - xmin) + xmin


# New weights with accuracy 80.858%
# https://pytorch.org/vision/stable/models.html
# weights = ResNet50_Weights.IMAGENET1K_V2
# vision_model = resnet50(weights=weights).to("cuda")

# weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
# vision_model = mobilenet_v3_small(weights=weights).to("cuda")
# torchvision 0.12.0 :(
vision_model = mobilenet_v3_small(pretrained=True).to("cuda")
# vision_model = mobilenet_v3_small(pretrained=False).to("cuda")

# +
## Initialize the inference transforms
# preprocess = weights.transforms()

# torchvision 0.12.0 :(
preprocess = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# +


class ParamMLP(nn.Module):
    def __init__(self):
        super().__init__()
        #    self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(NPARAMS, DIM)
        self.lin2 = nn.Linear(DIM, DIM)
        self.lin3 = nn.Linear(DIM, DIM)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x


parammlp = ParamMLP()
parammlp.cuda()


# +


class AudioEmbedding(nn.Module):
    def __init__(self, pqmf, vision_model):
        super().__init__()
        self.pqmf = pqmf
        self.vision_model = vision_model

    def forward(self, audio):
        x = audio

        z = self.pqmf(audio)

        # This will be [128, 3, 58800]
        zimg = z.reshape(-1, 3, 240, 245)
        # Convert float to unsigned bytes
        zimg8 = scale8(zimg)
        # torchvision 0.12.0
        zimg8 = zimg8.float() / 255.0

        # Apply inference preprocessing transforms
        zimg8preprocess = preprocess(zimg8)

        y = self.vision_model(zimg8preprocess)
        return y


audio_embedding = AudioEmbedding(pqmf, vision_model)

# +
# Adapted from
# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
# We remove the backbone because we do them externally
# And we remove torch.dist


class VICReg(nn.Module):
    def __init__(self, args, backbone1, backbone2):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        #        self.backbone = nn.Identity()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.embedding = 1000
        #        self.backbone, self.embedding = resnet.__dict__[args.arch](
        #            zero_init_residual=True
        #        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        #        x = self.projector(self.backbone(x))
        #        y = self.projector(self.backbone(y))
        x = self.projector(self.backbone1(x))
        y = self.projector(self.backbone2(y))
        # print(x)
        # print(y)

        repr_loss = F.mse_loss(x, y)

        #       x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #       y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        # print({"repr_loss": repr_loss.detach().cpu().numpy()})
        # print({"std_loss": std_loss.detach().cpu().numpy()})
        # print({"cov_loss": cov_loss.detach().cpu().numpy()})
        wandb.log({"repr_loss": repr_loss.detach().cpu().numpy()})
        wandb.log({"std_loss": std_loss.detach().cpu().numpy()})
        wandb.log({"cov_loss": cov_loss.detach().cpu().numpy()})

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def exclude_bias_and_norm(p):
    return p.ndim == 1


class Args(object):
    pass


args = Args()
# args.mlp = "8192-8192-8192"
args.mlp = f"1024-1024-{DIM}"
args.batch_size = BATCH_SIZE

args.sim_coeff = 25.0
args.std_coeff = 25.0
args.cov_coeff = 1.0
# args.sim_coeff = 1.0
# args.std_coeff = 0.0
# args.cov_coeff = 0.0
args.wd = 1e-6

# -

"""

# Initialize augmentation callable
apply_augmentation = Compose(
    transforms=[
        Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5,
        ),
        PolarityInversion(p=0.5)
    ]
)
"""

# +


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# +


class AudioEmbeddingToParams(nn.Module):
    def __init__(self):
        super().__init__()
        #    self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(DIM, DIM)
        self.lin2 = nn.Linear(DIM, DIM)
        self.lin3 = nn.Linear(DIM, NPARAMS)
        self.sigmoid = nn.Sigmoid()

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


audio_embedding_to_params = AudioEmbeddingToParams()
audio_embedding_to_params.cuda()


import torch
# +
import torchaudio.transforms

sample_rate = 44100
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

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

# -

# FUCKS with distributed stuff :(
"""
class VoiceIterableDataset(torch.utils.data.Dataset):
    def __init__(self, voice):
        super(VoiceIterableDataset).__init__()
        self.voice = voice

    def __getitem__(self, n):
        return self.voice(n)
    
    def __len__(self):
        return 1024 * 1024 * 1024

dataset = VoiceIterableDataset(voice)
"""


# +

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

    if batch_num % 10 == 9 and ((batch_num - 9) % PRETRAIN_STEPS_CHECKPOINT_EVERY == 0):
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


# +
# !rm *wav


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


downstream(0, vicreg)
# -


# +


wandb.finish()
