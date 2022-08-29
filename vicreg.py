# Adapted from
# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICReg(nn.Module):
    def __init__(
        self,
        cfg,
        backbone_audio,
        backbone_param,
    ):
        super().__init__()
        self.cfg = cfg
        self.reprdim = cfg.dim
        self.embeddim = cfg.embeddim
        #        self.backbone = nn.Identity()
        self.backbone_audio = backbone_audio
        self.backbone_param = backbone_param
        #        self.backbone, self.embedding = resnet.__dict__[cfg.arch](
        #            zero_init_residual=True
        #        )
        self.projector = Projector(cfg, self.reprdim)

    def forward(self, audio, params):
        x = self.projector(self.backbone_audio(audio))
        y = self.projector(self.backbone_param(params))
        return x, y


def Projector(cfg, reprdim):
    mlp_spec = f"{reprdim}-{cfg.vicreg.mlp}" % cfg.embeddim
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)



def exclude_bias_and_norm(p):
    return p.ndim == 1
