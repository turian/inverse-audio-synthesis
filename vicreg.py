# Adapted from
# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
# We remove the backbone because we do them externally
# And we remove torch.dist

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

from full_gather_layer import FullGatherLayer

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

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)

        #world_size = self.world_size
        world_size = dist.get_world_size()
        assert x.shape[0] == self.cfg.vicreg.batch_size * world_size
        assert y.shape[0] == self.cfg.vicreg.batch_size * world_size
        world_batch_size = self.cfg.vicreg.batch_size * world_size

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (world_batch_size - 1)
        cov_y = (y.T @ y) / (world_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.embeddim) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(self.embeddim)

        loss = (
            self.cfg.vicreg.sim_coeff * repr_loss
            + self.cfg.vicreg.std_coeff * std_loss
            + self.cfg.vicreg.cov_coeff * cov_loss
        )
        return loss, repr_loss, std_loss, cov_loss


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


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def exclude_bias_and_norm(p):
    return p.ndim == 1
