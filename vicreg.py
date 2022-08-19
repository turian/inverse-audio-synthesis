# Adapted from
# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
# We remove the backbone because we do them externally
# And we remove torch.dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import wandb


class VICReg(nn.Module):
    def __init__(self, cfg, backbone1, backbone2):
        super().__init__()
        self.cfg = cfg
        self.num_features = cfg.dim
        #        self.backbone = nn.Identity()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.embedding = cfg.dim
        #        self.backbone, self.embedding = resnet.__dict__[cfg.arch](
        #            zero_init_residual=True
        #        )
        self.projector = Projector(cfg, self.embedding)

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

        cov_x = (x.T @ x) / (self.cfg.vicreg.batch_size - 1)
        cov_y = (y.T @ y) / (self.cfg.vicreg.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.cfg.vicreg.sim_coeff * repr_loss
            + self.cfg.vicreg.std_coeff * std_loss
            + self.cfg.vicreg.cov_coeff * cov_loss
        )
        return loss, repr_loss, std_loss, cov_loss


def Projector(cfg, embedding):
    mlp_spec = f"{embedding}-{cfg.vicreg.mlp}" % cfg.dim
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


def adjust_learning_rate(cfg, optimizer, loader, step):
    max_steps = cfg.vicreg.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = cfg.vicrec.base_lr * cfg.vicreg.batch_size / 256
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
