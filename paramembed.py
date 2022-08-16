import torch.nn as nn
from torch import Tensor


class ParamEmbed(nn.Module):
    def __init__(self, nparams, dim):
        super().__init__()
        self.nparams = nparams
        self.dim = dim
        #    self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(self.nparams, self.dim)
        self.lin2 = nn.Linear(self.dim, self.dim)
        self.lin3 = nn.Linear(self.dim, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x
