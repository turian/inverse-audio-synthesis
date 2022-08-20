import torch.nn as nn
from torch import Tensor


class ParamEmbed(nn.Module):
    def __init__(self, nparams, dim, hidden_norm, dropout):
        super().__init__()
        self.nparams = nparams
        self.dim = dim
        #    self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(self.nparams, self.dim)
        if hidden_norm == "nn.BatchNorm1d":
            self.norm1 = nn.BatchNorm1d(self.dim)
        elif hidden_norm == "nn.Identity":
            self.norm1 = nn.Identity()
        else:
            assert False
        self.do1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(self.dim, self.dim)
        if hidden_norm == "nn.BatchNorm1d":
            self.norm2 = nn.BatchNorm1d(self.dim)
        elif hidden_norm == "nn.Identity":
            self.norm2 = nn.Identity()
        else:
            assert False
        self.do2 = nn.Dropout(dropout)
        self.lin3 = nn.Linear(self.dim, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.norm1(x)
        x = self.do1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.norm2(x)
        x = self.do2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x
