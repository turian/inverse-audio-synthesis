import torch.nn as nn
from torch import Tensor
from networks import WaveNet


class AudioEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=1)
        self.wavenet = WaveNet(layer_size=10, stack_size=5, in_channels=dim, res_channels=32)
        self.pool = nn.MaxPool1d(176400)

    def forward(self, audio):
        x = self.conv1(audio)
        print(x.shape)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        print(x.shape)
        x = self.wavenet(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        return x

    def features(self, audio):
        return self.forward(audio)
