import torch.nn as nn
from torch import Tensor
import torch

class AudioEmbedding(nn.Module):
    def __init__(self, audio_model, dim):
        super().__init__()
        self.audio_model = audio_model
        self.proj = nn.Linear(768, 128)

    def forward(self, audio):
        features = self.audio_model.extract_features(audio)[0]
        features = torch.stack(features)
        x = x.permute(1, 2, 0, 3)
        print(features.shape)
        x = features.permute(1, 2, 0, 3)
        print(x.shape)
        x = self.proj(x)
        print(x.shape)

        
        return x

    def features(self, audio):
        return self.forward(audio)
