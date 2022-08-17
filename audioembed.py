import torch.nn as nn
from torch import Tensor

from imgscale8 import scale8, unscale8


class AudioEmbedding(nn.Module):
    def __init__(self, pqmf, vision_model, img_preprocess):
        super().__init__()
        self.pqmf = pqmf
        self.vision_model = vision_model
        self.img_preprocess = img_preprocess

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
        zimg8preprocess = self.img_preprocess(zimg8)

        y = self.vision_model(zimg8preprocess)
        return y
