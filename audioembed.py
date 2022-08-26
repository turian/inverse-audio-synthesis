import torch.nn as nn
from torch import Tensor
import torch


class AudioEmbedding(nn.Module):
    def __init__(self, gram, vision_model, img_preprocess, dim):
        super().__init__()
        self.gram = gram
        self.vision_model = vision_model
        self.img_preprocess = img_preprocess
        self.dim = dim

        self.batchnorm = nn.BatchNorm2d(3)

        # 576 = number of channels in efficientnet
        # 64 just because this is one of the biggest modules in the whole vicreg :(
        self.conv7 = nn.Conv2d(in_channels=576, out_channels=self.dim, kernel_size=2)
        self.conv6 = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=2
        )
        self.conv5 = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=2
        )
        self.conv1 = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=2
        )

        self.lin = nn.Linear(self.dim * 4, self.dim)

    def _preprocess(self, audio):
        x = audio

        # PQMF is 3 channel
        #        z = self.gram(audio)

        # Three channels of mel in different ways
        zlin = self.gram(audio)
        zlog1 = torch.log(zlin + 1e-6)
        zlog2 = torch.log10(zlin + 1e-2)
        z = torch.stack([zlin, zlog1, zlog2]).permute(1, 0, 2, 3)

        z = self.batchnorm(z)

        #        # This will be [128, 3, 58800]
        #        zimg = z.reshape(-1, 3, 240, 245)
        ##        # Convert float to unsigned bytes
        ##        zimg8 = scale8(zimg)
        ##        # torchvision 0.12.0
        ##        zimg8 = zimg8.float() / 255.0
        #        zimg8 = zimg

        #        # Apply inference preprocessing transforms
        #        zimg8preprocess = self.img_preprocess(zimg8)

        zimg8preprocess = self.img_preprocess(z)

        return zimg8preprocess

    def forward(self, audio):
        # This uses avg pool, which is lame
        # return self.vision_model(self._preprocess(audio))

        # Instead, we know that this is the shape of the
        # efficientnet on 4 second audio: torch.Size([4, 576, 8, 8])
        # So we just keep convolving it down to self.dim
        # This gives us a 4 second (or so) receptive field
        t = self.vision_model.features(self._preprocess(audio))
        t = self.conv7(t)
        t = self.conv6(t)
        t = self.conv5(t)
        t = self.conv4(t)
        t = self.conv3(t)
        t = self.conv2(t)
        t = self.conv1(t)
        t = t.view(audio.shape[0], -1)
        t = self.lin(t)
        return t

    def features(self, audio):
        return self.forward(audio)
