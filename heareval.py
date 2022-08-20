#!/usr/bin/env python3

from vicreg_audio_params import VicregAudioParams
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    # TODO: Save cfg in checkpoint, don't pass in new cfg to create
    # VicregAudioParams
    model = VicregAudioParams.load_from_checkpoint("vicreg.ckpt", cfg=cfg)
    print(model.vicreg.backbone_audio(torch.rand(4, 1, 44100 * 4)).shape)

if __name__ == "__main__":
    app()
