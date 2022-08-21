#!/usr/bin/env python3
# vicreg on synth1b1 with 3 channel pqmfs
#
# TODO:
# * Add EMA
# * Interleave pretraining and downstream
# * multigpu

import numpy as np

import os
import os.path

import hydra
import torch

# import torch.distributed as dist
import torchaudio
import torchaudio.transforms
from omegaconf import DictConfig, OmegaConf
from pynvml import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small  # , MobileNet_V3_Small_Weights

import wandb
from audio_to_params import AudioToParams
from vicreg_audio_params import VicregAudioParams

from prettytable import PrettyTable

# https://stackoverflow.com/a/62508086/82733
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_filter_range(vicreg, logger):
        # Show a plot of what the filter values are like
        # on an excerpt from music
        (audio, _rate) = torchaudio.load("daddy.wav")
        audio.to(vicreg.device)
        vicreg.eval()
        with torch.no_grad():
            z = vicreg.audio_repr._preprocess(audio.unsqueeze(1))
            z = z.permute(1,0,2,3)
        z = z.detach().cpu().numpy()
        # There are three channels, each of which might be a different kind of
        # spectrogram. So
        for i in range(3):
            y = np.copy(z[i])
            np.random.shuffle(y)
            y = y[:1000]
            x = np.arange(0, len(y))
            data = [[x, y] for (x, y) in zip(x.tolist(), sorted(y.tolist()))]
            table = wandb.Table(data=data, columns = ["x", "y"])
            logger.experiment.log(
                {
                    f"filter_{i}": wandb.plot.line(table, "x", "y", title="Filter range")
                }
            )

@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #os.environ["WANDB_CACHE_DIR"] = "/fsx/turian/.cache"
    wandb.login()

    seed_everything(42, workers=True)

    # BUG: We use a batch_size of 128 for vicreg pretraining and a batch_size of
    # 4 for downstream inverse synthesis. However, we are not careful about
    # our train/test splits so test for downstream might have been used as
    # training for vicreg. I don't think this is a big deal tho.
    batch_nums = torch.tensor(list(range(cfg.num_batches)))
    # batch_num_dataset = torch.utils.data.DataSet(batch_nums)
    # batch_num_dataset = batch_num_dataset.to(device)

    ntest_batches = cfg.ntest_batches
    ntrain_batches = int((cfg.num_batches - ntest_batches) * 0.9)
    # ntest_batches = cfg.num_batches - ntrain_batches - nval_batches
    nval_batches = cfg.num_batches - ntrain_batches - ntest_batches
    (
        train_batch_num_dataset,
        val_batch_num_dataset,
        test_batch_num_dataset,
    ) = torch.utils.data.random_split(
        batch_nums,
        [ntrain_batches, nval_batches, ntest_batches],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_batch_num_dataloader = torch.utils.data.DataLoader(train_batch_num_dataset)
    val_batch_num_dataloader = torch.utils.data.DataLoader(val_batch_num_dataset)
    test_batch_num_dataloader = torch.utils.data.DataLoader(test_batch_num_dataset)

    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.torchsynth.rate,
        n_fft=cfg.mel.n_fft,
        win_length=cfg.mel.win_length,
        hop_length=cfg.mel.hop_length,
        center=cfg.mel.center,
        pad_mode=cfg.mel.pad_mode,
        power=cfg.mel.power,
        norm=cfg.mel.norm,
        onesided=cfg.mel.onesided,
        n_mels=cfg.mel.n_mels,
        mel_scale=cfg.mel.mel_scale,
    )
    """

    # vicreg_scaler = torch.cuda.amp.GradScaler()

    vicreg = VicregAudioParams(cfg)

    if cfg.log == "wand":
        # if not os.path.exists("/tmp/turian-wandb/wandb/"):
        #   os.makedirs("/tmp/turian-wandb/wandb/", exist_ok=True)
        logger = WandbLogger(
            # Set the project where this run will be logged
            project="vicreg-synth1b1-pqmfs",
            #      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #      name=f"experiment_{run}",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            # Log model checkpoints as they get created during training
            log_model="all",
            # save_dir="/tmp/turian-wandb",
        )
        # We don't use gradients much and the use a lot of logging space
        # logger.watch(vicreg)

        plot_filter_range(vicreg, logger)
    else:
        logger = None

    if cfg.vicreg.do_pretrain:
        count_parameters(vicreg)
        vicreg_model_checkpoint = ModelCheckpoint(
            every_n_train_steps=cfg.vicreg.checkpoint_every_nbatches,
            #            dirpath="chkpts/",
            filename="vicreg-{epoch:02d}-{step:04d}",
            monitor=None,
            save_last=True,
        )
        # TODO: Remove limit_train_batches
        vicreg_trainer = Trainer(
            logger=logger,
            limit_train_batches=cfg.vicreg.limit_train_batches,
            max_epochs=1,
            # precision=cfg.precision,
            detect_anomaly=True,  # useful logs about when and where the Nan or inf anomaly happens
            accelerator=cfg.accelerator,
            strategy=cfg.strategy,
            devices=cfg.devices,
            deterministic=True,
            callbacks=[vicreg_model_checkpoint],
            # callbacks = [vicreg_model_checkpoint, ORTCallback()],
            # Doesn't work with our CUDA version :(
            # https://github.com/Lightning-AI/lightning-bolts
            # callbacks = ORTCallback(),
        )
        #        from copy import deepcopy
        #        deepcopy(vicreg_trainer.callback_metrics)
        vicreg_trainer.fit(
            vicreg,  # vicreg_scaler, vicreg_optimizer,
            train_dataloaders=train_batch_num_dataloader,
        )

    audio_to_params = AudioToParams(cfg, vicreg)
    if vicreg is not None:
        audio_to_params_model_checkpoint = ModelCheckpoint(
            every_n_train_steps=cfg.audio_to_params.checkpoint_every_nbatches,
            dirpath="chkpts/",
            filename="audio_to_params-{epoch:02d}-{step:04d}",
            monitor=None,
            save_last=True,
        )
        # TODO: Remove limit_train_batches
        audio_to_params_trainer = Trainer(
            logger=logger,
            limit_train_batches=cfg.audio_to_params.limit_train_batches,
            max_epochs=1,
            # precision=cfg.precision,
            detect_anomaly=True,  # useful logs about when and where the Nan or inf anomaly happens
            accelerator=cfg.accelerator,
            strategy=cfg.strategy,
            devices=cfg.devices,
            deterministic=True,
            callbacks=[audio_to_params_model_checkpoint],
            # callbacks = [audio_to_params_model_checkpoint, ORTCallback()],
            # Doesn't work with our CUDA version :(
            # https://github.com/Lightning-AI/lightning-bolts
            # callbacks = ORTCallback(),
        )
        #        from copy import deepcopy
        #        deepcopy(audio_to_params_trainer.callback_metrics)
        audio_to_params_trainer.fit(
            audio_to_params,  # audio_to_params_scaler, audio_to_params_optimizer,
            train_dataloaders=train_batch_num_dataloader,
        )

        audio_to_params_trainer.test(
            audio_to_params, dataloaders=test_batch_num_dataloader
        )

    if cfg.log == "wand":
        wandb.finish()


if __name__ == "__main__":
    app()
