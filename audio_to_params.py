import flash.core
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from vicreg_audio_params import VicregAudioParams


class AudioRepresentationToParams(nn.Module):
    def __init__(self, nparams, dim, hidden_norm, dropout):
        super().__init__()
        self.nparams = nparams
        self.dim = dim
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(self.dim, self.dim)
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
        self.lin3 = nn.Linear(self.dim, self.nparams)
        self.sigmoid = nn.Sigmoid()

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
        # Want 0/1 range
        x = self.sigmoid(x)
        return x


# def train_audio_to_params_through_torchsynth(
#    cfg: DictConfig,
#    device: torch.device,
#    vicreg: VICReg,
#    train_batch_num_dataloader,
#    val_batch_num_dataloader,
#    test_batch_num_dataloader,
#    mel_spectrogram,
# ) -> None:
#    """
#    Do audio
#        =(vicreg)=> audiorepr
#        =(new repr to params)=> params
#        =(torchsynth) => audio
#        and compute true vs predicted audio mel MSE.
# 	    TBH this probably won't work because you have to backprop
#    	through torchsynth :\
#    """
#
#    # We need a new one of these every time we change the batch size,
#    # which varies model to model. And might me we don't holdout correctly :(
#    synthconfig = SynthConfig(
#        batch_size=cfg.audio_repr_to_params.batch_size,
#        reproducible=cfg.torchsynth.reproducible,
#        sample_rate=cfg.torchsynth.rate,
#        buffer_size_seconds=cfg.torchsynth.buffer_size_seconds,
#    )
#    voice = Voice(synthconfig=synthconfig)
#    voice = voice.to(device)
#
#    audio_repr_to_params = AudioRepresentationToParams(nparams=cfg.nparams, dim=cfg.dim)
#    audio_repr_to_params = audio_repr_to_params.to(device)
#    # TODO: PUt this and vicreg lr in conf
#    audio_repr_to_params_optimizer = optim.SGD(
#        audio_repr_to_params.parameters(), lr=0.1
#    )
#
#    #    audio_repr_to_params_scaler = torch.cuda.amp.GradScaler()
#
#    # One epoch training
#    for audio_repr_to_params_train_batch_num, voice_batch_num in tqdm(
#        enumerate(train_batch_num_dataloader)
#    ):
#        wandb_step = (
#            audio_repr_to_params_train_batch_num * cfg.audio_repr_to_params.batch_size
#        )
#        assert voice_batch_num.numpy().shape == (1,)
#        voice_batch_num = voice_batch_num.numpy()
#        assert len(voice_batch_num) == 1
#        #        voice_batch_num = voice_batch_num[0].item()
#        voice_batch_num = 0
#
#        with torch.no_grad():
#            audio, params, is_train = voice(voice_batch_num)
#            audio = audio.unsqueeze(1)
#            audio.requires_grad_(True)
#            #  audio2 = apply_augmentation(audio)
#
#        # TODO: Tune vicreg?
#        with torch.no_grad():
#            vicreg.train()  # Disable dropout etc.
#            # We take the backbone and don't do the avg pool, because we lose too many features
#            # Instead we use all layers as the features :)
#            predicted_audio_repr = vicreg.backbone2.features(audio)
#            # Don't use projector to embedding, just the representation from the backbone
#            # predicted_audio_repr = vicreg.project(vicreg.backbone2(audio))
#
#        print(predicted_audio_repr.shape)
#        return
#
#        audio_repr_to_params_optimizer.zero_grad()
#        for w in audio_repr_to_params.parameters():
#            w.requires_grad_()
#        predicted_audio_repr.requires_grad_(True)
#        predicted_params = audio_repr_to_params.forward(predicted_audio_repr)
#        predicted_params = predicted_params.T
#
#        for param_name, param_value in zip(
#            voice.get_parameters().keys(), predicted_params
#        ):
#            param_name1, param_name2 = param_name
#            getattr(voice, param_name1).set_parameter_0to1(param_name2, param_value)
#
#        with torch.no_grad():
#            voice.freeze_parameters(voice.get_parameters().keys())
#            # # WHY??????
#            voice = voice.to(device)
#            (
#                predicted_audio,
#                predicted_predicted_params,
#                predicted_is_train,
#            ) = voice(None)
#            voice.unfreeze_all_parameters()
#
#        true_mel = mel_spectrogram(audio)
#        predicted_mel = mel_spectrogram(predicted_audio)
#
#        mel_l1_error = torch.mean(torch.abs(true_mel - predicted_mel))
#
#        if cfg.log == "wand":
#            wandb.log(
#                {
#                    "audio_repr_to_params/mel_l1_error": mel_l1_error.detach()
#                    .cpu()
#                    .numpy()
#                },
#                step=wandb_step,
#            )
#
#        print(mel_l1_error)
#        mel_l1_error.backward()
#        audio_repr_to_params_optimizer.step()
#
#
##        audio_repr_to_params_scaler.scale(mel_l1_error).backward()
##        audio_repr_to_params_scaler.step(audio_repr_to_params_optimizer)
##        audio_repr_to_params_scaler.update()
#
# train = train_audio_to_params


class AudioToParams(pl.LightningModule):
    def __init__(self, cfg: DictConfig, vicreg: VicregAudioParams) -> None:
        super().__init__()

        self.cfg = cfg

        self.vicreg = vicreg
        self.vicreg.freeze()
        self.vicreg.eval()

        self.audio_repr_to_params = AudioRepresentationToParams(
            nparams=cfg.nparams,
            dim=cfg.dim,
            hidden_norm=cfg.audio_to_params.hidden_norm,
            dropout=cfg.audio_to_params.dropout,
        )

    def _step(self, batch, batch_idx, name):
        # TODO: Try removing CPU move
        assert batch.detach().cpu().numpy().shape == (1,)
        voice_batch_num = batch.detach().cpu().numpy()
        assert len(voice_batch_num) == 1
        voice_batch_num = voice_batch_num[0].item()

        self.vicreg.freeze()
        self.vicreg.eval()

        audio, params, is_train = self.vicreg.voice(voice_batch_num)
        audio = audio.unsqueeze(1)

        true_params_repr = self.vicreg.vicreg.backbone_param(params)
        true_params_embedding = self.vicreg.vicreg.projector(true_params_repr)

        audio_repr = self.vicreg.vicreg.backbone_audio(audio)
        true_audio_embedding = self.vicreg.vicreg.projector(audio_repr)

        predicted_params = self.audio_repr_to_params.forward(audio_repr)

        predicted_params_repr = self.vicreg.vicreg.backbone_param(predicted_params)
        predicted_params_embedding = self.vicreg.vicreg.projector(predicted_params_repr)

        repr_loss = F.mse_loss(true_params_embedding, predicted_params_embedding)
        # This is purely diagnostic, since vicreg is frozen
        frozen_vicreg_loss = F.mse_loss(true_params_embedding, true_audio_embedding)

        # TODO: Auraloss?

        self.log(f"audio_to_params/{name}/loss", repr_loss)
        self.log(f"audio_to_params/{name}/frozen_vicreg_loss", frozen_vicreg_loss)

        return repr_loss

    def training_step(self, batch, batch_idx):
        repr_loss = self._step(batch, batch_idx, "train")

        sch = self.lr_schedulers()

        # step every N batches
        if (batch_idx + 1) % 10000 == 0:
            sch.step()

        return repr_loss

    def test_step(self, batch, batch_idx):
        repr_loss = self._step(batch, batch_idx, "test")

        return repr_loss

    def configure_optimizers(self):
        if self.cfg.audio_to_params.optim.name == "sgd":
            return optim.SGD(self.parameters(), **self.cfg.audio_to_params.optim.args)
        elif self.cfg.audio_to_params.optim.name == "lars":
            # TODO: Add cosine scheduler?
            # https://arxiv.org/pdf/2105.04906.pdf
            # Section 4.2: "The learning rate follows a cosine decay
            # schedule Loshchilov & Hutter (2017), starting from 0 with
            # 10 warmup epochs and with final value of 0.002."
            return flash.core.optimizers.LARS(
                self.parameters(),
                weight_decay=self.cfg.audio_to_params.optim.args.weight_decay,
                # https://arxiv.org/pdf/2105.04906.pdf
                # section 4.2
                lr=self.cfg.audio_to_params.batch_size
                / 256
                * self.cfg.audio_to_params.optim.args.base_lr,
            )
        else:
            assert False
