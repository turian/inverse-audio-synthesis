import torch.nn as nn
from torch import Tensor


class AudioRepresentationToParams(nn.Module):
    def __init__(self, nparams, dim):
        super().__init__()
        self.nparams = nparams
        self.dim = dim
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(self.dim, self.dim)
        self.lin2 = nn.Linear(self.dim, self.dim)
        self.lin3 = nn.Linear(self.dim, self.nparams)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, DIM)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        # Want 0/1 range
        x = self.sigmoid(x)
        return x


def audio_repr_to_params_batch(
    cfg,
    device,
    batch_num,
    vicreg,
    voice,
    mel_spectrogram,
):
    with torch.no_grad():
        test_true_audio, test_true_params, test_true_is_train = voice(batch_num)

    # TODO: Tune vicreg?
    with torch.no_grad():
        # Don't use projector to embedding, just the representation from the backbone
        test_predicted_audio_representation = vicreg.backbone2(
            test_true_audio.unsqueeze(1)
        )
        # vicreg.project(vicreg.backbone2(test_true_audio.unsqueeze(1)))
        test_predicted_audio_representation.shape

    audio_repr_to_params = AudioRepresentationToParams(nparams=cfg.nparams, dim=cfg.dim)
    audio_repr_to_params = audio_repr_to_params.to(device)
    test_predicted_params = audio_repr_to_params.forward(
        test_predicted_audio_representation
    ).T

    for param_name, param_value in zip(
        voice.get_parameters().keys(), test_predicted_params
    ):
        param_name1, param_name2 = param_name
        getattr(voice, param_name1).set_parameter_0to1(param_name2, param_value)

    voice.freeze_parameters(voice.get_parameters().keys())
    # # WHY??????
    voice = voice.to(device)
    (
        test_predicted_audio,
        test_predicted_predicted_params,
        test_predicted_is_train,
    ) = voice(None)
    voice.unfreeze_all_parameters()

    test_true_mel = mel_spectrogram(test_true_audio)
    test_predicted_mel = mel_spectrogram(test_predicted_audio)

    mel_l1_error = torch.mean(torch.abs(test_true_mel - test_predicted_mel))

    if cfg.log == "wand":
        for i in tqdm(list(range(8))):
            silence = torch.zeros(int(RATE * 0.5))
            silence = silence.to(device)
            test_true_predict_audio = torch.cat(
                [test_true_audio[i], silence, test_predicted_audio[i]]
            )
            this_test_wav_filename = f"test_{'%010d' % batch_num}_{'%03d' % i}.wav"
            this_test_wav_numpy = (
                test_true_predict_audio.unsqueeze(1).detach().cpu().numpy()
            )
            soundfile.write(this_test_wav_filename, this_test_wav_numpy, RATE)
            artifact = wandb.Artifact(this_test_wav_filename, type="model")
            artifact.add_file(vicreg_checkpoint_filename)
            run.log_artifact(artifact)
            wandb.log(
                {
                    this_test_wav_filename: wandb.Audio(
                        this_test_wav_numpy,
                        caption=this_test_wav_filename,
                        sample_rate=RATE,
                    )
                }
            )
    return mel_l1_error


def audio_repr_to_params(
    cfg: DictConfig,
    device: torch.device,
    vicreg: VICReg,
    voice: Voice,
    train_batch_num_dataloader,
    val_batch_num_dataloader,
    test_batch_num_dataloader,
    mel_spectrogram,
) -> None:
    audio_repr_to_params_optimizer = optim.SGD(audio_repr_to_params.parameters(), lr=0.1)

    # One epoch training
    for audio_repr_to_params_train_batch_num, voice_batch_num in tqdm(
        enumerate(train_batch_num_dataloader)
    ):
        assert voice_batch_num.numpy().shape == (1,)
        voice_batch_num = voice_batch_num.numpy()
        assert len(voice_batch_num) == 1
        voice_batch_num = voice_batch_num[0].item()

        audio, params, is_train = voice(voice_batch_num)
        audio = audio.unsqueeze(1)
        #  audio2 = apply_augmentation(audio)

        if cfg.audio_repr_to_params.use_lars:
            lr = adjust_learning_rate(cfg, audio_repr_to_params_optimizer, loader, step)
            if cfg.log == "wand":
                wandb.log({"lars_lr": lr})
            audio_repr_to_params_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                audio_repr_to_params_loss = audio_repr_to_params.forward(params, audio)
        else:
            audio_repr_to_params_optimizer.zero_grad()
            audio_repr_to_params_loss = audio_repr_to_params.forward(params, audio)

        #  loss = audio_repr_to_params(audio2, audio)
        audio_repr_to_params_loss = audio_repr_to_params(params, audio)
        #  loss = audio_repr_to_params(params, params)
        audio_repr_to_params_lossval = audio_repr_to_params_loss.detach().cpu().numpy()
        if math.isnan(audio_repr_to_params_lossval):
            print("NAN")
            sys.stdout.flush()
            continue
        #            break
        if cfg.log == "wand":
            wandb.log({"audio_repr_to_params_loss": audio_repr_to_params_lossval})

        # loss.backward()
        # optimizer.step()

        audio_repr_to_params_scaler.scale(audio_repr_to_params_loss).backward()
        audio_repr_to_params_scaler.step(audio_repr_to_params_optimizer)
        audio_repr_to_params_scaler.update()