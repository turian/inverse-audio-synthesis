# inverse-audio-synthesis
Inverse audio synthesis

TODO:
* Dropout in downstream!
* Make sure to use .train for training models and .eval for eval
* More magic numbers in py files to config
* Play with the dim, smaller than 256 might be fine.

* Try sashimi backbone: https://github.com/HazyResearch/state-spaces/tree/main/sashimi
* Try audio diffusion: https://github.com/archinetai/audio-diffusion-pytorch
* Try musika: https://arxiv.org/abs/2208.08706


BUGS:
* We use a batch_size of 128 for vicreg pretraining and a batch_size of
4 for downstream inverse synthesis. However, we are not careful about
our train/test splits so test for downstream might have been used as
training for vicreg. I don't think this is a big deal tho.
