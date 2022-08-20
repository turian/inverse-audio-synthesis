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
* Should use FullGatherLayer in vicreg, maybe fix the batch size there too


Other ideas:
* "We proposed Prime-Dilated Convolution (PDC) – a new convolution
network layer structure specially designed for better utilization
of Constant-Q Transform (CQT) chromagram information of an audio
sample." (https://arxiv.org/pdf/2205.03043.pdf)
* "We demonstrated the benefit of using Multi-modal Features in
a combined network. The experiment showed that spectral (visual),
waveform (sequential), and statistical (numerical) sound feature
information altogether improved the generalizability of the model."
(https://arxiv.org/pdf/2205.03043.pdf)
