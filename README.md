# inverse-audio-synthesis
Inverse audio synthesis

TODO:
* More magic numbers in py files to config


BUGS:
* We use a batch_size of 128 for vicreg pretraining and a batch_size of
4 for downstream inverse synthesis. However, we are not careful about
our train/test splits so test for downstream might have been used as
training for vicreg. I don't think this is a big deal tho.
