# inverse-audio-synthesis
Inverse audio synthesis

TODO:
* More magic numbers in py files to config
* Play with the dim, smaller than 256 might be fine.


BUGS:
* We use a batch_size of 128 for vicreg pretraining and a batch_size of
4 for downstream inverse synthesis. However, we are not careful about
our train/test splits so test for downstream might have been used as
training for vicreg. I don't think this is a big deal tho.
