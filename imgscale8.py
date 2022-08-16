# Based upon 32K torchsynth sounds
# If you pass white noise, you get SMALLER values: 0.7891 .. -0.6486
maxval = 1.5680482
minval = -1.6843455

# TODO: Would be smarter but trickier to use quantile scaling
def scale8(x, xmin=minval, xmax=maxval):
    xscale = (x - xmin) / (xmax - xmin) * 255
    return torch.clip(xscale, 0, 255).type(torch.cuda.ByteTensor)


#    return torch.cuda.ByteTensor(torch.clip(xscale, 0, 255))


def unscale8(x, xmin=minval, xmax=maxval):
    return x / 255.0 * (xmax - xmin) + xmin
