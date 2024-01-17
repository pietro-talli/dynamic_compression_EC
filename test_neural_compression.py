import matplotlib.pyplot
import numpy
import torch

from neuralcompression.data import Kodak
from neural_compress._fpa import FactorizedPriorAutoencoder
import skimage.data
import skimage.transform
import torchvision.transforms
from PIL import Image
import urllib.request

x = Image.open("../dataset/images/0_0.png")

transform = torchvision.transforms.Compose([
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize((64, 128)),
    torchvision.transforms.ConvertImageDtype(torch.float32),
])

x = transform(x).unsqueeze(0)

matplotlib.pyplot.figure(figsize=(12, 18))

matplotlib.pyplot.imshow(x.squeeze().permute(1, 2, 0))

matplotlib.pyplot.show()

network = FactorizedPriorAutoencoder()

url = "https://dl.fbaipublicfiles.com/neuralcompression/models/factorized_prior_vimeo_90k_mse_128_192_0_0015625.pth"

state_dict = torch.hub.load_state_dict_from_url(url)

network.load_state_dict(state_dict, strict=False)
print('bottleneck', network.bottleneck._quantized_cdf) 

with torch.no_grad():
    strings, broadcast_size = network.compress(x)
    print(type(strings[0][0]))
    print(len(strings[0][0])*8)
    print(broadcast_size)

with torch.no_grad():
    x_hat = network.decompress(strings, broadcast_size)

matplotlib.pyplot.figure(figsize=(12, 18))

matplotlib.pyplot.imshow(x_hat.squeeze().permute(1, 2, 0))

matplotlib.pyplot.show()

import math

from torch import Tensor

import torch.nn.functional as F


def psnr(x: Tensor, x_hat: Tensor) -> float:
    print(x.shape, x_hat.shape)
    return -10 * math.log10(F.mse_loss(x, x_hat[:,:,:,0:360]).item())

print(psnr(x, x_hat))