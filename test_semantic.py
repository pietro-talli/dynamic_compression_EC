import torch

from nn_models.encoder import Encoder
from nn_models.quantizer import VectorQuantizerEMA
from nn_models.decoder import Decoder
from utils.utilities_dataset import FramesDataset, ToTensor
from torch.utils.data import DataLoader

from semantic_utils import load_ImageNet


# Create a dataset
dataset = FramesDataset('../dataset/description.csv', '../dataset/images', ToTensor())
dataloader = DataLoader(dataset, batch_size=128,
                        shuffle=True, num_workers=6)

# shape of the observation 
_, h,w = dataset[0]['curr'].shape

# Load the dataset
#imagenet_path = '/nfsd/signet4/imagenet'
imagenet_path = '/home/pietro/Projects/multihop-jscc/imagenet/tiny-imagenet-200'
trainloader, testloader = load_ImageNet([h,w], batch_size=128, path=imagenet_path, num_workers=4)

# Load the models
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 200
num_embeddings = 192

encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
quantizer = VectorQuantizerEMA(num_embeddings,embedding_dim)
decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, 3)

encoder.load_state_dict(torch.load('../models/encoder_training_sem.pt', map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('../models/decoder_training_sem.pt', map_location=torch.device('cpu')))
quantizer.load_state_dict(torch.load('../models/quantizer_192_training_sem.pt', map_location=torch.device('cpu')))

encoder.eval()
decoder.eval()
quantizer.eval()

im_batch, _ = next(iter(testloader))

# Encode the batch
z = encoder(im_batch)
_, z_q,_ , _ = quantizer(z)
im_dec = decoder(z_q)

# Plot the images
import matplotlib.pyplot as plt
import numpy as np

im_batch = im_batch.numpy()
im_dec = im_dec.detach().numpy()

im_batch = np.transpose(im_batch, (0,2,3,1))
im_dec = np.transpose(im_dec, (0,2,3,1))

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(im_batch[0])
ax[0,1].imshow(im_dec[0])
ax[1,0].imshow(im_batch[1])
ax[1,1].imshow(im_dec[1])
plt.show()
