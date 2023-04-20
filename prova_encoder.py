import torch
from nn_models.encoder import Encoder
from nn_models.decoder import Decoder
from nn_models.quantizer import VectorQuantizerEMA
import gym
from nn_models.sensor import get_screen
import matplotlib.pyplot as plt

num_codewords = 64
embedding_dim = 8

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
quantizer = VectorQuantizerEMA(num_codewords, embedding_dim)
decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

encoder.load_state_dict(torch.load('../models/encoder_training.pt', map_location=torch.device('cpu')))
quantizer.load_state_dict(torch.load('../models/quantizer_'+str(num_codewords)+'_training.pt', map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('../models/decoder_training.pt', map_location=torch.device('cpu')))

encoder.eval()
quantizer.eval()
decoder.eval()

env = gym.make('CartPole-v1', render_mode = 'rgb_array')
features = 8
latent_dim = features*embedding_dim

env.reset()
screen = get_screen(env)
plt.imshow(1-screen[0])

input_tensor = torch.cat([screen,screen])
_,h,w = input_tensor.shape
_,quantized,_,_ = quantizer(encoder(1-input_tensor.unsqueeze(0)))

recon = decoder(quantized)
recon = recon.detach().numpy()
plt.figure()
plt.imshow(recon[0,0])
plt.show()

from utils.utilities_dataset import FramesDataset, ToTensor
from torch.utils.data import DataLoader

# Create a dataset
dataset = FramesDataset('../dataset/description.csv', '../dataset/images', ToTensor())
dataloader = DataLoader(dataset, batch_size=128,
                        shuffle=True, num_workers=6)

print(dataset[0]['curr'].shape)

plt.figure()
plt.imshow(dataset[0]['curr'][0].numpy())
plt.show()
