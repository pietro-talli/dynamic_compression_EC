import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import sys
from nn_models.encoder import Encoder
from nn_models.decoder import Decoder
from nn_models.quantizer import VectorQuantizerEMA
from nn_models.policy import RecDQN
from nn_models.sensor import Sensor
from nn_models.regressor import PhysicalValueRegressor

import argparse

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'

print('using '+device)

num_codewords = 64
embedding_dim = 8
batch_size = 256
num_episodes = 20000
exploring = 0.2

parser = argparse.ArgumentParser(description='Train the sensor')
parser.add_argument('--num_episodes', type=int, help='number of episode to train the policy', required=False)
parser.add_argument('--embedding_dim', type=int, help='selct the latent space size (default is 64)', required=False)
parser.add_argument('--batch_size', type=int, required=False)
parser.add_argument('--level', type=str, help='one of the three level of communication either A, B or C', required = True)

args = parser.parse_args()

if args.batch_size: batch_size = args.batch_size
if args.num_episodes: num_episodes =  args.num_episodes 
if args.embedding_dim: embedding_dim = args.embedding_dim

level = args.level

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
quantizer = VectorQuantizerEMA(num_codewords, embedding_dim)
decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

encoder.load_state_dict(torch.load('../models/encoder.pt', map_location=torch.device('cpu')))
quantizer.load_state_dict(torch.load('../models/quantizer_'+str(num_codewords)+'.pt', map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('../models/decoder.pt', map_location=torch.device('cpu')))

encoder.eval()
quantizer.eval()
decoder.eval()

env = gym.make('CartPole-v1', render_mode = 'rgb_array')
state, _ = env.reset()

features = 8
latent_dim = features*embedding_dim


#Load all the quantizers 

quantizers = []
num_quantization_levels = [0,1,2,3,4,5,6]
num_codewords = [2,4,6,8,16,32,64]

for i in num_quantization_levels[1:]:
    quantizer = VectorQuantizerEMA(num_codewords[], embedding_dim)
    quantizers[i-1] = 

list_of_quantizers = nn.ModuleList()


from nn_models.policy import RecA2C
model = RecA2C(latent_dim, latent_dim, env.action_space.n)
model.load_state_dict(torch.load('../models/policy_a2c_'+str(num_codewords)+'.pt', map_location=torch.device('cpu')))
