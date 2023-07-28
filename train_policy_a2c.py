import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import gym
import sys
from nn_models.encoder import Encoder
from nn_models.decoder import Decoder
from nn_models.quantizer import VectorQuantizerEMA
from nn_models.policy import RecA2C
from nn_models.sensor import Sensor
from utils.rl_utils import A2C
import argparse

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'

print('using '+device)

num_codewords = 64
embedding_dim = 8
num_episodes = 20000

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--num_episodes', type=int, help='number of episode to train the policy', required=False)
parser.add_argument('--num_codewords', type=int, help='selct the quantizer to quantize the latent space (default is 64)', required=False)
parser.add_argument('--embedding_dim', type=int, help='selct the latent space size (default is 8)', required=False)

args = parser.parse_args()

if args.num_episodes: num_episodes =  args.num_episodes 
if args.embedding_dim: embedding_dim = args.embedding_dim
if args.num_codewords: num_codewords = args.num_codewords 

#Parameters for the encoder
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

#Model definition
encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
quantizer = VectorQuantizerEMA(num_codewords, embedding_dim)
decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

#Load the model
encoder.load_state_dict(torch.load('../models/encoder.pt', map_location=torch.device('cpu')))
quantizer.load_state_dict(torch.load('../models/quantizer_'+str(num_codewords)+'.pt', map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('../models/decoder.pt', map_location=torch.device('cpu')))

encoder.eval()
quantizer.eval()
decoder.eval()

#Create the environment
env = gym.make('CartPole-v1', render_mode = 'rgb_array')
features = 8
latent_dim = features*embedding_dim

#Create the sensor
sensor = Sensor(encoder, quantizer)

#Create the policy
model = RecA2C(latent_dim, latent_dim, env.action_space.n)

#Train the policy
model = A2C(env, model, sensor, num_episodes, gamma=0.99, num_codewords=num_codewords)
torch.save(model.state_dict(), '../models/policy_a2c_'+str(num_codewords)+'.pt')