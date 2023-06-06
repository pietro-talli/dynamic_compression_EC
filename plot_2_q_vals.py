from utils.test_utils import run_episode_for_gradient
import gym
from nn_models.policy import RecA2C
import torch
import numpy as np
import matplotlib.pyplot as plt
from nn_models.sensor import get_screen
from nn_models.encoder import Encoder
from nn_models.quantizer import VectorQuantizerEMA
import torch.nn as nn

latent_dim = 64
quantization_levels = 7
sensor_policy = RecA2C(latent_dim+1, latent_dim, quantization_levels)

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, 8)
encoder.load_state_dict(torch.load('../models/encoder.pt', map_location=torch.device('cpu')))

name = '../models/sensor_level_C_a2c_0.15_train.pt'
env = gym.make('CartPole-v1',render_mode='rgb_array')

sensor_policy.load_state_dict(torch.load(name, map_location=torch.device('cpu')).state_dict())

num_steps = 20

x_range = np.linspace(-2.4, 2.4, num_steps)
x_dot_range = np.linspace(-2, 2, num_steps)

theta_range = np.linspace(-0.2094, 0.2094, num_steps)
theta_dot_range = np.linspace(-1,1, num_steps)

env.reset()
screen = get_screen(env)
h,w = screen.squeeze().numpy().shape

quantizers = []
num_quantization_levels = [0,1,2,3,4,5,6]
num_codewords_s = [2,4,8,16,32,64]
embedding_dim = 8

for i in num_quantization_levels[1:]:
    quantizer = VectorQuantizerEMA(num_codewords_s[i-1], embedding_dim)
    quantizer.load_state_dict(torch.load('../models/quantizer_'+str(num_codewords_s[i-1])+'.pt', map_location=torch.device('cpu')))
    quantizers.append(quantizer)
    
list_of_quantizers = nn.ModuleList(quantizers)

value_tensor = np.zeros([num_steps, num_steps, num_steps, num_steps])

from tqdm import tqdm

for i,x in tqdm(enumerate(x_range)):
    for j,x_dot in tqdm(enumerate(x_dot_range)):
        for v,theta in enumerate(theta_range):
            for u,theta_dot in enumerate(theta_dot_range):
                env.reset()
                env.state[0] = x
                env.state[1] = x_dot
                env.state[2] = theta
                env.state[3] = theta_dot

                screen = get_screen(env)
                with torch.no_grad():
                    input_tensor = torch.reshape(torch.cat((screen, screen)), (1,2,h,w))

                    encoded = encoder(1-input_tensor)
                    _,quantized,_,_ = list_of_quantizers[-1](encoded)

                    quantized = quantized.reshape(-1)

                    prev_action = torch.tensor([0.5])

                    quantized = torch.cat([quantized, prev_action])
                    quantized = quantized.reshape(1,-1)
                    _, value = sensor_policy(quantized)
                
                value_tensor[i,j,v,u] = value.item()


                


