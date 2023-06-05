import torch
import torch.nn as nn
import numpy as np
import gym
from nn_models.policy import RecA2C
from nn_models.sensor import Sensor_not_quantized_level_A
from nn_models.encoder import Encoder
from nn_models.quantizer import VectorQuantizerEMA
from nn_models.regressor import PhysicalValueRegressor
from nn_models.decoder import Decoder
from collections import deque, namedtuple
from torch.distributions import Categorical
import torch.nn.functional as F





env = gym.make("CartPole-v1", render_mode = 'rgb_array')
embedding_dim = 8
num_f = 8
latent_dim = embedding_dim*num_f
agent_policy = RecA2C(latent_dim,latent_dim,env.action_space.n)

agent_policy.load_state_dict(torch.load("../models/policy_a2c_64.pt", map_location=torch.device('cpu')))

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
encoder.load_state_dict(torch.load('../models/encoder.pt', map_location=torch.device('cpu')))

decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
decoder.load_state_dict(torch.load('../models/decoder.pt', map_location=torch.device('cpu')))

sensor = Sensor_not_quantized_level_A(encoder)

#Load all the quantizers 

quantizers = []
num_quantization_levels = [0,1,2,3,4,5,6]
num_codewords_s = [2,4,8,16,32,64]

for i in num_quantization_levels[1:]:
    quantizer = VectorQuantizerEMA(num_codewords_s[i-1], embedding_dim)
    quantizer.load_state_dict(torch.load('../models/quantizer_'+str(num_codewords_s[i-1])+'.pt', map_location=torch.device('cpu')))
    quantizers.append(quantizer)
    
list_of_quantizers = nn.ModuleList(quantizers)

regressors = []
for i in num_quantization_levels[1:]:
    regressor = PhysicalValueRegressor(latent_dim, 4)
    regressor.load_state_dict(torch.load('../models/regressor_'+str(num_codewords_s[i-1])+'.pt', map_location=torch.device('cpu')))
    regressors.append(regressor)
        
list_of_regressors = nn.ModuleList(regressors)

SavedAction = namedtuple('SavedAction', ['action', 'value'])
eps = np.finfo(np.float32).eps.item()

def select_action(state, model):
    with torch.no_grad():
        probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(action.item(), state_value))

    # the action to take (left or right)
    return action.item()

def run_episode(sensor_policy,env,level):
    env.reset()
    done = False
    cost = 0
    ep_reward = 0

    prev_screen = None
    with torch.no_grad():
        state_not_quantized, curr_screen, frames = sensor(env, prev_screen)
    states = deque(maxlen = 20)
    states_quantized = deque(maxlen=20)
    score = 0
    q = -1
    while not done:
        q_tensor = torch.tensor([q])
        s_and_prev_q = torch.cat([state_not_quantized.reshape(-1), q_tensor])
        states.append(s_and_prev_q.reshape(1,-1))
        input_state = torch.cat(list(states),0)

        q = select_action(input_state, sensor_policy)

        with torch.no_grad():
            if q == 0 and score == 0:
                q = 1
            if q > 0:
                _,state_quantized,_,_ = list_of_quantizers[q-1](state_not_quantized)
            if q == 0 and score >0:
                state_quantized = state_quantized
            states_quantized.append(state_quantized.reshape(1,-1))
            input_state_quantized = torch.cat(list(states_quantized),0)
            action = select_action(input_state_quantized, agent_policy)

        state, reward, done, _, _ = env.step(action)

        state = env.state
        if level == 'B':
            with torch.no_grad():
                state_tensor = torch.tensor(state)
                reward = F.mse_loss(regressors[-1](input_state_quantized), state_tensor)
        if level == 'A':
            with torch.no_grad():
                reward = -10*torch.log10(F.mse_loss(decoder(state_quantized), frames))
        if level == 'C':
            reward = 1
        
        prev_screen = curr_screen
        with torch.no_grad():
            state_not_quantized, curr_screen, frames = sensor(env, prev_screen)
        
        ep_reward += reward
        score += 1
        cost+=q
        if score >= 500:
            done = True
    return cost, ep_reward, score, agent_policy.saved_actions, sensor_policy.saved_actions