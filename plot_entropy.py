from utils.test_utils import run_episode_for_gradient
import gym
from nn_models.policy import RecA2C
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

retrain = False

parser = argparse.ArgumentParser(description='Test for policy gradient')
parser.add_argument('--retrain', type=bool, help='collect agin the data', required=False)
parser.add_argument('--t_lag', type=int, help='time lag', required=False)

args = parser.parse_args()

if args.retrain: retrain = args.retrain

from tqdm import tqdm

if retrain:
    latent_dim = 64
    quantization_levels = 7
    sensor_policy = RecA2C(latent_dim+1, latent_dim, quantization_levels)

    name = '../models/sensor_level_A_a2c_2.0_train.pt'
    env = gym.make('CartPole-v1',render_mode='rgb_array')

    sensor_policy.load_state_dict(torch.load(name, map_location=torch.device('cpu')).state_dict())

    true_states = []
    for i in tqdm(range(1000)):
        cost, ep_reward, score, aa, sa, true_states = run_episode_for_gradient(sensor_policy,env,'C', true_states)

    agent_actions = [aa[t][0] for t in range(len(sa))]
    agent_values = [aa[t][1] for t in range(len(sa))]

    sensor_actions = [sa[t][0] for t in range(len(sa))] 
    sensor_values = [sa[t][1] for t in range(len(sa))] 

    assert len(agent_actions) == len(sensor_actions)

    true_states = np.concatenate(true_states, 0)

    np.save('../save_numpy/true_states_A.npy', true_states)
    np.save('../save_numpy/action_agent_A.npy', np.array(agent_actions))
    np.save('../save_numpy/values_agent_A.npy', np.array(agent_values))
    np.save('../save_numpy/action_sensor_A.npy', np.array(sensor_actions))
    np.save('../save_numpy/values_sensor_A.npy', np.array(sensor_values))

beta = '_15'

t_lag = 0

if args.t_lag: t_lag = args.t_lag 

true_states = np.load('../save_numpy/true_states_500_level_A.npy')
agent_actions = np.load('../save_numpy/action_agent_500_level_A.npy')
sensor_actions = np.load('../save_numpy/action_sensor_500_level_A.npy') # action_sensor_B
agent_values = np.load('../save_numpy/values_agent_500_level_A.npy')
sensor_values = np.load('../save_numpy/values_sensor_500_level_A.npy')

x = true_states[:,0]
x_dot = true_states[:,1]
theta = true_states[:,2]
theta_dot = true_states[:,3]

num_bins = 10
entropy_bins = 7
hyper_cube_count = np.zeros([num_bins,num_bins,num_bins,num_bins])
hyper_cube_action = np.zeros([num_bins,num_bins,num_bins,num_bins])
hyper_cube_entropy = np.zeros([num_bins,num_bins,num_bins,num_bins])

epsilon = 1e-8

x_range = [x.min() - epsilon, x.max() + epsilon]
x_dot_range = [x_dot.min() - epsilon,x_dot.max() + epsilon]
theta_range = [theta.min() - epsilon,theta.max() + epsilon]
theta_dot_range = [theta_dot.min() - epsilon,theta_dot.max() + epsilon]

from tqdm import tqdm
prev_state = np.array([5,5,5,5])
for idx, state in tqdm(enumerate(true_states)):
    x_i = int((state[0] - x_range[0]) // ((x_range[1] - x_range[0])/num_bins))
    x_dot_i = int((state[1] - x_dot_range[0]) // ((x_dot_range[1] - x_dot_range[0])/num_bins))
    theta_i = int((state[2] - theta_range[0]) // ((theta_range[1] - theta_range[0])/num_bins))
    theta_dot_i =  int((state[3] - theta_dot_range[0]) // ((theta_dot_range[1] - theta_dot_range[0])/num_bins))
    if np.abs(prev_state[2] - state[2]) < 0.2:
        if idx >= t_lag+1:
            check = 0
            if sensor_actions[idx-t_lag-1] != 0:
                for temp in range(t_lag):
                    check += sensor_actions[idx-temp-1]
                if check == 0:
                    hyper_cube_count[x_i,x_dot_i,theta_i,theta_dot_i] += 1
                    hyper_cube_action[x_i,x_dot_i,theta_i,theta_dot_i] += agent_actions[idx]
    prev_state = state

action_probs = hyper_cube_action/hyper_cube_count
action_probs = (action_probs+epsilon)/(1 + 2*epsilon)
hyper_cube_entropy = -action_probs*np.log2(action_probs) - (1-action_probs)*np.log2(1-action_probs)

hyper_cube_entropy = (hyper_cube_entropy-epsilon)/(1+2*epsilon)

all_entropies = hyper_cube_entropy.reshape(-1)
all_counts = hyper_cube_count.reshape(-1)

valid_entropies = []

bits_H = np.zeros([7,entropy_bins])

for idx in tqdm(range(num_bins**4)):
    if all_counts[idx] > 0:
        valid_entropies.append(all_entropies[idx])

valid_entropies = np.array(valid_entropies)


entropy_range = [0,1]

print(valid_entropies.max())

prev_state = np.array([5,5,5,5])
for idx, state in tqdm(enumerate(true_states)):
    x_i = int((state[0] - x_range[0]) // ((x_range[1] - x_range[0])/num_bins))
    x_dot_i = int((state[1] - x_dot_range[0]) // ((x_dot_range[1] - x_dot_range[0])/num_bins))
    theta_i = int((state[2] - theta_range[0]) // ((theta_range[1] - theta_range[0])/num_bins))
    theta_dot_i =  int((state[3] - theta_dot_range[0]) // ((theta_dot_range[1] - theta_dot_range[0])/num_bins))
    if np.abs(prev_state[2] - state[2]) < 0.2:
        if idx >= t_lag+1:
            check = 0
            if sensor_actions[idx-t_lag-1] != 0:
                for temp in range(t_lag):
                    check += sensor_actions[idx-temp-1]
                if check == 0:
                    if hyper_cube_count[x_i, x_dot_i, theta_i, theta_dot_i] > 0:
                        entropy_idx = int((hyper_cube_entropy[x_i, x_dot_i, theta_i, theta_dot_i] -entropy_range[0]) // ((entropy_range[1]-entropy_range[0])/entropy_bins))
                        bits_H[6-sensor_actions[idx],entropy_idx] += 1
    prev_state = state
print(bits_H)


for i in range(bits_H.shape[1]):
    bits_H[:,i] = bits_H[:,i]/ np.sum(bits_H[:,i])

if beta == '': beta = '015'

plt.figure()
plt.imshow(bits_H, extent=[entropy_range[0], entropy_range[1], 0, 7], aspect="auto", vmin=0, vmax=1)#, interpolation='gaussian')
plt.xlabel('Entropy')
plt.ylabel('Message Length')
plt.colorbar()
plt.title('beta = '+beta+', lag = '+str(t_lag))
plt.savefig('../figures/entropy_vs_sensor_action_'+beta+str(t_lag)+'.png')

import tikzplotlib

tikzplotlib.save('../figures/fig_lags/entropy_vs_sensor_action_A'+beta+str(t_lag)+'.tex')

plt.show()