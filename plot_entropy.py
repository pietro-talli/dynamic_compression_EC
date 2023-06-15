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

args = parser.parse_args()

if args.retrain: retrain = args.retrain

from tqdm import tqdm

if retrain:
    latent_dim = 64
    quantization_levels = 7
    sensor_policy = RecA2C(latent_dim+1, latent_dim, quantization_levels)

    name = '../models/sensor_level_C_a2c_0.07_train.pt'
    env = gym.make('CartPole-v1',render_mode='rgb_array')

    sensor_policy.load_state_dict(torch.load(name, map_location=torch.device('cpu')).state_dict())

    true_states = []
    for i in tqdm(range(500)):
        cost, ep_reward, score, aa, sa, true_states = run_episode_for_gradient(sensor_policy,env,'C', true_states)

    agent_actions = [aa[t][0] for t in range(len(sa))]
    agent_values = [aa[t][1] for t in range(len(sa))]

    sensor_actions = [sa[t][0] for t in range(len(sa))] 
    sensor_values = [sa[t][1] for t in range(len(sa))] 

    assert len(agent_actions) == len(sensor_actions)

    true_states = np.concatenate(true_states, 0)

    np.save('../save_numpy/true_states_500_007.npy', true_states)
    np.save('../save_numpy/action_agent_500_007.npy', np.array(agent_actions))
    np.save('../save_numpy/values_agent_500_007.npy', np.array(agent_values))
    np.save('../save_numpy/action_sensor_500_007.npy', np.array(sensor_actions))
    np.save('../save_numpy/values_sensor_500_007.npy', np.array(sensor_values))

true_states = np.load('../save_numpy/true_states_500_005.npy')
agent_actions = np.load('../save_numpy/action_agent_500_005.npy')
sensor_actions = np.load('../save_numpy/action_sensor_500_005.npy')
agent_values = np.load('../save_numpy/values_agent_500_005.npy')
sensor_values = np.load('../save_numpy/values_sensor_500_005.npy')

x = true_states[:,0]
x_dot = true_states[:,1]
theta = true_states[:,2]
theta_dot = true_states[:,3]

num_bins = 10

hyper_cube_count = np.zeros([num_bins,num_bins,num_bins,num_bins])
hyper_cube_action = np.zeros([num_bins,num_bins,num_bins,num_bins])
hyper_cube_entropy = np.zeros([num_bins,num_bins,num_bins,num_bins])
hyper_cube_bits = np.zeros([num_bins,num_bins,num_bins,num_bins,7])
bits_H = np.zeros([num_bins, 7])

epsilon = 1e-8

x_range = [x.min() - epsilon, x.max() + epsilon]
x_dot_range = [x_dot.min() - epsilon,x_dot.max() + epsilon]
theta_range = [theta.min() - epsilon,theta.max() + epsilon]
theta_dot_range = [theta_dot.min() - epsilon,theta_dot.max() + epsilon]

from tqdm import tqdm

for idx, state in tqdm(enumerate(true_states)):
    x_i = int((state[0] - x_range[0]) // ((x_range[1] - x_range[0])/num_bins))
    x_dot_i = int((state[1] - x_dot_range[0]) // ((x_dot_range[1] - x_dot_range[0])/num_bins))
    theta_i = int((state[2] - theta_range[0]) // ((theta_range[1] - theta_range[0])/num_bins))
    theta_dot_i =  int((state[3] - theta_dot_range[0]) // ((theta_dot_range[1] - theta_dot_range[0])/num_bins))

    hyper_cube_count[x_i,x_dot_i,theta_i,theta_dot_i] += 1
    hyper_cube_action[x_i,x_dot_i,theta_i,theta_dot_i] += agent_actions[idx]
    hyper_cube_bits[x_i,x_dot_i,theta_i,theta_dot_i,sensor_actions[idx]] += 1

action_probs = hyper_cube_action/hyper_cube_count
action_probs = (action_probs+epsilon)/(1 + 2*epsilon)
hyper_cube_entropy = -action_probs*np.log2(action_probs) - (1-action_probs)*np.log2(1-action_probs)


for idx, state in tqdm(enumerate(true_states)):
    x_i = int((state[0] - x_range[0]) // ((x_range[1] - x_range[0])/num_bins))
    x_dot_i = int((state[1] - x_dot_range[0]) // ((x_dot_range[1] - x_dot_range[0])/num_bins))
    theta_i = int((state[2] - theta_range[0]) // ((theta_range[1] - theta_range[0])/num_bins))
    theta_dot_i =  int((state[3] - theta_dot_range[0]) // ((theta_dot_range[1] - theta_dot_range[0])/num_bins))

    entropy_idx = hyper_cube_entropy[x_i, x_dot_i, theta_i, theta_dot_i]
    bits_H[entropy_idx, sensor_actions[idx]] += 1

avg_bits = hyper_cube_bits/np.sum(hyper_cube_bits)


all_entropies = hyper_cube_entropy.reshape(-1)
all_bits = avg_bits.reshape(-1)
all_counts = hyper_cube_count.reshape(-1)

valid_entropies = []
valid_bits = []

for idx in tqdm(range(num_bins**4)):
    if all_counts[idx] > 50:
        valid_entropies.append(all_entropies[idx])
        valid_bits.append(all_bits[idx])

valid_bits = np.array(valid_bits)
valid_entropies = np.array(valid_entropies)

x_bins = 4
y_bins = 4

final_matrix = np.zeros((x_bins,y_bins))

entropy_range = [valid_entropies.min() -epsilon, valid_entropies.max() + epsilon]
bits_range = [valid_bits.min() -epsilon, valid_bits.max() + epsilon]

print(valid_entropies)

for idx in range(valid_entropies.shape[0]):
    entropy_index = int((valid_entropies[idx] - entropy_range[0]) // ((entropy_range[1] - entropy_range[0])/x_bins))
    bits_index =  int((valid_bits[idx] - bits_range[0]) // ((bits_range[1] - bits_range[0])/y_bins))

    final_matrix[entropy_index,bits_index] += 1

for i in range(x_bins):
    final_matrix[i] = final_matrix[i]/final_matrix[i].sum()


fm = np.zeros_like(final_matrix)

for i in range(y_bins):
    fm[:,y_bins-1-i] = final_matrix[:,i]

plt.figure()
plt.imshow(fm.T, extent=[entropy_range[0], entropy_range[1], bits_range[0], bits_range[1]], aspect="auto", interpolation='gaussian')
plt.xlabel('Entropy')
plt.ylabel('Message Length')
plt.colorbar()
plt.show()