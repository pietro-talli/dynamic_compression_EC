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

if retrain:
    latent_dim = 64
    quantization_levels = 7
    sensor_policy = RecA2C(latent_dim+1, latent_dim, quantization_levels)

    name = '../models/sensor_level_C_a2c_0.15_train.pt'
    env = gym.make('CartPole-v1',render_mode='rgb_array')

    sensor_policy.load_state_dict(torch.load(name, map_location=torch.device('cpu')).state_dict())

    true_states = []
    for i in range(100):
        cost, ep_reward, score, agent_actions, sensor_actions, true_states = run_episode_for_gradient(sensor_policy,env,'C', true_states)

    agent_actions = [agent_actions[t][0] for t in range(len(sensor_actions))]
    sensor_actions = [sensor_actions[t][0] for t in range(len(sensor_actions))] 

    assert len(agent_actions) == len(sensor_actions)

    true_states = np.concatenate(true_states, 0)

    np.save('../save_numpy/true_states.npy', true_states)
    np.save('../save_numpy/action_agent.npy', np.array(agent_actions))
    np.save('../save_numpy/action_sensor.npy', np.array(sensor_actions))

true_states = np.load('../save_numpy/true_states.npy')
agent_actions = np.load('../save_numpy/action_agent.npy')
sensor_actions = np.load('../save_numpy/action_sensor.npy')

x = true_states[:,0]
x_dot = true_states[:,1]
theta = true_states[:,2]
theta_dot = true_states[:,3]

x_min = -0.15
x_max = 0.05

y_min = -0.1195
y_max = 0.1095

num_steps = 5
action_matrix = np.zeros((num_steps,num_steps))
q_matrix = np.zeros((num_steps,num_steps))
counter = np.zeros((num_steps,num_steps))

delta_x = (x_max - x_min)/num_steps
delta_y = (y_max - y_min)/num_steps

print(delta_x, delta_y)

for sample in range(len(sensor_actions)):
    if x[sample] > x_max or x[sample] < x_min: pass
    else:
        if theta[sample] > y_max or theta[sample] < y_min: pass
        else: 
            i = int( (x[sample] - x_min) // delta_x) 
            j = int( (theta[sample] - y_min) // delta_y) 

            action_matrix[i,j] += agent_actions[sample]
            q_matrix[i,j] += sensor_actions[sample]
            counter[i,j] += 1
print(sample)
plt.figure()

a = action_matrix/(counter +1e-10)
a = a*np.log(np.e*(counter>0))
plt.imshow(a, interpolation='gaussian')
plt.colorbar()

plt.figure()
plt.imshow((q_matrix/(counter +1e-10))*np.log(np.e*(counter>0)), interpolation='gaussian')
plt.colorbar()

plt.show()

