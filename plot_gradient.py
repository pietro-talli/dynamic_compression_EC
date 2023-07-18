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

    name = '../models/sensor_level_A_a2c_1.0_train.pt'
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

    np.save('../save_numpy/true_states_500_level_A.npy', true_states)
    np.save('../save_numpy/action_agent_500_level_A.npy', np.array(agent_actions))
    np.save('../save_numpy/values_agent_500_level_A.npy', np.array(agent_values))
    np.save('../save_numpy/action_sensor_500_level_A.npy', np.array(sensor_actions))
    np.save('../save_numpy/values_sensor_500_level_A.npy', np.array(sensor_values))

true_states = np.load('../save_numpy/true_states_500_level_A.npy')
agent_actions = np.load('../save_numpy/action_agent_500_level_A.npy')
sensor_actions = np.load('../save_numpy/action_sensor_500_level_A.npy')
agent_values = np.load('../save_numpy/values_agent_500_level_A.npy')
sensor_values = np.load('../save_numpy/values_sensor_500_level_A.npy')

x = true_states[:,0]
x_dot = true_states[:,1]
theta = true_states[:,2]
theta_dot = true_states[:,3]

#x_min = -0.15
#x_max = 0.05

#y_min = -0.1195
#y_max = 0.1095

#num_steps = 5

x_min = -1.97
x_max = 1.97

x = x_dot

y_min = -0.2
y_max = 0.2

num_steps = 7

action_matrix = np.zeros((num_steps,num_steps))
q_matrix = np.zeros((num_steps,num_steps))

a_v_matrix = np.zeros((num_steps, num_steps))
q_v_matrix = np.zeros((num_steps, num_steps))

counter = np.zeros((num_steps,num_steps))

delta_x = (x_max - x_min)/num_steps
delta_y = (y_max - y_min)/num_steps

print(delta_x, delta_y)

mean_v = np.mean(sensor_values)

for sample in range(len(sensor_actions)):
    if x[sample] > x_max or x[sample] < x_min: pass
    else:
        if theta[sample] > y_max or theta[sample] < y_min: pass
        else: 
            i = int( (x_max - x[sample]) // delta_x) 
            j = int( (theta[sample] - y_min) // delta_y) 

            action_matrix[i,j] += agent_actions[sample]
            q_matrix[i,j] += sensor_actions[sample]
            counter[i,j] += 1

            a_v_matrix[i,j] += agent_values[sample]
            q_v_matrix[i,j] += (mean_v - sensor_values[sample])**2

print(sample)

plt.figure()
#plt.subplot(2,2,1)
a = action_matrix/(counter +1e-10)
a = a*np.log(np.e*(counter>0))
plt.imshow(a, extent=[y_min, y_max, x_min, x_max], aspect="auto", interpolation='gaussian')
plt.ylabel('Cart velocity')
plt.xlabel('Angle')
plt.title('Control Actions')

plt.colorbar()

import tikzplotlib
tikzplotlib.save("../figures/control_actions_pos_vs_angle_A.tex")

#plt.savefig('../figures/control_actions_pos_vs_angle.pdf')

plt.figure()
#plt.subplot(2,2,2)
qq = (q_matrix/(counter +1e-10))*np.log(np.e*(counter>0))
plt.imshow(qq, interpolation='gaussian', extent=[y_min, y_max, x_min, x_max], aspect="auto")
plt.ylabel('Cart velocity')
plt.xlabel('Angle')
plt.colorbar()
plt.title('Average Bits per Feature')

#plt.savefig('../figures/bits_per_feature_pos_vs_angle.pdf')

tikzplotlib.save('../figures/bits_per_feature_pos_vs_angle_A.tex')

plt.figure()
#plt.subplot(2,2,3)
entropy = - a*np.log(a+1e-10) - (1-a)*np.log(1-a +1e-10)
entropy =entropy/entropy.max()

plt.imshow(entropy, extent=[y_min, y_max, x_min, x_max], aspect="auto",  interpolation='gaussian')
plt.ylabel('Cart velocity')
plt.xlabel('Angle')
plt.colorbar()
plt.title('Policy Entropy')

#plt.savefig('../figures/policy_entropy_pos_vs_angle.pdf')
tikzplotlib.save('../figures/policy_entropy_pos_vs_angle_A.tex')

#plt.subplot(2,2,4)
#plt.imshow(q_v_matrix/counter, interpolation='gaussian', extent=[y_min, y_max, x_max, x_min], aspect="auto")
#plt.ylabel('Cart velocity')
#plt.xlabel('Angle')
#plt.title('Value STD')
#plt.colorbar()



x_min = -1.67
x_max = 1.67

x = theta_dot

y_min = -0.2
y_max = 0.2

num_steps = 7

action_matrix = np.zeros((num_steps,num_steps))
q_matrix = np.zeros((num_steps,num_steps))

a_v_matrix = np.zeros((num_steps, num_steps))
q_v_matrix = np.zeros((num_steps, num_steps))

counter = np.zeros((num_steps,num_steps))

delta_x = (x_max - x_min)/num_steps
delta_y = (y_max - y_min)/num_steps

print(delta_x, delta_y)

mean_v = np.mean(sensor_values)

for sample in range(len(sensor_actions)):
    if x[sample] > x_max or x[sample] < x_min: pass
    else:
        if theta[sample] > y_max or theta[sample] < y_min: pass
        else: 
            i = int( (x_max - x[sample]) // delta_x) 
            j = int( (theta[sample] - y_min) // delta_y) 

            action_matrix[i,j] += agent_actions[sample]
            q_matrix[i,j] += sensor_actions[sample]
            counter[i,j] += 1

            a_v_matrix[i,j] += agent_values[sample]
            q_v_matrix[i,j] += (mean_v - sensor_values[sample])**2

print(sample)


print(action_matrix)



plt.figure()
#plt.subplot(2,2,1)
a = action_matrix/(counter +1e-10)
a = a*np.log(np.e*(counter>0))
plt.imshow(a, extent=[y_min, y_max, x_min, x_max], aspect="auto", interpolation='gaussian')
plt.ylabel('Pole Angular Velocity')
plt.xlabel('Angle')
plt.title('Control Actions')

plt.colorbar()

#plt.savefig('../figures/control_actions_omega_vs_angle.pdf')
tikzplotlib.save('../figures/control_actions_omega_vs_angle_A.tex')

plt.figure()
#plt.subplot(2,2,2)
qq = (q_matrix/(counter +1e-10))*np.log(np.e*(counter>0))
plt.imshow(qq, interpolation='gaussian', extent=[y_min, y_max, x_min, x_max], aspect="auto")
plt.ylabel('Pole Angular Velocity')
plt.xlabel('Angle')
plt.colorbar()
plt.title('Average Bits per Feature')


tikzplotlib.save('../figures/bits_per_feature_omega_vs_angle_A.tex')

plt.margins(0,0)

#plt.savefig('../figures/bits_per_feature_omega_vs_angle.png')

plt.figure()
#plt.subplot(2,2,3)
entropy = - a*np.log(a+1e-10) - (1-a)*np.log(1-a +1e-10)
entropy =entropy/entropy.max()

plt.imshow(entropy, extent=[y_min, y_max, x_min, x_max], aspect="auto",  interpolation='gaussian')
plt.ylabel('Pole Angular Velocity')
plt.xlabel('Angle')
plt.colorbar()
plt.title('Policy Entropy')

#plt.savefig('../figures/policy_entropy_omega_vs_angle.pdf')
tikzplotlib.save('../figures/policy_entropy_omega_vs_angle_A.tex')


#plt.subplot(2,2,4)
#plt.imshow(q_v_matrix/counter, interpolation='gaussian', extent=[y_min, y_max, x_max, x_min], aspect="auto")
#plt.ylabel('Pole Angular Velocity')
#plt.xlabel('Angle')
#plt.title('Value STD')
#plt.colorbar()

plt.show()