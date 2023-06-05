import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
import fnmatch
from nn_models.policy import RecA2C
from utils.test_utils import run_episode
import gym
import argparse


parser = argparse.ArgumentParser(description='Test the sensors')

parser.add_argument('--num_episodes', type=int, help='number of episode to test the sensor', required=True)
parser.add_argument('--level', type=str, help='level to test the sensor at', required=True)

args = parser.parse_args()

level = args.level
num_episode_to_test = args.num_episodes

# Search for trained models
list_of_models_names = []
for filename in os.listdir("../models/"):
    if fnmatch.fnmatch(filename, "sensor_level_"+level+"_a2c_*_train.pt"):
        list_of_models_names.append(filename)
        print(filename)

embedding_dim = 8
num_features = 8
latent_dim = embedding_dim*num_features
num_codewords = [1,2,4,8,16,32,64]
quantization_levels = len(num_codewords)

sensor_policy = RecA2C(latent_dim+1, latent_dim, quantization_levels)
# Load models
list_of_models = torch.nn.ModuleList()
for filename in list_of_models_names:
    name = "../models/"+filename
    print(name)
    sensor_policy = RecA2C(latent_dim+1, latent_dim, quantization_levels)
    sensor_policy.load_state_dict(torch.load(name, map_location=torch.device('cpu')).state_dict())
    list_of_models.append(sensor_policy)

total_cost = 0
total_performance = 0 

env = gym.make('CartPole-v1', render_mode = 'rgb_array')

for i in range(len(list_of_models)):
    sensor_policy = list_of_models[i]

    total_cost = 0
    total_performance = 0 
    scores = 0
    for ep in range(num_episode_to_test):
        cost, performance, score, agent_actions, sensor_actions = run_episode(sensor_policy,env,level)
        total_cost += cost
        total_performance += performance
        scores += score

    print(list_of_models_names[i])
    print(total_performance/scores)
    print(scores/num_episode_to_test)
    print(total_cost/scores)

    agent_actions = [agent_actions[-len(sensor_actions)+t][0] for t in range(len(sensor_actions))]
    sensor_actions = [sensor_actions[t][0] for t in range(len(sensor_actions))] 

    assert len(agent_actions) == len(sensor_actions)

    name = list_of_models_names[i]
    end = 0
    while name[19+end] != '_':
        end+=1

    print('beta: ' +name[19:19+end])
    np.save('../save_numpy/agent_actions_level_'+level+'_beta_'+name[19:19+end]+'.npy',np.array(agent_actions))
    np.save('../save_numpy/sensor_actions_level_'+level+'_beta_'+name[19:19+end]+'.npy',np.array(sensor_actions))