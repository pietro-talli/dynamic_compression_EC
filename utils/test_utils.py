import torch
import numpy as np
import gym
from nn_models.policy import RecA2C

env = gym.make("CartPole-v1", render_mode = 'rgb_array')
embedding_dim = 8
num_f = 8
latent_dim = embedding_dim*num_f
agent_policy = RecA2C(latent_dim,latent_dim,env.action_space.n)

agent_policy.load_state_dict("../models/policy_a2c_64.py")

def run_episode(sensor_policy):
    
    return cost, performance