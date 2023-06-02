import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
import fnmatch
from nn_models.policy import RecA2C
from utils.test_utils import run_episode

level = "A"

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
    sensor_policy.load_state_dict(torch.load(name, map_location=torch.device('cpu')).state_dict())
    list_of_models.append(sensor_policy)

num_episode_to_test = 100

total_cost = 0
total_performance = 0 



for i in range(num_episode_to_test):
    sensor_policy = list_of_models[i]
    cost, performance = run_episode(sensor_policy)
    total_cost += cost
    total_performance += performance



