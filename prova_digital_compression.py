from utils.utilities_dataset import FramesDataset, ToTensor
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import gym

from nn_models.sensor import SensorDigital

sensor = SensorDigital()

path = '../dataset'
num_samples = 1000

dataset = FramesDataset('../dataset/description.csv', '../dataset/images', transform=ToTensor())

first_sample = dataset.__getitem__(np.random.randint(0, num_samples))

img = first_sample['curr']

# save the image 
img = img.numpy()
img = img.transpose(1,2,0)

gray = img[:,:,0]
h,w = gray.shape

env = gym.make('CartPole-v1', render_mode='rgb_array')
env.reset()
img = env.render()

encoded, _ = sensor(env)
print(encoded.shape)