import gym
import numpy as np 
from PIL import Image
from copy import deepcopy
from typing import Optional, Union
import csv
            
import numpy as np 
from PIL import Image
import torch
import os
import pandas as pd
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader

def create_dataset(num_samples, path):
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env.reset()

    num_samples = num_samples

    with open(path+'/description.csv', 'w', encoding="UTF8") as file:
        writer = csv.writer(file)
        for i in range(num_samples):

            env = gym.make('CartPole-v1', render_mode = 'rgb_array')
            env.reset()

            img_path = path+'/images/'

            i_to_string = str(i)

            path1 = img_path + i_to_string + '_0.png'

            env.state[0] = np.random.uniform(-1.2,1.2)
            env.state[1] = np.random.uniform(-2,2)
            env.state[2] = np.random.uniform(-.2075, .2075)
            env.state[3] = np.random.uniform(-3,3)
            screen = env.render()
            screen_height, screen_width, _ = screen.shape
            screen = screen[int(screen_height*0.4):int(screen_height * 0.8), int(screen_width*0.2):int(screen_width*0.8), :]
            im1 = Image.fromarray(screen)
            im1.save(path1)

            action = np.random.randint(2)
            env.step(action)

            path2 = img_path + i_to_string + '_1.png'

            screen = env.render()
            screen_height, screen_width, _ = screen.shape
            screen = screen[int(screen_height*0.4):int(screen_height * 0.8), int(screen_width*0.2):int(screen_width*0.8), :]
            im2 = Image.fromarray(screen)
            im2.save(path2)

            data = [path1, path2, action]
            writer.writerow(data)
            


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img1, img2, action = sample['curr'], sample['next'], sample['action']

        
        img1 = img1.convert('L')
        #img1 = T.Resize(80, interpolation=T.InterpolationMode.BICUBIC)(img1)
        img1 = T.ToTensor()(img1)
        
        img2 = img2.convert('L') 
        #img2 = T.Resize(80, interpolation=T.InterpolationMode.BICUBIC)(img2)
        img2 = T.ToTensor()(img2)

        return {'curr': img1,
                'next': img2,
                'action': torch.from_numpy(np.array(int(action)))}

class FramesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_1_name = self.frame.iloc[idx, 0]
        img_2_name = self.frame.iloc[idx, 1]
        action = self.frame.iloc[idx, 2]
        
        img_1 = Image.open(img_1_name)
        img_2 = Image.open(img_2_name)
        
        sample = {'curr': img_1, 'next': img_2, 'action' : action}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample