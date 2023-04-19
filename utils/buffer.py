import torch
import numpy as np
from collections import deque
import random

class Episode:
    def __init__(self):
        self.length = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add_transition(self, state, action ,reward, next_state, done):
        self.length += 1
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        

class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.episode_queue = deque(maxlen=self.memory_size)

    def __len__(self,):
        return len(self.episode_queue)

    def add(self, episode):
        self.episode_queue.append(episode)

    def sample(self, num_episodes, episode_maxlen):
        batch_of_episodes = random.sample(self.episode_queue, num_episodes)

        sequences_of_states = []
        sequences_of_next_states = []
        actions = []
        rewards = []
        dones = []

        for episode in batch_of_episodes:
            time_step = np.random.randint(1, episode.length)
            if time_step - episode_maxlen > 0:
                tensor_of_states = torch.cat(episode.states[time_step-episode_maxlen:time_step])
                tensor_of_next_states = torch.cat(episode.next_states[time_step-episode_maxlen:time_step])
            else:
                tensor_of_states = torch.cat(episode.states[0:time_step])
                tensor_of_next_states = torch.cat(episode.next_states[0:time_step])

            actions.append(episode.actions[time_step])
            rewards.append(episode.rewards[time_step])
            dones.append(episode.dones[time_step])
            sequences_of_states.append(tensor_of_states)
            sequences_of_next_states.append(tensor_of_next_states)
        
        return sequences_of_states, actions, rewards, sequences_of_next_states, dones

        
