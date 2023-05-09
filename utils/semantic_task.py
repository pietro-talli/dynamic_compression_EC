import torch
import gym
import numpy as np
import random
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import Episode, Memory
from itertools import count
from collections import namedtuple, deque
from torch.distributions import Categorical
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### CODE FOR ACTOR CRITIC

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

def select_action(state, model):
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()

def A2C(env, model, sensor, regressor, num_episodes: int, num_codewords: int = 64):

    optimizer = torch.optim.Adam(regressor.parameters(), lr=3e-4)
    writer = SummaryWriter('../runs_regression/regressor'+str(num_codewords))

    mses = []
    for episode in range(num_episodes):
        true_state, _ = env.reset()
        done = False

        prev_screen = None
        with torch.no_grad():
            state_received, curr_screen = sensor(env, prev_screen)
        state_received = state_received
        states = deque(maxlen = 20)
        score = 0

        while not done:
            states.append(state_received)
            input_state = torch.cat(list(states),0)

            estimated_true_state = regressor(input_state)
            mses.append(F.mse_loss(estimated_true_state, true_state))

            action = select_action(input_state, model)
            true_state, _, done, _, _ = env.step(action)
            prev_screen = curr_screen
            with torch.no_grad():
                state_received, curr_screen = sensor(env, prev_screen)
            if score >= 500:
                done = True

        batch_mse = torch.cat(mses).sum()

        optimizer.zero_grad()
        batch_mse.backward()
        optimizer.step()

        writer.add_scalar('Performance/Score', score, episode)
        writer.add_scalar('Loss/train', batch_mse.item(), episode)
        
    return model