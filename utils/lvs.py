import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import namedtuple, deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
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

def select_action_policy(state, model):
    probs, state_value = model(state)
    action = torch.argmax(probs)
    return action.item()

def finish_episode(model, optimizer, gamma):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        temp = torch.tensor([R])
        value_losses.append(F.smooth_l1_loss(value, temp.to(device)))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2)
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]
    return loss.item()

from nn_models.sensor import Sensor_not_quantized, Sensor_not_quantized_level_A


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sensor_3_levels(model, env, sensor_policy, list_of_quantizers, level, num_episodes, beta, encoder, gamma = 0.99):
    optimizer = torch.optim.Adam(sensor_policy.parameters(), lr=3e-4)
    writer = SummaryWriter('../runs_sensor_level_'+level+'/a2c'+str(beta))
    sensor = Sensor_not_quantized(encoder=encoder)

    model.to(device)
    sensor.to(device)
    sensor_policy.to(device)
    list_of_quantizers.to(device)
    list_of_quantizers.eval()
    for episode in range(num_episodes):
        _ = env.reset()
        ep_reward = 0
        done = False

        prev_screen = None
        with torch.no_grad():
            state_not_quantized, curr_screen = sensor(env, prev_screen)
        states = deque(maxlen = 20)
        states_quantized = deque(maxlen=20)
        score = 0
        q = -1
        while not done:
            q_tensor = torch.tensor([q])
            s_and_prev_q = torch.cat([state_not_quantized.reshape(-1), q_tensor.to(device)])
            states.append(s_and_prev_q.reshape(1,-1))
            input_state = torch.cat(list(states),0)

            q = select_action(input_state.to(device), sensor_policy)

            with torch.no_grad():
                if q == 0 and score == 0:
                    q = 1
                if q > 0:
                    _,state_quantized,_,_ = list_of_quantizers[q-1](state_not_quantized)
                if q == 0 and score >0:
                    state_quantized = state_quantized
                states_quantized.append(state_quantized.reshape(1,-1))
                input_state_quantized = torch.cat(list(states_quantized),0)
                action = select_action_policy(input_state_quantized.to(device), model)


            state, reward, done, _, _ = env.step(action)
            prev_screen = curr_screen
            with torch.no_grad():
                state_not_quantized, curr_screen = sensor(env, prev_screen)
            
            
            x, x_dot, theta, theta_dot = env.state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2 - (beta*q)

            sensor_policy.rewards.append(reward)
            ep_reward += reward
            score += 1
            if score >= 500:
                done = True

        loss = finish_episode(sensor_policy, optimizer, gamma)
        writer.add_scalar('Performance/Score', score, episode)
        writer.add_scalar('Loss/train', loss, episode)
        
        del model.saved_actions[:]

        if episode%1000 == 0:
            torch.save(sensor_policy,'../models/sensor_level_'+level+'_a2c_'+str(beta)+'_train.pt')

    return sensor_policy

def sensor_2_levels(model, env, sensor_policy, list_of_quantizers, level, num_episodes, beta, encoder, regressors, gamma = 0.99):
    optimizer = torch.optim.Adam(sensor_policy.parameters(), lr=3e-4)
    writer = SummaryWriter('../runs_sensor_level_'+level+'/a2c'+str(beta))
    sensor = Sensor_not_quantized(encoder=encoder)

    model.to(device)
    sensor.to(device)
    sensor_policy.to(device)
    list_of_quantizers.to(device)
    regressors.to(device)

    for episode in range(num_episodes):
        _ = env.reset()
        ep_reward = 0
        done = False

        prev_screen = None
        with torch.no_grad():
            state_not_quantized, curr_screen = sensor(env, prev_screen)
        states = deque(maxlen = 20)
        states_quantized = deque(maxlen=20)
        score = 0
        q = -1
        while not done:
            q_tensor = torch.tensor([q])
            s_and_prev_q = torch.cat([state_not_quantized.reshape(-1), q_tensor.to(device)])
            states.append(s_and_prev_q.reshape(1,-1))
            input_state = torch.cat(list(states),0)

            q = select_action(input_state.to(device), sensor_policy)

            with torch.no_grad():
                if q == 0 and score == 0:
                    q = 1
                if q > 0:
                    _,state_quantized,_,_ = list_of_quantizers[q-1](state_not_quantized)
                if q == 0 and score >0:
                    state_quantized = state_quantized
                states_quantized.append(state_quantized.reshape(1,-1))
                input_state_quantized = torch.cat(list(states_quantized),0)
                action = select_action_policy(input_state_quantized.to(device), model)

            state = env.state
            with torch.no_grad():
                state_tensor = torch.tensor(state)
                reward = -F.mse_loss(regressors[-1](input_state_quantized), state_tensor.to(device)) - (beta*q)

            sensor_policy.rewards.append(reward)
            
            state, reward, done, _, _ = env.step(action)
            prev_screen = curr_screen
            with torch.no_grad():
                state_not_quantized, curr_screen = sensor(env, prev_screen)
            
            ep_reward += reward
            score += 1
            if score >= 500:
                done = True

        loss = finish_episode(sensor_policy, optimizer, gamma)
        writer.add_scalar('Performance/Score', score, episode)
        writer.add_scalar('Loss/train', loss, episode)
        
        if episode%1000 == 0:
            torch.save(sensor_policy,'../models/sensor_level_'+level+'_a2c_'+str(beta)+'_train.pt')

    return sensor_policy

def sensor_1_levels(model, env, sensor_policy, list_of_quantizers, level, num_episodes, beta, encoder, decoder, gamma = 0):
    optimizer = torch.optim.Adam(sensor_policy.parameters(), lr=3e-4)
    writer = SummaryWriter('../runs_sensor_level_'+level+'/a2c'+str(beta))
    sensor = Sensor_not_quantized_level_A(encoder=encoder)

    model.to(device)
    sensor.to(device)
    sensor_policy.to(device)
    list_of_quantizers.to(device)
    decoder.to(device)

    for episode in range(num_episodes):
        _ = env.reset()
        ep_reward = 0
        done = False

        prev_screen = None
        with torch.no_grad():
            state_not_quantized, curr_screen, frames = sensor(env, prev_screen)
        states = deque(maxlen = 20)
        states_quantized = deque(maxlen=20)
        score = 0
        q = -1
        while not done:
            q_tensor = torch.tensor([q])
            s_and_prev_q = torch.cat([state_not_quantized.reshape(-1), q_tensor.to(device)])
            states.append(s_and_prev_q.reshape(1,-1))
            input_state = torch.cat(list(states),0)

            q = select_action(input_state.to(device), sensor_policy)

            with torch.no_grad():
                if q == 0 and score == 0:
                    q = 1
                if q > 0:
                    _,state_quantized,_,_ = list_of_quantizers[q-1](state_not_quantized)
                if q == 0 and score >0:
                    state_quantized = state_quantized
                states_quantized.append(state_quantized.reshape(1,-1))
                input_state_quantized = torch.cat(list(states_quantized),0)
                action = select_action_policy(input_state_quantized.to(device), model)

            with torch.no_grad():
                reward = -10*torch.log10(F.mse_loss(decoder(state_quantized), frames.to(device))) - (beta*q)

            sensor_policy.rewards.append(reward)

            state, reward, done, _, _ = env.step(action)
            prev_screen = curr_screen
            with torch.no_grad():
                state_not_quantized, curr_screen, frames = sensor(env, prev_screen)

            ep_reward += reward
            score += 1
            if score >= 500:
                done = True

        loss = finish_episode(sensor_policy, optimizer, gamma)
        writer.add_scalar('Performance/Score', score, episode)
        writer.add_scalar('Loss/train', loss, episode)
        
        if episode%1000 == 0:
            torch.save(sensor_policy,'../models/sensor_level_'+level+'_a2c_'+str(beta)+'_train.pt')

    return sensor_policy