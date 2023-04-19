import torch
import gym
import numpy as np
import random
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import Episode, Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exp_p(num_episodes: int, exp_frac: float):
    # explore for the 50% of the number of steps with decreasing epsilon
    eps = 0.005*np.ones(num_episodes)
    exploration_steps = np.linspace(1,0.005,int(np.floor(num_episodes*exp_frac)))
    eps[0:len(exploration_steps)] = exploration_steps
    return eps

def choose_action_epsilon_greedy(net, state: list, epsilon: float):
    
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()

def rl_training_loop(env: gym.Env,
                     dqn,
                     dqn_target,
                     sensor,
                     num_episodes: int,
                     batch_size: int,
                     gamma: float,
                     exp_frac: float,
                     target_net_update_steps: int,
                     beta: float
                     ):
    writer = SummaryWriter('../runs_policy')
    loss_fn = nn.SmoothL1Loss()
    env.reset(seed = 0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    memory = Memory(10000)
    dqn.to(device)
    dqn_target.to(device)
    dqn_target.load_state_dict(dqn.state_dict()) #initialize the target network with the same parameters of the policy network

    exploration_profile = exp_p(num_episodes, exp_frac)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-4)

    for episode_num in range(num_episodes):
        # Reset the environment and get the initial state
        _ = env.reset(seed = episode_num)
        epsilon = exploration_profile[episode_num]
        score = 0
        done = False
        episode = Episode()
        states = []
        losses = [0]
        prev_screen = None
        state_received, curr_screen = sensor(env, prev_screen)
        state_received = state_received.detach()
        while not done:
            
            states.append(state_received)
            input_state = torch.cat(states,0)

            action, _ = choose_action_epsilon_greedy(dqn, [input_state], epsilon)
            _, _, done, _, _ = env.step(action)
            x, x_dot, theta, theta_dot = env.state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            score += 1      

            next_state_received, prev_screen = sensor(env, prev_screen)
            next_state_received = next_state_received.detach()
            if done: # if the pole has fallen down 
                reward = 0
                next_state_received = None
            # Update the replay memory
            episode.add_transition(state_received, action, reward, next_state_received, done)

            state_received = next_state_received

            # Update the network
            if len(memory) > batch_size: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
                loss = update_step_rec(dqn, dqn_target, memory, gamma, optimizer, loss_fn, batch_size)
                losses.append(loss)

        # Update the target network every target_net_update_steps episodes  
        memory.add(episode)
        
        if episode_num % target_net_update_steps == 0:
            dqn_target.load_state_dict(dqn.state_dict()) 
            for name, param in dqn_target.named_parameters():
                param.data = beta*param.data + (1-beta)*dqn.state_dict()[name]
        # Print the final score
        writer.add_scalar('Performance/Score', score, episode_num)
        writer.add_scalar('Strategy/Epsilon', epsilon, episode_num)
        writer.add_scalar('Loss/train', np.mean(losses), episode_num)

    return dqn     


def update_step_rec(dqn,dqn_target, memory: Memory,gamma,optimizer,loss_fn,batch_size):
    dqn.train()

    sequences_of_states, actions, rewards, sequences_of_next_states, dones = memory.sample(batch_size, 5) 

    q_values = dqn(sequences_of_states)
    q_values_next = dqn_target(sequences_of_next_states).detach()
    next_state_max_q_values = q_values_next.max(dim=-1)[0]

    rewards = torch.FloatTensor(rewards, device=device)
    actions = torch.LongTensor(actions)

    expected_state_action_values = torch.tensor(rewards) + (next_state_max_q_values * gamma)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    loss = loss_fn(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(dqn.parameters(), 2)
    optimizer.step()

    return loss.item()