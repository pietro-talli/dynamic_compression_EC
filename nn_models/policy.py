import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_sequence, unpack_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecDQN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_actions):
        super().__init__()
        
        self.rnn = nn.GRU(input_size, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_actions)

    def forward(self,x):
        """
        x is a list or a Tensor, where each item is a tensor of (L,input_size),
        where L is not fixed, but can be variable
        """
        packed_input = pack_sequence(x, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input.to(device))

        rec_out = unpack_sequence(packed_output)
        last_rec = torch.cat([t[-1,:].unsqueeze(0) for t in rec_out]) 
        return self.linear2(F.relu(self.linear1(F.relu(last_rec))))
    
class RecA2C(nn.Module):
    def __init__(self, input_size, hidden_dim, num_actions):
        super().__init__()

        self.rnn = nn.GRU(input_size, hidden_dim)
        self.linear_value = nn.Linear(hidden_dim,hidden_dim)
        self.linear_policy = nn.Linear(hidden_dim,hidden_dim)

        self.value = nn.Linear(hidden_dim,1)
        self.policy = nn.Linear(hidden_dim,num_actions)

        self.saved_actions = []
        self.rewards = []

    def forward(self,x):
        """
        x is a Tensor of (L,input_size),
        """
        
        rnn_out, _ = self.rnn(x.to(device))
        last_rec = rnn_out[-1,:]

        value = self.value(F.relu(self.linear_value(last_rec)))
        policy = F.softmax(self.policy(F.relu(self.linear_policy(last_rec))),dim=-1)

        return policy, value