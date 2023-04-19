import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicalValueRegressor(nn.Module):
    def __init__(self, n_inputs, n_output):
        super().__init__()
        self.linear1 = nn.Linear(in_features=n_inputs, out_features=n_inputs)
        self.linear2 = nn.Linear(in_features=n_inputs, out_features=n_inputs)
        self.output = nn.Linear(in_features=n_inputs, out_features=n_output)
            
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)