import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhysicalValueRegressor(nn.Module):
    def __init__(self, n_inputs, n_output):
        super().__init__()
        self.rnn = nn.GRU(n_inputs, n_inputs)
        self.linear1 = nn.Linear(in_features=n_inputs, out_features=n_inputs)
        self.output = nn.Linear(in_features=n_inputs, out_features=n_output)
            
    def forward(self, x):
        rnn_out, _ = self.rnn(x.to(device))
        last_rec = rnn_out[-1,:]
        x = F.relu(self.linear1(last_rec))
        return self.output(x)