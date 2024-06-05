import torch 
import torch.nn as nn 
import numpy as np 

from torch.distributions import Normal
from utils import get_model_path

class DoubleheadModel(nn.Module):
    def __init__(self, input_size, args, idx, hidden_size=64):
        super(DoubleheadModel, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size

        self.args = args
        self.which_obj = idx 
        self.save_path = get_model_path(args, 'single', f'{args.train_model_seed}-{idx}')

        self.max_logstd = nn.Parameter(torch.full((1, 1), np.log(0.2).astype(np.float32)), requires_grad=True)
        self.min_logstd = nn.Parameter(torch.full((1, 1), np.log(0.1).astype(np.float32)), requires_grad=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 2)
        )

        self.softplus = nn.Softplus()
        

    def forward(self, x):
        x = x.float()
        pred = self.dense_layers(x)

        mean, logstd = torch.chunk(pred, 2, dim=-1)

        logstd = self.max_logstd - self.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + self.softplus(logstd - self.min_logstd)

        return mean, logstd
    
    def get_params(self, x):
        mean, logstd = self(x)

        scale = torch.exp(logstd)
        return {'loc': mean, 'scale': scale}
    
    def get_distribution(self, x):
        params = self.get_params(x)
        return Normal(**params)
    
def create_roma_model(input_dim, args, idx, hidden_size=64):
    return DoubleheadModel(input_size=input_dim, args=args, idx=idx, hidden_size=hidden_size)