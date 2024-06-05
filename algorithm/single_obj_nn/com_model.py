import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from utils import get_model_path

class COM_NET(nn.Module):

    def __init__(self, n_dim, n_obj, args, idx,
                 hid_dim=2048,
                 out_dim=1):
        super(COM_NET, self).__init__()
        self.fc1 = nn.Linear(n_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

        self.n_dim = n_dim
        self.n_obj = n_obj
        self.env_name = args.env_name 
        self.args = args
        self.which_obj = idx
        self.save_path = get_model_path(args, 'single', f'{args.train_model_seed}-{idx}')

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return self.fc3(x).to(torch.float32)