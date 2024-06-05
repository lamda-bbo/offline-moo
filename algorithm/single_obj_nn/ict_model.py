import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_seed, get_model_path
import os 

class SimpleMLP(nn.Module):

    def __init__(self,
                 args,
                 seed,
                 idx,
                 in_dim,
                 hid_dim=2048,
                 out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.seed = seed 
        self.args = args 
        self.results_dir = os.path.join(
            self.args.results_dir, f'{seed}'
        )
        os.makedirs(self.results_dir, exist_ok=True)
        self.which_obj = idx 
        self.save_path = get_model_path(
            args, 'single', f'{args.train_model_seed}-{seed}-{idx}'
        )
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).float()

def create_ict_model(args, input_dim, which_obj):
    seeds = [2024, 2025, 2026]
    models = []
    for seed in seeds:
        set_seed(seed)
        models.append(SimpleMLP(args, seed, which_obj, input_dim))
    set_seed(args.train_model_seed)
    return models

