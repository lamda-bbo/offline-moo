import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_model_path

class DiscriminatorModel(nn.Module):
    def __init__(self, args, idx, input_size, hidden_size=1024, final_tanh=False):
        super(DiscriminatorModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() if not final_tanh else nn.Tanh()
        )
        self.args = args
        self.seed = args.train_model_seed
        self.results_dir = os.path.join(self.args.results_dir, f'{self.seed}')
        os.makedirs(self.results_dir, exist_ok=True)
        self.which_obj = idx 
        self.save_path = get_model_path(
            args, 'single', f'{args.train_model_seed}-{idx}-Discriminator'
        )

    def forward(self, x):
        return self.layers(x.float()).float()

class ForwardModel(nn.Module):
    def __init__(self, args, idx, input_size, activations=('relu', 'relu'), hidden_size=1024, final_tanh=False):
        super(ForwardModel, self).__init__()
        self.layers = nn.Sequential()
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
        }
        for act in activations:
            act_fn = activation_functions.get(act, act) if isinstance(act, str) else act()
            self.layers.append(nn.Linear(input_size if len(self.layers) == 0 else hidden_size, hidden_size))
            self.layers.append(act_fn)
        self.layers.append(nn.Linear(hidden_size, 1))
        if final_tanh:
            self.layers.append(nn.Tanh())
        self.args = args
        self.seed = args.train_model_seed
        self.results_dir = os.path.join(self.args.results_dir, f'{self.seed}')
        os.makedirs(self.results_dir, exist_ok=True)
        self.which_obj = idx 
        self.save_path = get_model_path(
            args, 'single', f'{args.train_model_seed}-{idx}-Forward'
        )

    def forward(self, x):
        return self.layers(x.float()).float()

class RepModel(nn.Module):
    def __init__(self, args, idx, input_size, output_shape, activations=('relu', 'relu'), hidden_size=2048, final_tanh=False):
        super(RepModel, self).__init__()
        self.layers = nn.Sequential()
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
        }
        for act in activations:
            act_fn = activation_functions.get(act, act) if isinstance(act, str) else act()
            self.layers.append(nn.Linear(input_size if len(self.layers) == 0 else hidden_size, hidden_size))
            self.layers.append(act_fn)
        self.layers.append(nn.Linear(hidden_size, np.prod(output_shape)))
        if final_tanh:
            self.layers.append(nn.Tanh())

        self.args = args
        self.seed = args.train_model_seed
        self.results_dir = os.path.join(self.args.results_dir, f'{self.seed}')
        os.makedirs(self.results_dir, exist_ok=True)
        self.which_obj = idx 
        self.save_path = get_model_path(
            args, 'single', f'{args.train_model_seed}-{idx}-Representation'
        )

    def forward(self, x):
        return self.layers(x.float()).float()


def create_iom_model(args, input_dim, which_obj):
    output_shape = 128
    models = {} 
    models['DiscriminatorModel'] = DiscriminatorModel(args, 
                                                      which_obj,
                                                      output_shape,
                                                      hidden_size=512,
                                                      final_tanh=False)
    models['ForwardModel'] = ForwardModel(args, 
                                          which_obj, 
                                          output_shape,
                                          activations=['relu', 'relu'],
                                          hidden_size=2048,
                                          final_tanh=False)
    models['RepModel'] = RepModel(args, 
                                  which_obj, 
                                  input_dim,
                                  output_shape,
                                  hidden_size=2048,
                                  activations=['relu', 'relu'],
                                  final_tanh=False)
    return models

    