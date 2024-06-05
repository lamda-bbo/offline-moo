import torch.nn as nn
import torch
import numpy as np
from utils import get_model_path

activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]

class MultiHeadModel(nn.Module):
    def __init__(self, n_dim, n_obj, args, hidden_size):
        super(MultiHeadModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.args = args
        self.env_name = args.env_name
        self.activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]
        self.save_path = get_model_path(args=args, model_type='multi_head', 
                                        name=f'{args.train_model_seed}')
        self.device = args.device

        self.feature_extractor = FeatureExtractor(n_dim, [2048], 2048, args)
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.obj2head = {}

    def create_head(self, learning_obj):
        new_head = Head(input_size=2048, hidden_size=[], args=self.args,
                        which_obj=learning_obj)
        new_head = new_head.to(self.device)
        self.obj2head[learning_obj] = new_head

    def forward(self, x, forward_objs):
        h = self.feature_extractor(x)
        q = [self.obj2head[obj](h) for obj in forward_objs]
        q = torch.cat(q, dim=1)
        return q


class FeatureExtractor(nn.Module):
    def __init__(self, n_dim, hidden_size, output_size, args):
        super(FeatureExtractor, self).__init__()
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.args = args
        self.env_name = args.env_name
        self.activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]
        self.save_path = get_model_path(args=args, model_type='multi_head', 
                                        name=f'{args.train_model_seed}-feature_extractor')

        layers = []
        layers.append(nn.Linear(n_dim, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size)-1], output_size))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size
        self.device = args.device
        self = self.to(self.device)

    def forward(self, x):
        '''
        BUG: It may raise dtype Runtime Error while conducting Multi-Head + MOBO, I suggest using the 'try' part.
        '''
        x = x.to(torch.float32)
        for i in range(len(self.hidden_size)+1):
            x = self.layers[i](x)
            x = self.activation_functions[i](x)

        return x.to(torch.float32)
        # try:
        #     x = x.to(torch.float64)
        #     for i in range(len(self.hidden_size)+1):
        #         x = self.layers[i](x)
        #         x = self.activation_functions[i](x)

        #     return x.to(torch.float64)
        # except:
        #     x = x.to(torch.float32)
        #     for i in range(len(self.hidden_size)+1):
        #         x = self.layers[i](x)
        #         x = self.activation_functions[i](x)

        #     return x.to(torch.float32)
    
class Head(nn.Module):
    def __init__(self, input_size, hidden_size, args, which_obj):
        super(Head, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.which_obj = which_obj
        self.args = args
        self.env_name = args.env_name
        self.activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]
        self.save_path = get_model_path(args=args, model_type='multi_head', 
                                        name=f'{args.train_model_seed}-head-{which_obj}')

        layers = []
        if hidden_size != []:
            layers.append(nn.Linear(input_size, hidden_size[0]))
            for i in range(len(hidden_size)-1):
                layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(nn.Linear(hidden_size[len(hidden_size)-1], 1))
        else:
            layers.append(nn.Linear(input_size, 1))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

    def forward(self, x):
        '''
        BUG: It may raise dtype Runtime Error while conducting Multi-Head + MOBO, we suggest using the 'try' part
        '''
        x = x.to(torch.float32)
        for i in range(len(self.hidden_size)+1):
            x = self.layers[i](x)
            x = self.activation_functions[i](x)

        return x.to(torch.float32)
        # try:
        #     x = x.to(torch.float64)
        #     for i in range(len(self.hidden_size)+1):
        #         x = self.layers[i](x)
        #         x = self.activation_functions[i](x)

        #     return x.to(torch.float64)
        # except:
        #     x = x.to(torch.float32)
        #     for i in range(len(self.hidden_size)+1):
        #         x = self.layers[i](x)
        #         x = self.activation_functions[i](x)

        #     return x.to(torch.float32)