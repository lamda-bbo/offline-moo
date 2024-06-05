from pymoo.core.problem import Problem
import numpy as np
import torch

class SingleObjSurrogateProblem(Problem):
    def __init__(self, n_var, n_obj, model, device, args):
        super().__init__(
            n_var = n_var, 
            n_obj = n_obj, 
            xl = 0,
            xu = 1
        )
        self.args = args 
        if args.train_mode not in ['ict', 'tri_mentoring', 'iom']:
            self.model = [model0.to(device) for model0 in model] \
                if isinstance(model, list) else model.to(device)
        elif args.train_mode == 'iom':
            self.model = [{name: model0.to(device) for name, model0 in model_dict.items()}
                          for model_dict in model]
        else:
            self.model = [[model0.to(device) for model0 in model_list] for model_list in model]
        # for model in model:
        #     if not isinstance(model, list):
        #         self.model = model.to(device)
        #     else:
        #         models = []
        #         for model0 in model:
        #             if not isinstance(model0, list):
        #                 model0.to(device)
        #             else:
        #                 model0 = [model_ind.to(device) for model_ind in model0]
        #             models.append(model0)
        #         self.model = models
        self.device = device

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        # print(x.shape)
        if isinstance(self.model, list):
            assert len(self.model) == self.n_obj
            y = torch.zeros((x.shape[0], 0)).to(self.device)
            if self.args.train_mode not in ['ict', 'tri_mentoring', 'roma', 'iom']:
                for model in self.model:
                    res = model(x).reshape((-1, 1))
                    y = torch.cat((y, res), axis=1)
            elif self.args.train_mode == 'roma':
                for model in self.model:
                    d = model.get_distribution(x)
                    res = d.mean.reshape((-1, 1))
                    y = torch.cat((y, res), axis = 1)
            elif self.args.train_mode == 'iom':
                for models in self.model:
                    rep = models['RepModel']
                    model = models['ForwardModel']
                    x_rep = rep(x)
                    x_rep = x_rep / (torch.sqrt(torch.sum(x_rep ** 2, dim = -1, keepdim=True) + 1e-6) + 1e-6)
                    res = model(x_rep).reshape((-1, 1))
                    y = torch.cat((y, res), axis = 1)
            else:
                for model_list in self.model:
                    preds = [] 
                    for model in model_list:
                        preds.append(model(x).reshape(-1, 1))
                    res = torch.mean(torch.cat(preds, dim=1), axis=1).reshape((-1, 1))
                    # assert 0, res.shape
                    y = torch.cat((y, res), axis=1)
                    # assert 0, (res, y)
        else:
            y = self.model(x)
        # assert 0, (y, y.shape)
        out['F'] = y.detach().cpu().numpy()