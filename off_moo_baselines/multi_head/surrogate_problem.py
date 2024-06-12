from pymoo.core.problem import Problem 
from off_moo_baselines.data import tkwargs
import torch 

class MultiHeadSurrogateProblem(Problem):
    def __init__(self, n_var, n_obj, model):
        super().__init__(
            n_var = n_var, 
            n_obj = n_obj,
            xl = 0,
            xu = 1
        )
        self.model = model
        self.model.set_kwargs(**tkwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(**tkwargs)
        if isinstance(self.model, list):
            assert len(self.model) == self.n_obj
            y = torch.zeros((0, 1)).to(**tkwargs)
            for model in self.model:
                res = model(x, forward_objs=list(model.obj2head.keys()))
                y = torch.cat((y, res), axis=0)
        else:
            y = self.model(x, forward_objs=list(self.model.obj2head.keys()))
        out['F'] = y.detach().cpu().numpy()
