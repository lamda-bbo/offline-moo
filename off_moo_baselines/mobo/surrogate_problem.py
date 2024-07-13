import numpy as np 
import torch 
from pymoo.core.problem import Problem 

from off_moo_baselines.mobo.mobo_utils import tkwargs

class LCB_Perm_Problem(Problem):
    def __init__(self, n_var, n_obj, model):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.model = model

    def _get_acq_value(self, X, model):
        X = torch.tensor(X)
        with torch.no_grad():
            X = X.to(**tkwargs)
            posterior = model.posterior(X)
            mean = posterior.mean
            var = posterior.variance
            return (mean - 0.2*(torch.sqrt(var))).detach().cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._get_acq_value(x, self.model)
        
class LCB_Seq_Problem(Problem):
    def __init__(self, n_var, n_obj, model, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj,
                         xl = xl,
                         xu = xu,
                         )
        self.model = model

    def _get_acq_value(self, X, model):
        X = torch.tensor(X)
        with torch.no_grad():
            X = X.to(**tkwargs)
            posterior = model.posterior(X)
            mean = posterior.mean
            var = posterior.variance
            return (mean - 0.2*(torch.sqrt(var))).detach().cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._get_acq_value(x, self.model)
        
