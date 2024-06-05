import math
import torch
import gpytorch
from gpytorch.kernels import Kernel
from torch import Tensor
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
import numpy as np
from botorch.models import FixedNoiseGP
from numpy import ndarray
from botorch import fit_gpytorch_mll
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from utils import get_N_nondominated_index

tkwargs = {
    "dtype": torch.double,
    "device": "cpu",
}


class StartFromZeroRepair(Repair):
    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]
        
        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])
        
        return X


class FeatureCache:
    def __init__(self, input_type='torch'):
        self.input_type = input_type
        self.cache = dict()

    def _get_key(self, x):
        return tuple(x.tolist())

    def push(self, x):
        feature = self.get(x)
        if feature is None:
            feature = self._featurize(x, self.input_type)
            self.cache[self._get_key(x)] = feature
        return feature

    def get(self, x):
        return self.cache.get(self._get_key(x), None)
    
    def _featurize(self, x, ret_type='torch'):
        assert ret_type in ['torch', 'numpy']
        if ret_type == 'torch':
            assert x.dim() == 1
        else:
            assert x.ndim == 1
        featurize_x = []
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                featurize_x.append(1 if x[i] > x[j] else -1)
        if ret_type == 'torch':
            featurize_x = torch.tensor(featurize_x, dtype=torch.float)
        elif ret_type == 'numpy':
            featurize_x = np.array(featurize_x, dtype=np.float64)
        else:
            assert 0
        normalizer = np.sqrt(len(x) * (len(x) - 1) / 2)
        return featurize_x / normalizer
    

feature_cache = FeatureCache()

class OrderKernel(Kernel):
    has_lengthscale = True
    """
    def _count_discordant_pairs(self, x1, x2):
        count = 0
        n = len(x1)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if the pair is discordant
                if (x1[i] < x1[j] and x2[i] > x2[j]) or (x1[i] > x1[j] and x2[i] < x2[j]):
                    count += 1
        return count
    """

    def forward(self, X, X2, **params):
        # assert 0, X.shape
        mat = torch.zeros((len(X), len(X2))).to(X)
        x1 = []
        for i in range(len(X)):
            x1.append(feature_cache.push(X[i]))
        x2 = []
        for j in range(len(X2)):
            x2.append(feature_cache.push(X2[j]))
        #mat = self._count_discordant_pairs(x1, x2)
        x1 = torch.vstack(x1)
        x2 = torch.vstack(x2)
        x1 = torch.reshape(x1, (x1.shape[0], 1, -1))
        x2 = torch.reshape(x2, (1, x2.shape[0], -1))
        x1 = torch.tile(x1, (1, x2.shape[0], 1))
        x2 = torch.tile(x2, (x1.shape[0], 1, 1))
        mat = torch.sum((x1 - x2)**2, dim=-1)
        mat = torch.exp(- self.lengthscale * mat)
        return mat


class LCB_Problem(Problem):
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
            return mean - 0.2*(torch.sqrt(var)).detach().cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._get_acq_value(x, self.model)


class MOBO_Permutation:
    def __init__(self, X_init: Tensor, Y_init: Tensor, env_name, train_gp_data_size: int = 256, output_size: int = 256) -> None:
        """
            args: 
                X_init: '(N,D)' data of decision variable.
                Y_init: '(N,m)' data of objective values.
                ref_point : '(m,)' reference point.
                train_gp_data_size: Size of data for fitting GP.
                bounds: '(2, D)' bounds of decision variable.
                output_size: Size of data for evluating once.
        """
        global num_obj_MOBO_permutation
        X_init = X_init.to(**tkwargs)
        Y_init = Y_init.to(**tkwargs)
        self.dim = X_init.shape[1]
        self.num_obj = Y_init.shape[1]
        num_obj_MOBO_permutation = self.num_obj
        self.X_init, self.Y_init = self._sample_data(X_init.detach().cpu().numpy(), Y_init.detach().cpu().numpy(), train_gp_data_size)
        self.output_size = output_size
        self.model = None
        self.env_name = env_name

    def _sample_data(self, X_init, Y_init, train_gp_data_size) -> (Tensor, Tensor):
        # fronts = NonDominatedSorting().do(Y_init, return_rank=True, n_stop_if_ranked=train_gp_data_size)[0]
        # indices_cnt = 0
        # indices_select = []
        # for front in fronts:
        #     if indices_cnt + len(front) < train_gp_data_size:
        #         indices_cnt += len(front)
        #         indices_select += [int(i) for i in front]
        #     else:
        #         idx = np.random.randint(len(front), size=(train_gp_data_size-indices_cnt, ))
        #         indices_select += [int(i) for i in front[idx]]
        #         break
        indices_select = get_N_nondominated_index(Y_init, train_gp_data_size)
        Y_init = Y_init[indices_select]
        X_init = X_init[indices_select]
        return torch.tensor(X_init), torch.tensor(Y_init)

    def _get_model(self, train_X: Tensor, train_Y: Tensor):
        models = []
        for i in range(train_Y.shape[-1]):
            kernel = OrderKernel().to(**tkwargs)
            train_y = train_Y[..., i : i + 1]
            train_yvar = torch.full_like(train_y, 0.01 ** 2)
            models.append(
                FixedNoiseGP(train_X, train_y, train_yvar, covar_module=kernel)
            )
        model = ModelListGP(*models)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**tkwargs)
        mll = SumMarginalLogLikelihood(likelihood, model).to(**tkwargs)
        fit_gpytorch_mll(mll)
        return model
        
    def run(self) -> ndarray:
        model = self._get_model(self.X_init, self.Y_init)
        problem = LCB_Problem(self.dim, self.num_obj, model)
        print('----nsag2,solving...-----')
        _algo = NSGA2(
            pop_size=self.output_size,
            sampling=PermutationRandomSampling(),
            mutation=InversionMutation(),
            crossover=OrderCrossover(),
            repair=StartFromZeroRepair() if self.env_name in ['mo_tsp', 'mo_cvrp'] else None,
            eliminate_duplicates=True,)
        res = minimize(problem=problem, algorithm=_algo, termination=('n_gen', 500), verbose=True)
        x = res.pop.get('X')
        return x
    

if __name__ == '__main__':
    num_samples = 1000
    perm_length = 10
    all_samples = []
    while len(all_samples) < num_samples:
        random_permutation = torch.randperm(perm_length)
        all_samples.append(random_permutation.tolist())
    unique_samples = torch.unique(torch.tensor(all_samples), dim=0)
    train_x = unique_samples
    train_y = torch.rand((num_samples,3))
    solver = MOBO_Permutation(train_x, train_y, train_gp_data_size=256)
    print(solver.run())