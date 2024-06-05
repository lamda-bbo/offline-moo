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
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
# from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from utils import get_N_nondominated_index, base_path
import os 
from datetime import datetime

tkwargs = {
    "dtype": torch.double,
    "device": 'cpu',
}

class RoundingRepair(Repair):

    def __init__(self, **kwargs) -> None:
        """

        Returns
        -------
        object
        """
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        return np.around(X).astype(int)
    
class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.around(X).astype(int)

class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).float()
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).to(**tkwargs)
        return k_cat.to(**tkwargs)

class TransformedCategorical(CategoricalOverlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        # expand x1 and x2 to calc hamming distance
        M1_expanded = x1.unsqueeze(2)
        M2_expanded = x2.unsqueeze(1)

        # calc hamming distance
        diff = (M1_expanded != M2_expanded)

        # (# batch, # batch)
        diff1 = diff
        # diff1 = torch.sum(diff, dim=2)
        # assert 0, (diff.shape, diff1.shape, x1.shape, x2.shape)
        def rbf(d, ard):
            if ard:
                return torch.exp(-torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
            else:
                return torch.exp(-self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard):
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diag(k_cat).to(**tkwargs)
        return k_cat.to(**tkwargs)

class LCB_Problem(Problem):
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


class MOBO_Sequence:
    def __init__(self, X_init: Tensor, Y_init: Tensor,  xl = None, xu = None, train_gp_data_size: int = 256, output_size: int = 256) -> None:
        """
            args: 
                X_init: '(N,D)' data of decision variable.
                Y_init: '(N,m)' data of objective values.
                xl:
                xu:
                ref_point : '(m,)' reference point.
                train_gp_data_size: Size of data for fitting GP.
                bounds: '(2, D)' bounds of decision variable.
                output_size: Size of data for evluating once.
        """
        global num_obj_MOBO_sequence
        X_init = X_init.to(**tkwargs)
        Y_init = Y_init.to(**tkwargs)
        self.dim = X_init.shape[1]
        self.num_obj = Y_init.shape[1]
        num_obj_MOBO_sequence = self.num_obj
        self.X_init, self.Y_init = self._sample_data(X_init.detach().cpu().numpy(), Y_init.detach().cpu().numpy(), train_gp_data_size)
        self.output_size = output_size
        self.model = None
        self.xu = xu
        self.xl = xl
        # self.t1 = args.t1
        # time_file = f'{args.env_name}-MOBO-{args.train_mode}.txt'
        # self.time_file = os.path.join(base_path, 'time_record', time_file)

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
        return torch.tensor(X_init).to(**tkwargs), torch.tensor(Y_init).to(**tkwargs)

    def _get_model(self, train_X: Tensor, train_Y: Tensor):
        Y_var = torch.full_like(train_Y, 0.01).to(**tkwargs)
        kernel = TransformedCategorical().to(**tkwargs)
        model = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel).to(**tkwargs)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**tkwargs)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(**tkwargs)
        fit_gpytorch_mll(mll)
        return model
        
    def run(self) -> ndarray:

        # t1 = self.t1
        # t2 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nData preprocess Ended!\n' + f'Now time: {t2}' + '\n' + f'Time for data preprocessing: {t2 - t1}')

        # t3 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nBegin to fit GP model!\n' + f'Now time: {t3}')


        model = self._get_model(self.X_init, self.Y_init).to(**tkwargs)

        # t4 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nGP model has been fitted!\n' + f'Now time: {t4}' + '\n' + f'Time for Model Training: {t4 - t3}')

        problem = LCB_Problem(self.dim, self.num_obj, model, self.xl, self.xu)
        print('----nsag2,solving...-----')

        # t5 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nBegin to optimize the acquisition function!\n' + f'Now time: {t5}')

        try:
            from pymoo.factory import get_crossover, get_mutation, get_sampling
            _algo = NSGA2(pop_size=self.output_size, sampling=get_sampling('int_random'),
                            crossover=get_crossover(name='int_sbx', prob=1.0, eta=3.0),
                            mutation=get_mutation(name='int_pm', prob=1.0, eta=3.0),
                            eliminate_duplicates=True)
        except:
            from off_moo_bench.problem.mo_nas import get_genetic_operator
            sampling, crossover, mutation = get_genetic_operator()
            _algo = NSGA2(pop_size=self.output_size, sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            eliminate_duplicates=True)
        res = minimize(problem=problem, algorithm=_algo, termination=('n_gen', 100), verbose=True)
        
        # t6 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nThe acquisition function has been optimized!\n' + f'Now time: {t6}' + '\n' + f'Time for Solutions searching: {t6 - t5}')
        
        x = res.pop.get('X')
        return x

if __name__ == '__main__':
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(parent_dir)
    from utils import read_data, read_raw_data
    x_np, y_np, _ = read_data(env_name='regex', filter_type='best', return_rank=False)
    print(y_np)

    print('after normalization......')
    x_raw, y_raw, _ = read_raw_data(env_name='regex', filter_type='best', return_rank=False)
    y_np = (y_np - np.min(y_raw, axis=0)) / (np.max(y_raw, axis=0) - np.min(y_raw, axis=0))
    print(y_np)

    train_x = torch.from_numpy(x_np).to(**tkwargs)
    train_y = torch.from_numpy(y_np).to(**tkwargs)
    from off_moo_bench.problem import get_problem
    _problem = get_problem('rfp')
    _xl, _xu = _problem.xl, _problem.xu
    solver = MOBO_Sequence(train_x, train_y, _xl, _xu, train_gp_data_size=256)
    print(solver.run())
