import numpy as np
import torch
from torch import Tensor
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
from botorch import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import sample_simplex
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement, qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from numpy import ndarray
from utils import get_N_nondominated_index

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

class MOBO_ParEGO_Once:
    def __init__(self, X_init: Tensor, Y_init: Tensor, 
                 ref_point: Tensor,  bounds: Tensor, 
                 train_gp_data_size: int = 256, 
                 output_size: int = 256, negate=True) -> None:
        """
            args: 
                X_init: '(N,D)' data of decision variable.
                Y_init: '(N,m)' data of objective values.
                ref_point : '(m,)' reference point.
                train_gp_data_size: Size of data for fitting GP.
                bounds: '(2, D)' bounds of decision variable.
                output_size: Size of data for evluating once.
        """
        self.dim = X_init.shape[1]
        self.num_obj = Y_init.shape[1]
        self.X_init, self.Y_init = self._sample_data(X_init.detach().cpu().numpy(), Y_init.detach().cpu().numpy(), train_gp_data_size)
        self.bounds = bounds.to(**tkwargs)
        self.ref_point = ref_point.to(**tkwargs)
        self.X_init, self.Y_init = (self.X_init.to(**tkwargs), self.Y_init.to(**tkwargs))
        if negate:
            self.Y_init *= -1
            self.ref_point *= -1
        self.output_size = output_size

    
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

    def _get_model(self, train_x, train_obj):
        train_x = normalize(train_x, self.bounds)
        model = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def run(self) -> ndarray:
        """
            return: 
                 ret: (output_size, D) data decision variable with one BO iteration with output_size batches.
        """
        MC_SAMPLES = 128
        RAW_SAMPLES = 256
        NUM_RESTARTS = 10
        standard_bounds = torch.zeros(2, self.dim, **tkwargs)
        standard_bounds[1] = 1
        model = self._get_model(self.X_init, self.Y_init)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        with torch.no_grad():
            pred = model.posterior(normalize(self.X_init, self.bounds)).mean
        acq_func_list = []
        for _ in range(self.output_size):
            weights = sample_simplex(self.num_obj, **tkwargs).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )
            acq_func = qExpectedImprovement(
                model=model, 
                objective=objective,
                X_baseline=normalize(self.X_init, self.bounds),
                best_f=objective(self.Y_init).max(),
                sampler=sampler,
                prune_baseline=True
            )
            acq_func_list.append(acq_func)

        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=standard_bounds,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 10, "maxiter": 200},
        )
        candidates = unnormalize(candidates.detach(), bounds=self.bounds)
        return candidates.cpu().numpy()


if __name__ == '__main__':
    X_init = torch.rand((1000,7))
    Y_init = torch.rand(1000,2)
    ref_point = torch.zeros(3)
    train_gp_data_size = 32
    bounds_ = torch.zeros((2,7))
    bounds_[1] = 1.0
    mobo_once = MOBO_ParEGO_Once(X_init, Y_init, ref_point, bounds_, train_gp_data_size)
    print(mobo_once.run())
