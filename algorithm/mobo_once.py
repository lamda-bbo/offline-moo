import numpy as np
from datetime import datetime
import torch
from torch import Tensor
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
from botorch import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize, normalize
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from numpy import ndarray
from utils import get_N_nondominated_index, base_path
import os 



tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

class MOBO_Once:
    def __init__(self, X_init: Tensor, Y_init: Tensor, 
                 ref_point: Tensor,  bounds: Tensor, 
                #  args,
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

        # t1 = self.t1
        # t2 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nData preprocess Ended!\n' + f'Now time: {t2}' + '\n' + f'Time for data preprocessing: {t2 - t1}')

        # t3 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nBegin to fit GP model!\n' + f'Now time: {t3}')

        model = self._get_model(self.X_init, self.Y_init)

        # t4 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nGP model has been fitted!\n' + f'Now time: {t4}' + '\n' + f'Time for Model Training: {t4 - t3}')

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        with torch.no_grad():
            pred = model.posterior(normalize(self.X_init, self.bounds)).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.zeros_like(self.ref_point, device=self.ref_point.device, dtype=self.ref_point.dtype),
            Y=pred,
        )
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            X_baseline=normalize(self.X_init, self.bounds),
            partitioning=partitioning,
            sampler=sampler,
        )

        # t5 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nBegin to optimize the acquisition function!\n' + f'Now time: {t5}')

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.output_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 10, "maxiter": 200},
            sequential=True,
        )

        # t6 = datetime.now()
        # with open(self.time_file, 'a') as f:
        #     f.write('\n\nThe acquisition function has been optimized!\n' + f'Now time: {t6}' + '\n' + f'Time for Solutions searching: {t6 - t5}')

        candidates = unnormalize(candidates.detach(), bounds=self.bounds)
        return candidates.cpu().numpy()


if __name__ == '__main__':
    X_init = torch.rand((1000,7))
    Y_init = torch.rand(1000,3)
    ref_point = torch.zeros(3)
    train_gp_data_size = 32
    bounds_ = torch.zeros((2,7))
    bounds_[1] = 1.0
    mobo_once = MOBO_Once(X_init, Y_init, ref_point, train_gp_data_size, bounds_)
    print(mobo_once.run())
