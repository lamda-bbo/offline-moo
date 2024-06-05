import numpy as np
import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import Kernel
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
from botorch.models import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE 
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize
from numpy import ndarray
from collections import OrderedDict
from utils import get_N_nondominated_index

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

class StartFromZeroRepair(Repair):
    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]
        
        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])
        
        return X


class FeatureCache:
    def __init__(self, input_type='torch', max_size=2000):
        self.input_type = input_type
        self.cache = OrderedDict()
        self.max_size = max_size

    def _get_key(self, x):
        if isinstance(x, torch.Tensor) and x.dim() > 1:
            return tuple(map(tuple, x.tolist()))
        else:
            return tuple(x.tolist())
    
    def __len__(self):
        return len(self.cache)

    def push(self, x):
        if isinstance(x, torch.Tensor) and x.dim() > 1:
            features = []
            for sample in x:
                feature = self.get(sample)
                if feature is None:
                    feature = self._featurize(sample, self.input_type)
                    self._put(self._get_key(sample), feature)
                features.append(feature)
            return torch.stack(features)
        else:
            feature = self.get(x)
            if feature is None:
                feature = self._featurize(x, self.input_type)
                self._put(self._get_key(x), feature)
            return feature

    def get(self, x):
        key = self._get_key(x)
        if key in self.cache:
            # Move the key to the end to show that it was recently accessed
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return None
    
    # LRU strategy
    def _put(self, key, value):
        if key in self.cache:
            # Update the key and move it to the end
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove the first item from the ordered dictionary
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def _featurize(self, x, ret_type='torch'):
        if x.dim() == 1:
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
        else: 
            assert x.dim() == 2  
            batch_size, num_features = x.shape
            comparison = x.unsqueeze(2) > x.unsqueeze(1)

            comparison = torch.tril(comparison, diagonal=-1) * 2 - 1
            featurize_x = comparison[comparison != 0].view(batch_size, -1)

            normalizer = torch.sqrt(torch.tensor(num_features * (num_features - 1) / 2, dtype=torch.float))
            featurize_x = featurize_x / normalizer

            return featurize_x
    
    def _featurize_before(self, x, ret_type='torch'):

        assert ret_type in ['torch', 'numpy']
        if ret_type == 'torch':
            assert x.dim() == 1 or x.dim() == 2
        else:
            assert x.ndim == 1 or x.ndim == 2

        if x.ndim == 1 or x.dim() == 1:  # Handle single sample
            x = [x]  # Wrap single sample in a list to use the batch processing loop

        featurized_batch = []

        for sample in x:
            featurize_x = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    featurize_x.append(1 if sample[i] > sample[j] else -1)

            if ret_type == 'torch':
                featurize_x = torch.tensor(featurize_x, dtype=torch.float)
            elif ret_type == 'numpy':
                featurize_x = np.array(featurize_x, dtype=np.float64)
            
            normalizer = np.sqrt(len(sample) * (len(sample) - 1) / 2)
            featurized_batch.append(featurize_x / normalizer)

        if ret_type == 'torch':
            return torch.stack(featurized_batch)
        elif ret_type == 'numpy':
            return np.stack(featurized_batch)
    

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
        # print('X.shape:', X.shape, "X2.shape:", X2.shape)
        if len(X.shape) > 2:
            assert X.shape[0] == X2.shape[0]
            batch_size = X.shape[0]

            x1 = feature_cache.push(X).to(**tkwargs)
            x2 = feature_cache.push(X2).to(**tkwargs)
            # print(len(feature_cache))

            mat = (x1.unsqueeze(2) - x2.unsqueeze(1)).pow(2).sum(dim=-1)
            mat = torch.exp(-self.lengthscale * mat)

            mat = mat.view(batch_size, -1, mat.shape[-1])
            # print('mat:', mat.shape)
            return mat
        else:
            mat = torch.zeros((len(X), len(X2))).to(X)
            x1 = []
            for i in range(len(X)):
                x1.append(feature_cache.push(X[i]))
            x2 = []
            for j in range(len(X2)):
                x2.append(feature_cache.push(X2[j]))
            # print(len(feature_cache))
            #mat = self._count_discordant_pairs(x1, x2)
            x1 = torch.vstack(x1).to(**tkwargs)
            x2 = torch.vstack(x2).to(**tkwargs)
            x1 = torch.reshape(x1, (x1.shape[0], 1, -1))
            x2 = torch.reshape(x2, (1, x2.shape[0], -1))
            x1 = torch.tile(x1, (1, x2.shape[0], 1))
            x2 = torch.tile(x2, (x1.shape[0], 1, 1))
            mat = torch.sum((x1 - x2)**2, dim=-1)
            mat = torch.exp(- self.lengthscale * mat)
            # print('mat:', mat.shape)
            return mat


class AcqfListProblem(Problem):
    def __init__(self, n_var, n_obj, acqf_list):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.acqf_list = acqf_list

    def _evaluate(self, x, out, *args, **kwargs):
        if isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x).to(**tkwargs)

        y = torch.zeros(x.shape[0], 0).to(**tkwargs)
        for acqf in self.acqf_list:
            # print('x.shape = ', x.shape)
            acq_value = acqf(x.unsqueeze(1)).reshape(-1, 1) * (-1) # Negate since qEI aims at maximizing
            y = torch.cat((y, acq_value), axis=1)
            # print(y.shape)
        out["F"] = y.detach().cpu().numpy() 

class AcqfProblem(Problem):
    def __init__(self, n_var, acq_func):
        super().__init__(n_var=n_var, n_obj=1)
        self.acq_func = acq_func

    def _evaluate(self, x, out, *args, **kwargs):
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x).to(**tkwargs)
        out["F"] = self.acq_func(x.unsqueeze(1)).reshape(-1, 1).detach().cpu().numpy() * (-1)


class MOBO_ParEGO_Permutation:
    def __init__(self, X_init: Tensor, Y_init: Tensor, env_name, 
                 train_gp_data_size: int = 256, output_size: int = 256,
                 negate = True) -> None:
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
        if negate:
            self.Y_init *= -1

    
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
        """
            return: 
                 ret: (output_size, D) data decision variable with one BO iteration with output_size batches.
        """
        MC_SAMPLES = 128
        RAW_SAMPLES = 256
        NUM_RESTARTS = 10
        model = self._get_model(self.X_init, self.Y_init)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        with torch.no_grad():
            pred = model.posterior(self.X_init).mean
        acq_func_list = []
        for _ in range(self.output_size):
            weights = sample_simplex(self.num_obj, **tkwargs).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )
            acq_func = qExpectedImprovement(
                model=model, 
                q=self.output_size,
                objective=objective,
                X_baseline=self.X_init,
                best_f=objective(self.Y_init).max(),
                sampler=sampler,
                prune_baseline=True
            )
            acq_func_list.append(acq_func)

        x_all = []

        for acq_func in acq_func_list:
            problem = AcqfProblem(self.dim, acq_func)
            print('----GA, solving...----')
            # _algo = DE(
            #     pop_size=self.output_size,
            #     sampling=PermutationRandomSampling(),
            #     mutation=InversionMutation(),
            #     crossover=OrderCrossover(),
            #     repair=StartFromZeroRepair() if self.env_name.startswith(('motsp', 'mocvrp')) else None,
            #     eliminate_duplicates=True,
            #     CR=0.3,
            #     variant="DE/rand/1/bin",
            #     dither="vector",
            #     jitter=False
            # )
            _algo = GA(
                pop_size=50,
                sampling=PermutationRandomSampling(),
                mutation=InversionMutation(),
                crossover=OrderCrossover(),
                repair=StartFromZeroRepair() if self.env_name.startswith(('motsp', 'mocvrp')) else None,
                eliminate_duplicates=True
            )

            res = minimize(problem=problem, algorithm=_algo, termination=('n_gen', 500), verbose=True)
            x_all.append(res.X.reshape(1, -1))
            print(np.concatenate(x_all, axis=0).shape)
        
        x_all = np.concatenate(x_all, axis=0)
        print(x_all.shape)
        return x_all
            

        # problem = AcqfListProblem(self.dim, self.output_size, acq_func_list)

        # print('----nsag2,solving...-----')
        # callback = lambda algo: print(algo.n_iter)
        # _algo = NSGA2(
        #     pop_size=self.output_size,
        #     sampling=PermutationRandomSampling(),
        #     mutation=InversionMutation(),
        #     crossover=OrderCrossover(),
        #     repair=StartFromZeroRepair() if self.env_name.startswith(('motsp', 'mocvrp')) \
        #         else None,
        #     eliminate_duplicates=True,
        #     )
        # res = minimize(problem=problem, algorithm=_algo, termination=('n_gen', 100), verbose=True)
        # x = res.pop.get('X')
        # return x


if __name__ == '__main__':
    X_init = torch.rand((1000,7))
    Y_init = torch.rand(1000,2)
    ref_point = torch.zeros(3)
    train_gp_data_size = 32
    bounds_ = torch.zeros((2,7))
    bounds_[1] = 1.0
    mobo_once = MOBO_ParEGO_Permutation(X_init, Y_init, ref_point, bounds_, train_gp_data_size)
    print(mobo_once.run())
