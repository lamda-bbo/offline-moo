import logging
import math
import os
import pickle
import sys
import time
from collections import deque
from copy import deepcopy

import gpytorch
import numpy as np
import psutil
import ray
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.kernels import Kernel
from torch import Tensor

try:
    from botorch import fit_gpytorch_mll
except:
    from botorch import fit_gpytorch_model as fit_gpytorch_mll

from botorch.acquisition import ExpectedImprovement
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling, PermutationRandomSampling
from pymoo.optimize import minimize
from src.algorithm.ea.ops import REGISTRY as OPS_REGISTRY
from src.algorithm.sampling import REGISTRY as SAMPLE_REGISTRY

from utils.constant import INF
from utils.data_utils import FeatureCache
from utils.debug import *
from utils.random_parser import set_state

from ..basic_algo import BasicAlgo
from .acqf_problem import (
    GridGuideAcquisitionFuncProblem,
    SequencePairAcquisitionFuncProblem,
)
from .kernel import CombinedOrderKernel, OrderKernel, TransformedCategorical


class _SPRandomSampling(PermutationRandomSampling):
    def __init__(self, args) -> None:
        super(_SPRandomSampling, self).__init__()
        self.args = args

    def _do(self, problem, n_samples, **kwargs):
        sub_n_var = problem.n_var // 2
        sub_xl = problem.xl[:sub_n_var]
        sub_xu = problem.xu[:sub_n_var]
        sub_problem = deepcopy(problem)
        sub_problem.n_var = sub_n_var
        sub_problem.xl = sub_xl
        sub_problem.xu = sub_xu
        X1 = PermutationRandomSampling._do(
            self, problem=sub_problem, n_samples=n_samples
        )
        X2 = PermutationRandomSampling._do(
            self, problem=sub_problem, n_samples=n_samples
        )
        X = np.concatenate([X1, X2], axis=1)
        return X


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class IntegerRandomSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.around(X).astype(int)


class BO(BasicAlgo):
    def __init__(self, args, evaluator, logger):
        super(BO, self).__init__(args=args, evaluator=evaluator, logger=logger)

        self.n_var = evaluator.node_cnt

        self.evaluator = evaluator
        self.placer_type = args.placer
        self.n_init = self.args.n_init
        self.batch_size = self.args.batch_size

        self.acqf_type = self.args.acqf_type
        self.kernel_type = self.args.kernel_type

        if args.placer == "gg":
            self.kernel_type = "tc"
            self.acqf_type = "LCB_lower_bound"
        elif args.placer == "sp":
            self.kernel_type = "comb_order"
            self.acqf_type = "LCB_lower_bound"

    def _init_model(self, train_X: Tensor, train_Y: Tensor, state_dict=None):
        assert not torch.isnan(train_X).any() and not torch.isinf(train_X).any()
        assert not torch.isnan(train_Y).any() and not torch.isinf(train_Y).any()
        NOISE_SE = 0.5
        train_yvar = torch.tensor(NOISE_SE**2).to(**tkwargs)

        Y_var = torch.full_like(train_Y, 0.01).to(**tkwargs)
        kernel = self._get_kernel(self.kernel_type)
        model = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel).to(**tkwargs)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**tkwargs)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(**tkwargs)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        fit_gpytorch_mll(mll)
        return model

    def _init_samples(self, n_samples):
        x_np = SAMPLE_REGISTRY[self.placer_type](
            self.args, self.evaluator, self._record_results
        ).do(n_samples)
        x = torch.from_numpy(x_np).to(**tkwargs)

        res, macro_pos_all = self._get_observations(x, return_macro_pos=True)
        hpwl = res["hpwl"]
        self._record_results(hpwl, macro_pos_all)

        if isinstance(hpwl, (np.ndarray, list)):
            hpwl = torch.Tensor(hpwl).to(**tkwargs)

        return x, hpwl.reshape(-1, 1)

    def _get_kernel(self, kernel_type):
        if kernel_type.lower() == "tc":
            kernel = TransformedCategorical()
        elif kernel_type.lower() == "comb_order":
            kernel = CombinedOrderKernel(n=self.evaluator.node_cnt)
        elif kernel_type.lower() == "default":
            kernel = None
        else:
            raise NotImplementedError
        return kernel

    def _get_acqf(self, acqf_type, model, train_X: Tensor, train_Y: Tensor):
        if acqf_type == "EI":
            AF = ExpectedImprovement(model, best_f=train_Y.max().item()).to(**tkwargs)

        elif acqf_type == "LCB_lower_bound":

            def LCB_lower_bound(X: Tensor):
                with torch.no_grad():
                    X = X.to(**tkwargs)
                    posterior = model.posterior(X)
                    mean = posterior.mean
                    var = posterior.variance
                    return mean - 0.2 * (torch.sqrt(var))

            AF = LCB_lower_bound

        else:
            raise NotImplementedError
        return AF

    def _get_observations(self, x, return_macro_pos=True):
        is_torch_tensor = isinstance(x, torch.Tensor)

        if is_torch_tensor:
            x = x.detach().cpu().numpy().astype(np.int32)

        y, macro_pos_all = self.evaluator.evaluate(x)

        return y, (macro_pos_all if return_macro_pos else None)

    def _optimize_acqf_and_get_observations(self, acqf, num_samples=1):
        if self.placer_type == "gg":
            acqf_problem = GridGuideAcquisitionFuncProblem(
                self.args.n_grid_x, self.args.n_grid_y, self.evaluator.node_cnt, acqf
            )
            algo = GA(
                pop_size=100,
                sampling=IntegerRandomSampling(),
                mutation=OPS_REGISTRY["mutation"][self.placer_type](self.args),
                crossover=OPS_REGISTRY["crossover"][self.placer_type](self.args),
                eliminate_duplicates=True,
            )

            res = minimize(
                problem=acqf_problem,
                algorithm=algo,
                termination=("n_gen", self.args.opt_acqf_iter),
                verbose=False,
            )

            proposed_X = res.pop.get("X")

        elif self.placer_type == "sp":
            acqf_problem = SequencePairAcquisitionFuncProblem(
                node_cnt=self.evaluator.node_cnt, acqf=acqf
            )
            algo = GA(
                pop_size=100,
                sampling=_SPRandomSampling(args=self.args),
                mutation=OPS_REGISTRY["mutation"][self.placer_type](self.args),
                crossover=OPS_REGISTRY["crossover"][self.placer_type](self.args),
                eliminate_duplicates=True,
            )

            res = minimize(
                problem=acqf_problem,
                algorithm=algo,
                termination=("n_gen", self.args.opt_acqf_iter),
                verbose=False,
            )

            proposed_X = res.pop.get("X")
        else:
            raise NotImplementedError
        assert len(proposed_X) >= num_samples
        if len(proposed_X) > num_samples:
            indices = np.random.choice(proposed_X.shape[0], num_samples, replace=False)
            proposed_X = proposed_X[indices, :]

        res, macro_pos_all = self._get_observations(proposed_X, return_macro_pos=True)
        hpwl = res["hpwl"]

        t_temp = time.time()
        t_eval = t_temp - self.t
        self.t_total += t_eval
        t_each_eval = t_eval / num_samples
        avg_t_each_eval = self.t_total / (self.n_eval + self.n_init)
        self.t = t_temp

        self._record_results(
            hpwl,
            macro_pos_all,
            t_each_eval=t_each_eval,
            avg_t_each_eval=avg_t_each_eval,
        )

        return torch.from_numpy(proposed_X).to(**tkwargs), torch.from_numpy(hpwl).to(
            **tkwargs
        )

    def run(self):
        self.t = time.time()

        checkpoint = self._load_checkpoint()
        if checkpoint is not None:
            self.model_state_dict = checkpoint["model_state_dict"]
            self.train_X = checkpoint["train_X"].to(**tkwargs)
            self.train_Y = checkpoint["train_Y"].to(**tkwargs)
        else:
            self.train_X, self.train_Y = self._init_samples(n_samples=self.n_init)

        if len(self.train_X) < self.n_init:
            extended_X = self._init_samples(self.n_init - len(self.train_X))
            self.train_X = torch.cat(self.train_X, extended_X)

        train_Y_tensor = deepcopy(self.train_Y)
        train_Y_tensor = (train_Y_tensor - train_Y_tensor.mean()) / (
            train_Y_tensor.std() + 1e-6
        )
        self.model = self._init_model(
            self.train_X,
            train_Y_tensor,
            state_dict=self.model_state_dict
            if hasattr(self, "model_state_dict")
            else None,
        )

        # calculate how many batch
        n_batch = math.ceil(
            (
                self.args.max_evals
                + 1
                - self.n_init * (self.args.n_sampling_repeat - 1)
                - len(self.train_X)
            )
            / self.batch_size
        )

        for i in range(1, n_batch + 1):
            t0 = time.monotonic()

            acqf = self._get_acqf(
                self.acqf_type, self.model, self.train_X, train_Y_tensor
            )
            proposed_X, proposed_Y = self._optimize_acqf_and_get_observations(
                acqf, self.batch_size
            )

            assert len(proposed_X) == self.batch_size

            self.train_X = torch.cat((self.train_X, proposed_X))
            self.train_Y = torch.cat((self.train_Y, proposed_Y.reshape(-1, 1)))

            self._save_checkpoint()

            train_Y_tensor = deepcopy(self.train_Y)
            train_Y_tensor = (train_Y_tensor - train_Y_tensor.mean()) / (
                train_Y_tensor.std() + 1e-6
            )
            self.model = self._init_model(
                self.train_X, train_Y_tensor, self.model.state_dict()
            )

            t1 = time.monotonic()

            if self.args.verbose:
                print(
                    f"\nBatch {i:>2}: best_hpwl = "
                    f"{self.best_hpwl}    "
                    f"time = {t1-t0:>4.2f}.",
                )
            else:
                print(".", end="")

    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            super()._load_checkpoint()
            checkpoint_path = os.path.join(self.args.checkpoint, "bo.pt")
            checkpoint = torch.load(checkpoint_path)
            self.start_from_checkpoint = True
        else:
            checkpoint = None
            self.start_from_checkpoint = False

        return checkpoint

    def _save_checkpoint(self):
        super()._save_checkpoint()

        model_file = os.path.join(self.checkpoint_path, "bo.pt")
        self.model = self.model.to("cpu")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "train_X": self.train_X.detach().cpu(),
            "train_Y": self.train_Y.detach().cpu(),
        }
        torch.save(obj=checkpoint, f=model_file)
        self.model = self.model.to(**tkwargs)
