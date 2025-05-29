import os
import pickle
import time

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.population import Population
from pymoo.optimize import minimize
from src.algorithm.sampling import REGISTRY as SAMPLE_REGISTRY

from bboplace_utils.constant import INF
from bboplace_utils.debug import *

from ..basic_algo import BasicAlgo
from .ops import REGISTRY as OPS_REGISTRY
from .problem import PlacementProblem


class EA(BasicAlgo):
    def __init__(self, args, evaluator, logger):
        super(EA, self).__init__(args=args, evaluator=evaluator, logger=logger)
        self.node_cnt = evaluator.node_cnt
        self.best_hpwl = INF

        self.problem = PlacementProblem(evaluator=evaluator)

    def run(self):
        checkpoint = self._load_checkpoint()
        if checkpoint is not None:
            assert len(checkpoint["population"]) == self.args.n_population
            sampling = Population.new(
                X=checkpoint["population"], F=checkpoint["fitness"]
            )
            algo_n_gen = checkpoint["n_gen"] - 1
        else:
            x = SAMPLE_REGISTRY[self.args.placer](
                self.args, self.evaluator, self._record_results
            ).do(self.args.n_population)
            sampling = Population.new(X=x)
            algo_n_gen = 0

        self._algo = GA(
            pop_size=self.args.n_population,
            sampling=sampling,
            crossover=OPS_REGISTRY["crossover"][self.args.placer](self.args),
            mutation=OPS_REGISTRY["mutation"][self.args.placer](self.args),
            callback=self._save_callback,
            eliminate_duplicates=True,
        )

        self.t = time.time()

        max_n_gen = (
            self.args.max_evals // self.args.n_population
            - self.args.n_sampling_repeat
            + 1
        )
        res = minimize(
            problem=self.problem,
            algorithm=self._algo,
            termination=("n_gen", max_n_gen - algo_n_gen),
            verbose=True,
        )

    def _save_callback(self, algo):
        # compute time
        t_temp = time.time()
        t_eval = t_temp - self.t
        self.t_total += t_eval
        t_each_eval = t_eval / self.args.n_population
        avg_t_each_eval = self.t_total / (self.n_eval + self.args.n_population * 2)
        self.t = t_temp

        macro_pos_all = algo.pop.get("macro_pos")
        hpwl = algo.pop.get("F").flatten()

        if not self.start_from_checkpoint:
            self._record_results(
                hpwl=hpwl,
                macro_pos_all=macro_pos_all,
                t_each_eval=t_each_eval,
                avg_t_each_eval=avg_t_each_eval,
            )
        else:
            self.start_from_checkpoint = False

        self._save_checkpoint(
            population=algo.pop.get("X"),
            fitness=algo.pop.get("F"),
            n_gen=self._algo.n_gen,
        )

    def _save_checkpoint(self, population, fitness, n_gen):
        super()._save_checkpoint()

        with open(os.path.join(self.checkpoint_path, "ea.pkl"), "wb") as f:
            pickle.dump(
                {"population": population, "fitness": fitness, "n_gen": n_gen}, file=f
            )

    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            super()._load_checkpoint()
            with open(os.path.join(self.args.checkpoint, "ea.pkl"), "rb") as f:
                checkpoint = pickle.load(f)
                self.start_from_checkpoint = True
        else:
            checkpoint = None
            self.start_from_checkpoint = False

        return checkpoint
