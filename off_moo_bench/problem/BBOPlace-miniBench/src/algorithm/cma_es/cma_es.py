import os
import pickle
import time

import cma
import numpy as np
from src.algorithm.sampling import REGISTRY as SAMPLE_REGISTRY

from utils.constant import INF
from utils.debug import *

from ..basic_algo import BasicAlgo


class CMAES(BasicAlgo):
    def __init__(self, args, evaluator, logger):
        super(CMAES, self).__init__(args=args, evaluator=evaluator, logger=logger)
        self.node_cnt = evaluator.node_cnt
        self.best_hpwl = INF

        self.xl = evaluator.xl
        self.xu = evaluator.xu

    def run(self):
        self.t = time.time()

        checkpoint = self._load_checkpoint()
        if checkpoint is not None:
            self.cmaes = checkpoint["obj"]
        else:
            x_init = SAMPLE_REGISTRY[self.args.placer](
                self.args, self.evaluator, self._record_results
            ).do(1)
            self.cmaes = cma.CMAEvolutionStrategy(
                x_init,
                self.args.sigma,
                {
                    "popsize": self.args.pop_size,
                    "bounds": [self.xl, self.xu],
                    "seed": self.args.seed,
                },
            )

        def round_to_discrete(x):
            return np.round(x).astype(int)

        while self.n_eval < self.args.max_evals:
            population = self.cmaes.ask()

            if self.args.placer == "gg":
                processed_population = [round_to_discrete(x) for x in population]
            elif self.args.placer == "sp":
                raise ValueError("CMA-ES is not supported for SP")

            res, macro_pos_all = self.evaluator.evaluate(processed_population)
            fitness = res["hpwl"]

            t_temp = time.time()
            t_eval = t_temp - self.t
            self.t_total += t_eval
            t_each_eval = t_eval / self.args.pop_size
            avg_t_each_eval = self.t_total / (self.n_eval + self.args.pop_size)
            self.t = t_temp

            self._record_results(
                fitness,
                macro_pos_all,
                t_each_eval=t_each_eval,
                avg_t_each_eval=avg_t_each_eval,
            )

            self._save_checkpoint()

    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            super()._load_checkpoint()
            with open(os.path.join(self.args.checkpoint, "cma_es.pkl"), "rb") as f:
                checkpoint = pickle.load(f)
                self.start_from_checkpoint = True
        else:
            checkpoint = None
            self.start_from_checkpoint = False

        return checkpoint

    def _save_checkpoint(self):
        super()._save_checkpoint()

        with open(os.path.join(self.checkpoint_path, "cma_es.pkl"), "wb") as f:
            pickle.dump(
                {
                    "obj": self.cmaes,
                },
                file=f,
            )
