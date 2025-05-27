import os
import sys
import time 

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from types import SimpleNamespace

import numpy as np
from src.evaluator import Evaluator
from src.utils.args_parser import parse_args

args = SimpleNamespace(
    **{
        "n_cpu": 16,
        "placer": "gg",  # GG in our paper
        "benchmark": "ispd2005/adaptec1",  # choose which placement benchmark,
    }
)

# Read config (i.e. benchmark, placer)
args = parse_args(args)

# Instantiate the evaluator
evaluator = Evaluator(args)

# Read problem metadata
dim: int = evaluator.n_dim
xl: np.ndarray = evaluator.xl
xu: np.ndarray = evaluator.xu
assert len(xl) == len(xu) == dim

batch_size = 10
x = np.random.uniform(low=xl, high=xu, size=(batch_size, dim))
# x = np.random.rand(batch_size, dim) * (xl - xu) + xu
t0 = time.time()
res, macor_pos = evaluator.evaluate(x)
# print(np.max(hpwl), np.min(hpwl), np.mean(hpwl))
print(res)

print(time.time() - t0)
exit()

from off_moo_bench.problem.base import BaseProblem


class BBOPlacement(BaseProblem):
    def __init__(self, benchmark_name):
        args = SimpleNamespace(
            **{
                "n_cpu": 128,
                "placer": "gg",  # GG in our paper
                "benchmark": benchmark_name,  # choose which placement benchmark
            }
        )
        args = parse_args(args)
        self.evaluator = Evaluator(args)
        super().__init__(
            name=self.__class__.__name__,
            n_dim=self.evaluator.n_dim,
            n_obj=self.evaluator.n_obj,
            problem_type="continuous",
            xl=self.evaluator.xl,
            xu=self.evaluator.xu,
        )
