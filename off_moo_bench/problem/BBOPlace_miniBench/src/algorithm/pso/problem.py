import numpy as np
from pymoo.core.problem import Problem

from bboplace_utils.debug import *


class PlacementProblem(Problem):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        n_var = evaluator.n_dim
        xl = evaluator.xl
        xu = evaluator.xu
        super().__init__(n_var=n_var, xl=xl, xu=xu, n_obj=1, vtype=np.int64)

    def fitness_function(self, x):
        return self.evaluator.evaluate(x)

    def get_problem_dict(self):
        return {
            "fitness_function": self.fitness_function,
            "ndim_problem": self.n_var,
            "upper_boundary": self.xu,
            "lower_boundary": self.xl,
        }
