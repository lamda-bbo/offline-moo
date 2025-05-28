import numpy as np
from pymoo.core.problem import Problem


class PlacementProblem(Problem):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        n_var = evaluator.n_dim
        xl = evaluator.xl
        xu = evaluator.xu
        super().__init__(n_var=n_var, xl=xl, xu=xu, n_obj=1, vtype=np.int64)

    def _evaluate(self, x, out, *args, **kwargs):
        y, macro_pos = self.evaluator.evaluate(x)

        out["F"] = y["hpwl"]
        out["macro_pos"] = macro_pos


class MOPlacementProblem(Problem):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        n_var = evaluator.n_dim
        xl = evaluator.xl
        xu = evaluator.xu
        super().__init__(n_var=n_var, xl=xl, xu=xu, n_obj=3, vtype=np.int64)

    def _evaluate(self, x, out, *args, **kwargs):
        y, macro_pos = self.evaluator.evaluate(x)

        out["F"] = np.concatenate([y["hpwl"].reshape(-1, 1), 
                                   y["congestion"].reshape(-1, 1), 
                                   y["regularity"].reshape(-1, 1)],
                                   axis=1)
        out["macro_pos"] = macro_pos
