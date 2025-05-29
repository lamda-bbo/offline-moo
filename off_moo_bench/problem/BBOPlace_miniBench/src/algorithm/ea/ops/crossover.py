import numpy as np
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.ux import UX

###################################################################
#  Grid Guide crossover
###################################################################


class GGUniformCrossover(UX):
    def __init__(self, args):
        super(GGUniformCrossover, self).__init__()
        self.args = args


###################################################################
#  SP crossover
###################################################################


class SPOrderCrossover(OrderCrossover):
    def __init__(self, args):
        super(SPOrderCrossover, self).__init__(shift=False)
        self.args = args

    def _do(self, problem, X, **kwargs):
        _, _, n_var = X.shape
        node_cnt = n_var // 2
        X1 = X[:, :, :node_cnt]
        X2 = X[:, :, node_cnt:]

        X1 = super(SPOrderCrossover, self)._do(problem=problem, X=X1)
        X2 = super(SPOrderCrossover, self)._do(problem=problem, X=X2)

        X = np.concatenate([X1, X2], axis=-1)
        return X
