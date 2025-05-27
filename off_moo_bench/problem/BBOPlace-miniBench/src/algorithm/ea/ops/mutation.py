import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.mutation.inversion import InversionMutation

###################################################################
#  Grid Guide mutation
###################################################################


class GGShuffleMutation(Mutation):
    def __init__(self, args):
        self.args = args
        # super(GGShuffleMutation, self).__init__(prob=1, prob_var=None)
        super(GGShuffleMutation, self).__init__()

    def _do(self, problem, X, **kwargs):
        node_cnt = X.shape[1] // 2
        _X = X.copy()
        for id in range(X.shape[0]):
            chosen_idx = np.random.choice(list(range(node_cnt)), size=4, replace=False)
            shuffled_idx = chosen_idx.copy()
            np.random.shuffle(shuffled_idx)
            for origin_idx, target_idx in zip(chosen_idx, shuffled_idx):
                _X[id][origin_idx], _X[id][origin_idx + node_cnt] = (
                    X[id][target_idx],
                    X[id][target_idx + node_cnt],
                )

        return _X


###################################################################
#  SP mutation
###################################################################


class SPInversionMutation(InversionMutation):
    def __init__(self, args):
        super(SPInversionMutation, self).__init__(prob=1.0)
        self.args = args

    def _do(self, problem, X, **kwargs):
        node_cnt = X.shape[1] // 2
        X1 = X[:, :node_cnt]
        X2 = X[:, node_cnt:]
        assert X1.shape == X2.shape
        X1 = super(SPInversionMutation, self)._do(problem, X1)
        X2 = super(SPInversionMutation, self)._do(problem, X2)
        X = np.concatenate([X1, X2], axis=1)
        return X
