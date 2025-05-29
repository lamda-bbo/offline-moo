import os

import numpy as np
from pymoo.core.callback import Callback

_all_X = None 
_all_F = None 


class HistoryCallback(Callback):
    def __init__(self, save_path="./nsgaii_results") -> None:
        super().__init__()
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def notify(self, algorithm):
        global _all_X, _all_F
        pop = algorithm.pop

        X = pop.get("X")
        F = pop.get("F")

        if X is not None and F is not None:
            if _all_X is None:
                _all_X = X.copy()
                _all_F = F.copy()
            else:
                _all_X = np.vstack([_all_X, X])
                _all_F = np.vstack([_all_F, F])

            np.save(os.path.join(self.save_path, "all_X.npy"), _all_X)
            np.save(os.path.join(self.save_path, "all_F.npy"), _all_F)
        
        np.save(os.path.join(self.save_path, "all_X.npy"), _all_X)
        np.save(os.path.join(self.save_path, "all_F.npy"), _all_F)
